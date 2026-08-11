from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from core.message_schema import A2AMessage


class SharedLog:
    """
    Append-only SQLite-backed conversation log with two channels:

    1. messages          — structured A2AMessage turns (DA analysis)
    2. reasoning_traces  — free-text agent reasoning notes (debugging + SIGDIAL)

    Per-session .log files are also written to logs/narrative/ for easy reading.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._narrative_dir = Path(db_path).parent / "narrative"
        self._narrative_dir.mkdir(parents=True, exist_ok=True)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id          TEXT PRIMARY KEY,
                    session_id  TEXT NOT NULL,
                    turn        INTEGER NOT NULL,
                    sender      TEXT NOT NULL,
                    recipient   TEXT NOT NULL,
                    da_type     TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    confidence  REAL,
                    flags       TEXT,
                    payload     TEXT NOT NULL,
                    ts          TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session "
                "ON messages(session_id, turn)"
            )
            # Narrative reasoning traces — not DA-typed, just free text
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reasoning_traces (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    agent      TEXT NOT NULL,
                    step       TEXT NOT NULL,
                    note       TEXT NOT NULL,
                    ts         TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_traces "
                "ON reasoning_traces(session_id)"
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(self, msg: A2AMessage) -> None:
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO messages VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        msg.id,
                        msg.session_id,
                        msg.turn,
                        msg.sender,
                        msg.recipient,
                        msg.da_type.value,
                        msg.content,
                        msg.confidence,
                        json.dumps([f.value for f in msg.uncertainty_flags]),
                        msg.model_dump_json(),
                        msg.timestamp,
                    ),
                )
                conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> List[A2AMessage]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT payload FROM messages "
                "WHERE session_id=? ORDER BY turn ASC",
                (session_id,),
            ).fetchall()
        return [A2AMessage.model_validate_json(row[0]) for row in rows]

    def next_turn(self, session_id: str) -> int:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT MAX(turn) FROM messages WHERE session_id=?",
                (session_id,),
            ).fetchone()
        return (row[0] or 0) + 1

    # ------------------------------------------------------------------
    # Narrative reasoning log
    # ------------------------------------------------------------------

    def log_reasoning(
        self, session_id: str, agent: str, step: str, note: str
    ) -> None:
        """
        Append a free-text reasoning note from an agent.

        Args:
            session_id : current session
            agent      : versioned agent id, e.g. "sql_agent/1.0.0"
            step       : label for this reasoning step, e.g. "classify", "sql_gen", "retry"
            note       : human-readable explanation of what the agent is doing / why
        """
        ts = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT INTO reasoning_traces (session_id, agent, step, note, ts) "
                    "VALUES (?,?,?,?,?)",
                    (session_id, agent, step, note, ts),
                )
                conn.commit()
            # Also append to per-session narrative file
            log_file = self._narrative_dir / f"{session_id}.log"
            with open(log_file, "a") as f:
                f.write(f"[{ts}] {agent} | {step}\n  {note}\n\n")

    def get_reasoning(self, session_id: str) -> list[dict]:
        """Return all reasoning traces for a session as a list of dicts."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT agent, step, note, ts FROM reasoning_traces "
                "WHERE session_id=? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [{"agent": r[0], "step": r[1], "note": r[2], "ts": r[3]} for r in rows]

    def format_reasoning(self, session_id: str) -> str:
        """Return a human-readable reasoning trace for a session."""
        traces = self.get_reasoning(session_id)
        if not traces:
            return "(no reasoning traces for this session)"
        lines = [f"Reasoning trace — session {session_id}:\n"]
        for t in traces:
            lines.append(f"  [{t['agent']}] {t['step']}")
            lines.append(f"    {t['note']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Formatting for LLM context
    # ------------------------------------------------------------------

    # DA types that are pure routing — they carry no new information beyond
    # forwarding the original query. Excluding them stops agents from reading
    # the query echoed back as if it were inter-agent communication.
    _ROUTING_DAS = {"USER_QUERY", "REQUEST", "TERMINATE"}

    def format_for_llm(self, session_id: str) -> str:
        """
        Return a transcript of substantive agent outputs for LLM injection.

        Filters out pure routing turns (USER_QUERY, REQUEST, TERMINATE) whose
        content is just the forwarded query — keeping only turns where agents
        produced new information: INFORM, CAVEAT, CHALLENGE, CONFIRM, REJECT,
        SYNTHESIZE, VALIDATE.

        This ensures agents reading the log see actual peer outputs rather than
        the original query echoed back through the routing chain.
        """
        messages = self.get_session(session_id)
        if not messages:
            return "(no prior dialog in this session)"

        # Always include the original user query for context
        user_q = next(
            (m.content for m in messages if m.da_type.value == "USER_QUERY"), None
        )
        lines: list[str] = []
        if user_q:
            lines.append(f"[User query]\n{user_q}")

        for m in messages:
            if m.da_type.value in self._ROUTING_DAS:
                continue   # skip forwarding noise

            flags_str = (
                f"  flags=[{', '.join(f.value for f in m.uncertainty_flags)}]"
                if m.uncertainty_flags else ""
            )
            conf_str = (
                f"  confidence={m.confidence:.2f}" if m.confidence is not None else ""
            )
            trig_str = (
                f"  trigger={m.delegation_trigger.value}"
                if m.delegation_trigger else ""
            )
            header = (
                f"[{m.sender_name()} → {m.recipient_name()} | "
                f"DA={m.da_type.value}{conf_str}{trig_str}{flags_str}]"
            )
            lines.append(f"{header}\n{m.content[:800]}")   # cap per-turn length

        if len(lines) <= 1:
            return "(no substantive agent output yet)"
        return "\n\n".join(lines)
