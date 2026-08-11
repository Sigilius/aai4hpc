"""
agents/data_explorer_agent.py  —  v1.0.0

DataExplorerAgent: lightweight column profiler.

Called by SQLAgent (P2P REQUEST) before generating SQL when the query
involves unknown categorical columns or ambiguous data types.

Two modes:
  full_categorical_scan = True  → profile ALL VARCHAR columns in jobs
  full_categorical_scan = False → infer relevant columns from query text

No LLM required — pure DuckDB introspection.

DA types emitted:
  INFORM  — profile results returned successfully
  CAVEAT  — column found but data quality issues (high nulls, sentinel values)
  REJECT  — column not found or query yields no columns to profile
"""
from __future__ import annotations

from core.message_schema import (
    A2AMessage,
    DialogueActType,
    UncertaintyFlag,
)
from core.shared_log import SharedLog
from agents.base_agent import BaseAgent
from shared.db import get_connection


class DataExplorerAgent(BaseAgent):
    """
    Profiles the actual column distributions in the Fugaku jobs table.

    Helps SQLAgent know:
    - Exact distinct values for categorical columns (e.g. pclass, jobenv_req)
    - Numeric ranges and averages
    - Whether a column is CATEGORICAL (never use AVG/SUM on it)
    - Null rates

    Logic adapted from research/mas_system_old/agents/data_explorer_agent.py
    with DA tagging added.
    """

    name    = "data_explorer"
    version = "1.0.0"

    def __init__(self, log: SharedLog, verbose: bool = False) -> None:
        super().__init__(log, verbose)
        self._db = get_connection()
        count = self._db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        if self.verbose:
            from rich.console import Console
            Console().print(f"  [dim][data_explorer] {count:,} rows loaded[/dim]")

    # ── Entry point ───────────────────────────────────────────────────────────

    async def handle(self, msg: A2AMessage) -> A2AMessage:
        query    = msg.content
        full_cat = msg.metadata.get("full_categorical_scan", False)

        # Decide which columns to profile
        if full_cat:
            columns = self._all_categorical_columns()
        else:
            columns = self._infer_columns_from_query(query)

        if not columns:
            return self.reply(
                msg, DialogueActType.REJECT,
                content="Could not identify any relevant columns from the query.",
            )

        # Profile each column
        profiles: dict[str, dict] = {col: self._profile(col) for col in columns}
        summary  = self._format(profiles)

        # Decide DA type: CAVEAT if any column has high nulls or only sentinel values
        flags = self._detect_flags(profiles)
        da    = DialogueActType.CAVEAT if flags else DialogueActType.INFORM

        return self.reply(
            msg, da,
            content=summary,
            flags=flags,
            metadata={
                "profiles":         profiles,
                "columns_profiled": list(profiles.keys()),
                "full_scan":        full_cat,
            },
        )

    # ── Column selection ──────────────────────────────────────────────────────

    def _infer_columns_from_query(self, query: str) -> list[str]:
        """Return columns whose names appear literally in the query string."""
        all_cols  = self._all_columns()
        q_lower   = query.lower()
        return [c for c in all_cols if c.lower() in q_lower]

    def _all_columns(self) -> list[str]:
        rows = self._db.execute("DESCRIBE jobs").fetchall()
        return [r[0] for r in rows]

    def _all_categorical_columns(self) -> list[str]:
        rows = self._db.execute("DESCRIBE jobs").fetchall()
        return [
            r[0] for r in rows
            if any(t in r[1].upper() for t in ("VARCHAR", "TEXT", "CHAR"))
        ]

    # ── Column profiling ──────────────────────────────────────────────────────

    def _profile(self, col: str) -> dict:
        """
        Profile a single column.
        Returns a dict with dtype, null_count, and type-specific stats.
        Adapted verbatim from research DataExplorerAgent.
        """
        try:
            dtype_row = self._db.execute(
                "SELECT column_type FROM (DESCRIBE jobs) WHERE column_name = ?",
                [col],
            ).fetchone()
            if not dtype_row:
                return {"error": f"Column '{col}' not found in schema."}

            dtype = dtype_row[0].upper()
            total = self._db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            nulls = self._db.execute(
                f'SELECT COUNT(*) FROM jobs WHERE "{col}" IS NULL'
            ).fetchone()[0]

            profile: dict = {"dtype": dtype, "total_rows": total, "null_count": nulls}

            if any(t in dtype for t in ("VARCHAR", "TEXT", "CHAR")):
                distinct = self._db.execute(
                    f'SELECT DISTINCT "{col}" FROM jobs WHERE "{col}" IS NOT NULL LIMIT 20'
                ).fetchall()
                n_distinct = self._db.execute(
                    f'SELECT COUNT(DISTINCT "{col}") FROM jobs'
                ).fetchone()[0]
                profile["type"]       = "categorical"
                profile["distinct"]   = [r[0] for r in distinct]
                profile["n_distinct"] = n_distinct

            else:
                stats = self._db.execute(
                    f'SELECT MIN("{col}"), MAX("{col}"), AVG("{col}") FROM jobs'
                ).fetchone()
                profile["type"] = "numeric"
                profile["min"]  = stats[0]
                profile["max"]  = stats[1]
                profile["avg"]  = round(stats[2], 4) if stats[2] is not None else None

            return profile

        except Exception as exc:
            return {"error": str(exc)}

    # ── Uncertainty detection ──────────────────────────────────────────────────

    def _detect_flags(self, profiles: dict[str, dict]) -> list[UncertaintyFlag]:
        flags = []
        total = next(
            (p.get("total_rows", 0) for p in profiles.values() if "total_rows" in p),
            0,
        )
        for col, p in profiles.items():
            if "error" in p:
                continue
            null_rate = p.get("null_count", 0) / max(total, 1)
            if null_rate > 0.2:
                flags.append(UncertaintyFlag.NULL_VALUES)
                break
        return list(set(flags))

    # ── Formatting ────────────────────────────────────────────────────────────

    def _format(self, profiles: dict[str, dict]) -> str:
        """
        Format profiles as a human-readable string for injection into
        the SQLAgent's LLM context.
        Adapted verbatim from research DataExplorerAgent.
        """
        lines = ["Column profiles from the Fugaku jobs table:\n"]
        for col, p in profiles.items():
            if "error" in p:
                lines.append(f"  {col}: ERROR — {p['error']}")
                continue

            lines.append(f"  {col} ({p['dtype']})")
            lines.append(f"    nulls: {p['null_count']} / {p['total_rows']}")

            if p.get("type") == "categorical":
                lines.append(
                    "    TYPE: CATEGORICAL — do NOT use with AVG, SUM, or any "
                    "numeric operation. Never cast to numeric."
                )
                lines.append(
                    f"    distinct values ({p['n_distinct']} total): {p['distinct']}"
                )
            elif p.get("type") == "numeric":
                lines.append(
                    f"    range: {p['min']} → {p['max']},  avg: {p['avg']}"
                )
            lines.append("")

        return "\n".join(lines)
