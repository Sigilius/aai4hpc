"""
agents/gateway_agent.py  —  v2.0.0

GatewayAgent: thin entry/exit router. Nothing more.

v2 P2P refactor
---------------
The gateway no longer decomposes queries, synthesizes answers, or sets
delegation triggers.  Those decisions belong to the agents themselves.

What gateway does:
  1. Classify intent: sql | doc | predict
  2. Forward the raw user query to the entry peer — bare REQUEST
  3. The agent chain (sql → synthesizer → reflector) runs autonomously
  4. Receive the final answer back and emit TERMINATE

DA types emitted
----------------
  REQUEST   — forward to entry agent
  SYNTHESIZE — pass final answer to user
  REJECT     — entry peer unavailable
  TERMINATE  — session closed
"""
from __future__ import annotations

from core.message_schema import (
    A2AMessage,
    DialogueActType,
    UncertaintyFlag,
)
from core.shared_log import SharedLog
from agents.base_agent import BaseAgent

_MAX_HISTORY = 3


class GatewayAgent(BaseAgent):
    """
    Entry / exit point for all HPC user queries.

    Deliberately thin — all routing, decomposition, and synthesis
    decisions happen inside the agent chain, not here.
    """

    name    = "gateway"
    version = "2.0.0"

    def __init__(self, log: SharedLog, verbose: bool = False) -> None:
        super().__init__(log, verbose)
        self._history: list[dict] = []

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(self, raw_query: str, session_id: str) -> str:
        user_msg = A2AMessage(
            session_id = session_id,
            turn       = self.log.next_turn(session_id),
            sender     = "user/1.0.0",
            recipient  = self.agent_id(),
            da_type    = DialogueActType.USER_QUERY,
            content    = raw_query,
        )
        self.log.append(user_msg)
        self._print_msg(user_msg, direction="in")

        result = await self.handle(user_msg)
        answer = result.content
        answer = self._add_footnotes(answer, session_id)

        self.reply(user_msg, DialogueActType.TERMINATE, content=answer)
        self._history.append({"query": raw_query, "answer": answer})
        return answer

    # ── handle() ─────────────────────────────────────────────────────────────

    async def handle(self, msg: A2AMessage) -> A2AMessage:
        query = msg.content

        # 1. Classify intent
        label = self._classify(query)
        self._note(msg, "classify", f"intent → {label}")

        # 2. Route to entry agent — mixed sql+doc queries are handled inside
        #    sql_agent (delegates forward to doc_agent, which sends to synthesizer).
        #    Gateway stays thin: classify and dispatch only.
        entry_peer = _label_to_peer(label)
        if not self.has_peer(entry_peer):
            if entry_peer != "sql_agent" and self.has_peer("sql_agent"):
                entry_peer = "sql_agent"
                self._note(msg, "fallback", f"{label} agent unavailable → sql_agent")
            else:
                return self.reply(
                    msg, DialogueActType.REJECT,
                    content=f"The '{entry_peer}' agent is not available.",
                )

        req = self.make_request(
            entry_peer, msg,
            DialogueActType.REQUEST,
            query,
            metadata={"history": self._history_str()},
        )
        self._note(msg, "dispatch", f"→ {entry_peer}")
        peer_resp = await self.ask_peer(entry_peer, req)
        self._note(msg, "peer_response",
                   f"{entry_peer} replied DA={peer_resp.da_type.value}")

        if peer_resp.da_type == DialogueActType.REJECT:
            answer = self._format_rejection(query, peer_resp)
        else:
            answer = peer_resp.content

        return self.reply(
            msg,
            DialogueActType.SYNTHESIZE,
            content    = answer,
            confidence = peer_resp.confidence,
            flags      = peer_resp.uncertainty_flags or [],
            metadata   = {"entry_peer": entry_peer},
        )

    # ── Classification ────────────────────────────────────────────────────────

    def _classify(self, query: str) -> str:
        prompt = (
            "Classify this user query about the Fugaku supercomputer into ONE category.\n\n"
            f"Recent conversation:\n{self._history_str()}\n\n"
            "Categories:\n"
            "- predict: Use when ANY part of the query asks the prediction system to assign\n"
            "  a failure probability or risk score. The strongest indicators are:\n"
            "  'what failure probability does the predictor assign',\n"
            "  'combined probability that at least one job fails',\n"
            "  'what does the system predict for', 'what risk does the predictor assign',\n"
            "  'predict failure for each', 'failure probability for a new job'.\n"
            "  Even if the query starts with historical SQL questions (failure rates, job sizes),\n"
            "  classify as predict when the query ends by asking the predictor for a probability.\n"
            "  Also predict if the query asks for failure risk for a SPECIFIC NEW JOB:\n"
            "  'failure risk for my job', 'will it fail', 'is it safe to submit',\n"
            "  'predict runtime for a N-node job', 'how risky is a job with X nodes',\n"
            "  'what is the risk for a new user', 'risk for a new user running a job',\n"
            "  'use those averages to predict', 'predict failure risk for a new job'.\n"
            "  NOT predict: 'failure rate for Windows machines', 'historical failure rate',\n"
            "  'what percentage of jobs failed', 'failure rate by OS/user/class'\n\n"
            "- sql: use if ANY PART of the query asks for MEASUREMENTS from historical job\n"
            "  telemetry data. This includes: average energy consumption, failure rates, counts,\n"
            "  averages, trends, 'how many jobs', 'what percentage', specific user stats, job\n"
            "  duration distributions — including attributes that may not exist in the DB\n"
            "  (e.g., OS type, machine type). Always sql if asking about past data.\n"
            "  KEY RULE: If ANY PART asks for historical measurements (energy, power, fail rates,\n"
            "  counts, averages), classify as sql — even if the query ALSO has doc questions.\n"
            "  The sql_agent can handle doc sub-questions internally.\n"
            "  Signals: 'average energy', 'energy consumption', 'how much power', 'energy per job',\n"
            "  'average runtime', 'failure rate', 'how many jobs', 'sustainability report on usage'\n\n"
            "- doc: use ONLY if the query asks PURELY about COMMANDS, POLICIES, RULES, or\n"
            "  GUIDANCE and does NOT ask for ANY historical measurements or statistics.\n"
            "  Examples: walltime limits, job scripts, compiler flags, system commands,\n"
            "  'what command', 'how do I', 'what is the purpose of', 'what happens when'\n\n"
            "Priority order: predict (new-job only) > sql > doc\n"
            f"Current query: {query}\n\n"
            "Reply with exactly one word: doc, sql, or predict."
        )
        label = self._llm(prompt, model="gpt-4o-mini", max_tokens=5).lower().strip()
        if label not in ("doc", "sql", "predict"):
            label = "sql"
        if self.verbose:
            from rich.console import Console
            Console().print(f"  [dim][gateway] classified → [bold]{label}[/bold][/dim]")
        return label

    # ── Rejection formatting ──────────────────────────────────────────────────

    def _format_rejection(self, query: str, peer_resp: A2AMessage) -> str:
        domain = peer_resp.metadata.get("domain_info", peer_resp.content)
        prompt = (
            "You are an HPC analyst assistant for the Fugaku supercomputer.\n\n"
            f"The user asked: {query}\n\n"
            f"Investigation found:\n{domain}\n\n"
            "The query cannot be answered as asked.\n\n"
            "In 2-3 sentences:\n"
            "1. Explain clearly why this specific query doesn't apply.\n"
            "2. Suggest 1-2 alternative questions the user could ask instead.\n"
            "Be specific — use actual column names and values from the investigation."
        )
        return self._llm(prompt)

    # ── Uncertainty footnotes ─────────────────────────────────────────────────

    def _add_footnotes(self, answer: str, session_id: str) -> str:
        all_flags: set[UncertaintyFlag] = set()
        low_sample_caution = ""
        for m in self.log.get_session(session_id):
            all_flags.update(m.uncertainty_flags)
            if not low_sample_caution and m.metadata.get("low_sample_caution"):
                low_sample_caution = m.metadata["low_sample_caution"]
        if UncertaintyFlag.NOT_FOUND in all_flags:
            answer += "\n\n⚠  Note: some requested data was not found in the database."
        if UncertaintyFlag.PARTIALLY_FOUND in all_flags:
            answer += "\n\n⚠  Note: answer is based on partial data — results may be incomplete."
        if UncertaintyFlag.CONFIDENCE_LOW in all_flags:
            answer += "\n\n⚠  Note: one or more values have low confidence."
        if UncertaintyFlag.NULL_VALUES in all_flags:
            answer += "\n\n⚠  Note: significant null values were detected in the retrieved data."
        if UncertaintyFlag.LOW_SAMPLE in all_flags:
            caution = low_sample_caution or (
                "This prediction is based on very limited historical data for this "
                "configuration. The failure estimate may be unreliable — proceed with caution."
            )
            answer += f"\n\n⚠  Caution: {caution}"
        if UncertaintyFlag.UNCONFIRMED_REFLECTOR in all_flags:
            answer += (
                "\n\n⚠  Caution: this answer could not be fully verified — the quality "
                "checker flagged potential issues that could not be resolved. "
                "Please cross-check key values independently."
            )
        return answer

    # ── History ───────────────────────────────────────────────────────────────

    def _history_str(self) -> str:
        if not self._history:
            return "None"
        lines = [f"Q: {h['query']}\nA: {h['answer']}" for h in self._history[-_MAX_HISTORY:]]
        return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _label_to_peer(label: str) -> str:
    return {"predict": "pa_agent", "doc": "doc_agent"}.get(label, "sql_agent")
