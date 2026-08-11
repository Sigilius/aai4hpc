"""
research/unstructured_baseline/agent.py

Unstructured MAS Baseline  —  SIGDIAL 2026 ablation study.

Architecture (Panel C from the paper):
  - Same 7 roles as the Full MAS (gateway, data_explorer, sql, doc, pa,
    synthesizer, reflector).
  - Same tools: run_sql, predict_job, rag_search, column profiling.
  - Causal conversation log: append-only, temporally ordered.
  - Agents communicate via NATURAL LANGUAGE messages in the log — they can
    delegate, negotiate, request follow-up, and coordinate freely.
  - CRITICAL DIFFERENCE from Full MAS: no DA type enum, no UncertaintyFlag
    field, no DelegationTrigger enum. All coordination is expressed as plain
    prose appended to the shared log.

Ablation variable isolated vs. Full MAS:
  - The typed message schema (DA types + UncertaintyFlag fields).
  - Causal ordering IS preserved (unlike Blackboard).
  - Multi-round coordination IS present (unlike the old single-pass version).
  - Expected SIGDIAL result: moderate fact coverage (tools still work, agents
    coordinate naturally), near-zero structured uncertainty acknowledgment
    (flags are text mentions at best, cannot propagate through a typed schema).

Coordination protocol:
  Round 0 — classify which agents are needed.
  Round 1 — each active agent runs in order (data_explorer first so its column
             profiles are in the log before SQL is written, then sql, pa, doc).
             PA agent reads SQL output from the log and uses those values.
  Round 2 — coordination round: each agent re-reads the full log and decides
             whether any follow-up action is needed (extra SQL, extra doc
             lookup, delegation request). Max 1 extra action per agent.
  Round 3 — synthesizer combines all log entries; reflector reviews once.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent
_SA_DIR    = _HERE.parent / "single_agent_baseline"
_SHARED    = _HERE.parent / "shared"
_HPC_SHARED = _HERE.parent.parent / "shared"   # shared/ — same DocRetriever as Full MAS
_ANALYTICS = _HERE.parent.parent / "analytics"
_ROOT      = _HERE.parent.parent

for _p in (str(_SA_DIR), str(_SHARED), str(_HPC_SHARED), str(_ANALYTICS), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

load_dotenv(str(_ROOT / ".env"))

from tools import run_sql, predict_job, get_schema_context   # noqa: E402
from doc_retriever import DocRetriever                       # noqa: E402 — same retriever as Full MAS
from data_explorer import explore_for_query                  # noqa: E402 — same profiler as Full MAS
import mas_agents as MAS                                     # noqa: E402 — shared agent implementations

# ── Context budget (ablation fairness) ────────────────────────────────────────
# This system replayed up to 8000 chars of the causal log into every agent while
# the blackboard baseline passed narrow, hard-truncated slots — sql_context[:400]
# into the PA agent and [:1200] per section into the synthesizer. That made the
# BB-vs-UNBB comparison a contest between context budgets rather than between
# unordered shared state and an ordered causal log, which is the variable the
# ablation is supposed to isolate. On the previous run it showed up as UNBB
# spending 4x BB's tokens and scoring 48.1% FR-Multi against BB's 29.6%.
#
# These are the blackboard baseline's own constants, applied here so both systems
# carry the same amount of upstream information and differ only in how it is
# organised.
CTX_PA      = 400    # == blackboard _pa_agent:  sql_context[:400]
CTX_SECTION = 1200   # == blackboard _synthesizer: explorer/doc [:1200]

# Lazy singleton — mirrors DocAgent pattern in agents/doc_agent.py
_doc_retriever: Optional[DocRetriever] = None

def _get_doc_retriever() -> DocRetriever:
    global _doc_retriever
    if _doc_retriever is None:
        _doc_retriever = DocRetriever()
    return _doc_retriever

# ── Azure client ──────────────────────────────────────────────────────────────
_AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
_AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
_MODEL         = os.getenv("GPT_VERSION", "gpt-4o")


def _make_client() -> OpenAI:
    return OpenAI(base_url=_AZURE_ENDPOINT, api_key=_AZURE_API_KEY)


# ── Causal conversation log ───────────────────────────────────────────────────

class ConversationLog:
    """
    Append-only ordered list of plain-text messages — the only communication
    channel between agents. No typed fields; all coordination is in prose.
    """
    def __init__(self):
        self._messages: list[dict] = []

    def append(self, sender: str, content: str) -> None:
        self._messages.append({
            "sender":  sender,
            "content": content,
            "ts":      datetime.now(timezone.utc).isoformat(),
        })

    def as_context(self, max_chars: int = 8000) -> str:
        lines = [f"[{m['sender']}]: {m['content']}" for m in self._messages]
        full  = "\n\n".join(lines)
        return full[-max_chars:] if len(full) > max_chars else full

    def messages(self) -> list[dict]:
        return list(self._messages)

    def last_from(self, sender: str) -> str:
        for m in reversed(self._messages):
            if m["sender"] == sender:
                return m["content"]
        return ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_fences(sql: str) -> str:
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return sql


def _extract_floats_from_text(text: str) -> list[float]:
    """Pull numeric values out of a SQL result string."""
    return [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", text)]


# ── SQL Agent ─────────────────────────────────────────────────────────────────


def _data_explorer(log: ConversationLog, query: str, llm: OpenAI) -> None:
    """
    Profile the columns relevant to the query and append the profile to the
    causal log as plain prose. Same profiler the Full MAS DataExplorerAgent uses.

    The log preserves causal order, so the profile reliably precedes the SQL
    agent's turn — but it carries no DA type and no UncertaintyFlag, so a
    high-null-rate observation is only prose the next agent may or may not act
    on. That is the ablation variable being isolated.
    """
    profile = explore_for_query(query, llm=llm)
    if profile:
        log.append("data_explorer", profile)


def _sql_agent(
    log: ConversationLog,
    query: str,
    llm: OpenAI,
    schema_ctx: str,
    follow_up: str | None = None,
) -> None:
    """
    Run the shared MAS SQLAgent and append its text result to the causal log.

    Same code as the Structured MAS (agents/sql_agent.py), constructed with no
    peers so it decomposes and generates identically but never delegates. The
    reply's da_type and uncertainty_flags are dropped at the adapter boundary:
    what reaches the log is prose, ordered but untyped. Ordering is preserved
    (unlike Blackboard); typing is not (unlike the MAS). That pair is the
    ablation variable this configuration isolates.
    """
    question = follow_up or query
    context = log.as_context(max_chars=2 * CTX_SECTION)
    out = MAS.sql(question, context=context)
    if out.strip():
        log.append("sql_agent", out)


def _pa_agent(
    log: ConversationLog,
    query: str,
    llm: OpenAI,
) -> None:
    """Shared MAS PAAgent; upstream values reach it as ordered prose only."""
    out = MAS.pa(query, context=log.as_context(max_chars=CTX_PA))
    if out.strip():
        log.append("pa_agent", out)


def _doc_agent(
    log: ConversationLog,
    query: str,
    llm: OpenAI,
) -> None:
    """Shared MAS DocAgent; retrieval status arrives as prose."""
    out = MAS.doc(query, context=log.as_context(max_chars=CTX_SECTION))
    if out.strip():
        log.append("doc_agent", out)


def _check_for_delegation(log: ConversationLog, llm: OpenAI, schema_ctx: str) -> None:
    """
    After the initial agent round, scan the conversation for any outstanding
    delegation requests (e.g. PA asked SQL for more data) and fulfill them.
    This implements natural-language peer coordination without typed triggers.
    """
    context = log.as_context(max_chars=CTX_SECTION)

    detect_prompt = f"""\
Read this multi-agent conversation and identify any OUTSTANDING delegation requests
that have NOT yet been fulfilled.

Look for patterns like:
- "SQL agent: please look up X"
- "Note to SQL agent: ..."
- "Doc agent: could you check ..."
- "I need the SQL agent to provide ..."

Conversation:
{context}

List each outstanding request as a JSON array:
[
  {{"target": "sql_agent", "request": "the specific data needed"}},
  {{"target": "doc_agent", "request": "the specific docs needed"}}
]
If no outstanding requests exist, return: []
Return ONLY valid JSON."""

    try:
        raw = llm.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": detect_prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        ).choices[0].message.content.strip()
        data = json.loads(raw)
        requests = data if isinstance(data, list) else data.get("requests", [])
    except Exception:
        requests = []

    for req in requests[:3]:  # cap at 3 follow-ups
        target  = req.get("target", "")
        request = req.get("request", "")
        if not request:
            continue
        if target == "sql_agent":
            _sql_agent(log, request, llm, schema_ctx, follow_up=request)
        elif target == "doc_agent":
            _doc_agent(log, request, llm, follow_up=request)


# ── Synthesizer ───────────────────────────────────────────────────────────────

def _synthesizer(log: ConversationLog, query: str, llm: OpenAI) -> str:
    """Shared MAS SynthesizerAgent over the causal log."""
    return MAS.synthesize(query, log.as_context(max_chars=2 * CTX_SECTION))


def _reflector(log: ConversationLog, query: str, candidate: str, llm: OpenAI) -> str:
    """Shared MAS ReflectorAgent, single pass; no typed challenge channel back."""
    return MAS.reflect(query, candidate)


# ── Intent classifier ─────────────────────────────────────────────────────────

def _classify_agents(query: str, llm: OpenAI) -> list[str]:
    prompt = f"""\
Classify which specialist agents are needed to answer this Fugaku HPC query.

data_explorer — profiles a column before it is queried: its distinct values,
                units, data type, and null rate. Needed when the query touches a
                column whose values are uncertain, or breaks results down by a
                category.
sql_agent  — job counts, failure rates, averages, statistics, historical data trends
pa_agent   — failure risk prediction, job spec analysis, energy/runtime forecast
doc_agent  — pjsub/pjdel/pjstat commands, resource groups, system policies, how-to

Query: {query}

Return JSON: {{"agents": ["data_explorer", "sql_agent", "pa_agent", "doc_agent"]}}
Include only the agents this query actually needs."""

    _VALID = ("data_explorer", "sql_agent", "pa_agent", "doc_agent")
    try:
        raw  = llm.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        ).choices[0].message.content.strip()
        data   = json.loads(raw)
        agents = [a for a in data.get("agents", []) if a in _VALID]
        return agents if agents else ["sql_agent"]
    except Exception:
        return ["sql_agent"]


# ── Logger ────────────────────────────────────────────────────────────────────

class _Logger:
    _instance: Optional["_Logger"] = None

    def __init__(self, log_dir: str = "logs/unstructured"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = Path(log_dir) / f"unstructured_{ts}.jsonl"
        self._fh   = open(self._path, "a", buffering=1)

    @classmethod
    def get(cls) -> "_Logger":
        if cls._instance is None:
            cls._instance = _Logger()
        return cls._instance

    def log(self, event: str, **kwargs) -> None:
        self._fh.write(json.dumps({
            "event": event,
            "ts":    datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }) + "\n")


# ── Main class ────────────────────────────────────────────────────────────────

class UnstructuredMAS:
    """
    Unstructured multi-agent system for Fugaku HPC queries.

    Agents communicate through an append-only causal conversation log using
    natural language — they can delegate, negotiate, and coordinate freely.
    The only ablation vs. Full MAS: no typed DialogueAct schema,
    no UncertaintyFlag enum, no DelegationTrigger enum.

    Coordination happens in two rounds:
      Round 1: sql → pa → doc (in order; pa reads SQL output from log)
      Round 2: delegation scan — any follow-up requests are fulfilled
      Final:   synthesizer + reflector produce the answer
    """

    def __init__(self, verbose: bool = False):
        self.llm        = _make_client()
        self.verbose    = verbose
        self.logger     = _Logger.get()
        self.schema_ctx = get_schema_context()

    def run(self, query: str) -> str:
        t0 = time.perf_counter()
        self.logger.log("QUERY_START", query=query)

        log = ConversationLog()
        log.append("user", query)

        # ── Round 0: classify ─────────────────────────────────────────────────
        agents     = _classify_agents(query, self.llm)
        call_order = [a for a in ("data_explorer", "sql_agent", "pa_agent", "doc_agent")
                      if a in agents]
        self.logger.log("ROUTE", agents=call_order)
        if self.verbose:
            print(f"[Unstructured] routing → {call_order}")

        # ── Round 1: initial agent pass ───────────────────────────────────────
        for agent_name in call_order:
            if self.verbose:
                print(f"[Unstructured] round1 → {agent_name}")
            if agent_name == "data_explorer":
                _data_explorer(log, query, self.llm)
            elif agent_name == "sql_agent":
                _sql_agent(log, query, self.llm, self.schema_ctx)
            elif agent_name == "pa_agent":
                _pa_agent(log, query, self.llm)
            elif agent_name == "doc_agent":
                _doc_agent(log, query, self.llm)

        # ── Round 2: coordination — fulfill delegation requests ───────────────
        if self.verbose:
            print("[Unstructured] round2 → coordination")
        _check_for_delegation(log, self.llm, self.schema_ctx)

        # ── If PA was in the plan and SQL just added new data, re-run PA ──────
        if "pa_agent" in call_order:
            sql_msgs  = [m for m in log.messages() if m["sender"] == "sql_agent"]
            pa_msgs   = [m for m in log.messages() if m["sender"] == "pa_agent"]
            # Re-run PA only if SQL produced a second result after PA ran
            if len(sql_msgs) > 1 or (sql_msgs and pa_msgs and
               sql_msgs[-1]["ts"] > pa_msgs[-1]["ts"]):
                if self.verbose:
                    print("[Unstructured] round2 → re-running pa_agent with updated SQL")
                _pa_agent(log, query, self.llm)

        # ── Round 3: synthesize + reflect ─────────────────────────────────────
        answer = _synthesizer(log, query, self.llm)
        answer = _reflector(log, query, answer, self.llm)

        duration = time.perf_counter() - t0
        self.logger.log("FINAL_ANSWER", answer=answer, duration_s=round(duration, 2),
                        conversation_log=log.messages())
        return answer
