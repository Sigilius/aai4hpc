"""
research/unstructured_a2a_baseline/agent.py

Natural-Language A2A MAS Baseline  —  SIGDIAL 2026 ablation study.

Architecture (Panel D from the paper):
  - Same 6 roles as Full MAS: data_explorer, sql_agent, pa_agent, doc_agent,
    synthesizer, reflector.
  - TRUE directed A2A communication: PA explicitly calls ask_peer("sql_agent", question)
    mid-execution and receives an immediate synchronous response.
  - Shared causal conversation log records all A2A message exchanges.
  - CRITICAL DIFFERENCE from Full MAS: no DialogueActType, no UncertaintyFlag,
    no DelegationTrigger. All A2A messages are plain natural-language strings.
    SQL responds "This data is not available" instead of REJECT(da_type).
    PA has no PARTIALLY_FOUND code path — it must infer uncertainty from text.

Ablation ladder:
  UN     — no A2A, no types                  (Panel A)
  BB     — shared dict, no directed A2A, no types  (Panel B)
  UN-A2A — directed A2A ✓, types ✗           (Panel D — this system)
  MAS    — directed A2A ✓, types ✓           (Panel C)

Isolated variable (UN-A2A vs MAS):
  Typed schema (REJECT/INFORM/CAVEAT + UncertaintyFlag + DelegationTrigger)
  vs natural-language string responses over the same A2A peer graph.

Expected result:
  FC  ≈ MAS   — same targeted A2A sub-queries enable PA to fetch nnumr before predicting
  UAA < MAS   — plain "not available" text is less reliable than typed REJECT propagation;
                LLM may still extrapolate from adjacent data (billing from node-hours)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv
from openai import OpenAI

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).parent
_SA_DIR     = _HERE.parent / "single_agent_baseline"
_SHARED     = _HERE.parent / "shared"
_HPC_SHARED = _HERE.parent.parent / "shared"
_ANALYTICS  = _HERE.parent.parent / "analytics"
_ROOT       = _HERE.parent.parent

for _p in (str(_SA_DIR), str(_SHARED), str(_HPC_SHARED), str(_ANALYTICS), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

load_dotenv(str(_ROOT / ".env"))

from tools import run_sql, predict_job, get_schema_context   # noqa: E402
from doc_retriever import DocRetriever                       # noqa: E402
from data_explorer import explore_for_query                  # noqa: E402 — same profiler as Full MAS
import mas_agents as MAS                                     # noqa: E402 — shared agent implementations

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
    Append-only causally-ordered message log.
    Records both standalone agent outputs and directed A2A exchanges.
    Messages are plain text — no typed fields.
    """
    def __init__(self):
        self._messages: list[dict] = []

    def append(self, sender: str, content: str) -> None:
        self._messages.append({
            "sender":  sender,
            "content": content,
            "ts":      datetime.now(timezone.utc).isoformat(),
        })

    def append_a2a(self, from_agent: str, to_agent: str,
                   request: str, response: str) -> None:
        """Record a directed A2A exchange as a single log entry."""
        self._messages.append({
            "sender":   f"{from_agent}→{to_agent}",
            "request":  request,
            "response": response,
            "ts":       datetime.now(timezone.utc).isoformat(),
        })

    def as_context(self, max_chars: int = 8000) -> str:
        lines = []
        for m in self._messages:
            if "request" in m:
                lines.append(
                    f"[{m['sender']}]\n"
                    f"  REQUEST: {m['request']}\n"
                    f"  RESPONSE: {m['response']}"
                )
            else:
                lines.append(f"[{m['sender']}]: {m['content']}")
        full = "\n\n".join(lines)
        return full[-max_chars:] if len(full) > max_chars else full

    def messages(self) -> list[dict]:
        return list(self._messages)

    def has_from(self, sender_prefix: str) -> bool:
        return any(m.get("sender", "").startswith(sender_prefix)
                   for m in self._messages)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_fences(sql: str) -> str:
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return sql


# ── SQL agent (two modes) ─────────────────────────────────────────────────────

def _sql_agent_respond(question: str, log: ConversationLog,
                       schema_ctx: str, llm: OpenAI) -> str:
    """
    Shared MAS SQLAgent answering a directed peer request.

    This configuration keeps directed messaging — a peer really does address
    this agent and receive its reply — but the reply is a plain string. The
    da_type that would mark it INFORM or REJECT, and any uncertainty_flags, are
    dropped at the adapter boundary. The caller must infer epistemic status from
    prose, which is the single difference from the Structured MAS.
    """
    return MAS.sql(question, context=log.as_context(max_chars=3000))


def _sql_agent_full(log: ConversationLog, query: str,
                    schema_ctx: str, llm: OpenAI) -> None:
    """Shared MAS SQLAgent over the whole query; result appended as prose."""
    out = MAS.sql(query, context=log.as_context(max_chars=3000))
    if out.strip():
        log.append("sql_agent", out)


def _data_explorer_respond(question: str, log: ConversationLog, llm: OpenAI) -> str:
    """
    Targeted profile response to a specific A2A request from a peer agent.

    Returns the rendered profile as a plain string — no DA type, no
    UncertaintyFlag. A high null rate comes back as prose the caller must read
    and interpret, where the MAS would attach NULL_VALUES to the message header.
    That difference is the ablation variable, not a capability gap.
    """
    profile = explore_for_query(question, force=True)
    return profile or "Could not identify any relevant columns from that request."


def _data_explorer_full(log: ConversationLog, query: str, llm: OpenAI) -> None:
    """Standalone profiling pass over the user query; result appended to the log."""
    profile = explore_for_query(query, llm=llm)
    if profile:
        log.append("data_explorer", profile)


def _doc_agent_respond(question: str, log: ConversationLog, llm: OpenAI) -> str:
    """Shared MAS DocAgent answering a directed peer request; reply is prose."""
    return MAS.doc(question, context=log.as_context(max_chars=3000))


def _doc_agent_full(log: ConversationLog, query: str, llm: OpenAI) -> None:
    """Shared MAS DocAgent over the whole query; result appended as prose."""
    out = MAS.doc(query, context=log.as_context(max_chars=3000))
    if out.strip():
        log.append("doc_agent", out)


def _pa_agent(
    log: ConversationLog,
    query: str,
    llm: OpenAI,
    ask_peer: Callable[[str, str], str],
) -> None:
    """
    Shared MAS PAAgent with directed but untyped peer access.

    This is the one configuration where the agent keeps a real delegation
    channel: it may ask sql_agent or doc_agent for what it lacks, exactly as in
    the Structured MAS. What it cannot do is read a da_type or an
    uncertainty_flag off the reply — the peer shim rewrites every response as a
    bare INFORM with no flags. So the wiring is identical to the MAS and only
    the typed contract is removed, which is the variable this configuration
    isolates.
    """
    out = MAS.pa_with_untyped_peers(query, context=log.as_context(max_chars=4000))
    if out.strip():
        log.append("pa_agent", out)


def _synthesizer(log: ConversationLog, query: str, llm: OpenAI) -> str:
    """Shared MAS SynthesizerAgent over the causal log."""
    return MAS.synthesize(query, log.as_context(max_chars=4000))


def _reflector(log: ConversationLog, query: str, candidate: str, llm: OpenAI) -> str:
    """Shared MAS ReflectorAgent, single pass; no typed challenge channel."""
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
Include all that are clearly needed."""

    _VALID = ("data_explorer", "sql_agent", "pa_agent", "doc_agent")
    try:
        raw = llm.chat.completions.create(
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

    def __init__(self, log_dir: str = "logs/unstructured_a2a"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = Path(log_dir) / f"unstructured_a2a_{ts}.jsonl"
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

class NaturalA2AMAS:
    """
    Natural-Language A2A multi-agent system for Fugaku HPC queries.

    PA agent can directly call ask_peer("sql_agent", question) to get
    targeted SQL responses before running predictions. This mirrors MAS's
    typed DATA_INSUFFICIENCY trigger, but uses plain prose instead of
    typed DialogueAct messages.

    The key ablation question: does replacing typed REJECT with plain
    "not available" text degrade uncertainty acknowledgment while
    preserving fact coverage?
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

        # Build ask_peer closure bound to this run's log
        def ask_peer(peer_name: str, question: str) -> str:
            if self.verbose:
                print(f"  [A2A] pa_agent → {peer_name}: {question[:60]}...")
            if peer_name == "sql_agent":
                return _sql_agent_respond(question, log, self.schema_ctx, self.llm)
            elif peer_name == "doc_agent":
                return _doc_agent_respond(question, log, self.llm)
            elif peer_name == "data_explorer":
                return _data_explorer_respond(question, log, self.llm)
            return "Peer not available."

        agents     = _classify_agents(query, self.llm)
        call_order = [a for a in ("data_explorer", "pa_agent", "sql_agent", "doc_agent")
                      if a in agents]
        self.logger.log("ROUTE", agents=call_order)
        if self.verbose:
            print(f"[NaturalA2A] routing → {call_order}")

        # data_explorer runs first so its column profiles are in the causal log
        # before any SQL is written — mirroring the MAS, where sql_agent profiles
        # before generating SQL.
        if "data_explorer" in call_order:
            if self.verbose:
                print("[NaturalA2A] running data_explorer")
            _data_explorer_full(log, query, self.llm)

        # PA runs FIRST so it can ask SQL for prediction parameters via ask_peer
        # (same ordering as MAS — PA triggers SQL sub-queries, not the other way around)
        if "pa_agent" in call_order:
            if self.verbose:
                print("[NaturalA2A] running pa_agent (with ask_peer)")
            _pa_agent(log, query, self.llm, ask_peer)

        # SQL agent for remaining query parts not covered by A2A sub-requests
        if "sql_agent" in call_order:
            if self.verbose:
                print("[NaturalA2A] running sql_agent (full query pass)")
            _sql_agent_full(log, query, self.schema_ctx, self.llm)

        # Doc agent for documentation parts
        if "doc_agent" in call_order:
            if self.verbose:
                print("[NaturalA2A] running doc_agent")
            _doc_agent_full(log, query, self.llm)

        # Synthesize + reflect
        answer = _synthesizer(log, query, self.llm)
        answer = _reflector(log, query, answer, self.llm)

        duration = time.perf_counter() - t0
        self.logger.log("FINAL_ANSWER", answer=answer, duration_s=round(duration, 2),
                        conversation_log=log.messages())
        if self.verbose:
            print(f"[NaturalA2A] done ({duration:.1f}s)")
        return answer
