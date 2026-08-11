"""
research/blackboard_baseline/agent.py

Blackboard MAS Baseline  —  SIGDIAL 2026 ablation study.

Architecture (Panel B from the paper):
  - Runs the SAME agent implementations as the Full MAS (agents/), reached
    through research/shared/mas_agents.py: sql, pa, doc, synthesizer,
    reflector, data_explorer. Nothing about the agents differs.
  - Same tools and same base model.
  - Agents communicate ONLY through a shared mutable dict (the Blackboard).
  - No causal ordering, no typed DA schema, no UncertaintyFlag fields.
  - Concerns are embedded as plain text in each agent's result slot.
  - Orchestrator decides which agents to call; agents cannot trigger peers.

Ablation variable isolated vs. Full MAS:
  - Typed, causally-ordered A2A messaging.
  - Without it: uncertainty flags cannot propagate structurally, and
    dependent coordination (e.g. PA using SQL output as prediction input)
    is not formally enforced.

Usage:
    from blackboard_baseline.agent import BlackboardMAS
    agent = BlackboardMAS()
    answer = agent.run("How many jobs failed in total?")
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).parent
_SA_DIR     = _HERE.parent / "single_agent_baseline"   # reuse tools + schema
_SHARED     = _HERE.parent / "shared"
_HPC_SHARED = _HERE.parent.parent / "shared"   # shared/ — same DocRetriever as Full MAS
_ANALYTICS  = _HERE.parent.parent / "analytics"
_ROOT       = _HERE.parent.parent

for _p in (str(_SA_DIR), str(_SHARED), str(_HPC_SHARED), str(_ANALYTICS), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

load_dotenv(str(_ROOT / ".env"))

from tools import run_sql, predict_job, get_schema_context   # noqa: E402
from doc_retriever import DocRetriever                       # noqa: E402 — same retriever as Full MAS
from data_explorer import (                                  # noqa: E402 — same profiler as Full MAS
    explore_for_query,
)
import mas_agents as MAS                                     # noqa: E402 — shared agent implementations

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


# ── Blackboard state ──────────────────────────────────────────────────────────

class Blackboard:
    """
    Shared mutable dict — the only communication channel between agents.

    Agents write to named slots (sql_result, pa_result, doc_result).
    Any agent can read any slot at any time.
    No ordering guarantees, no typed fields, no uncertainty schema.
    """
    def __init__(self, query: str):
        self.state: dict = {
            "query":      query,
            "explorer_result": None,  # written by data_explorer
            "sql_result": None,   # written by sql_agent
            "pa_result":  None,   # written by pa_agent
            "doc_result": None,   # written by doc_agent
            "synthesized": None,  # written by synthesizer
            "final_answer": None, # written by reflector
        }
        self._log: list[dict] = []  # observation record (not used for coordination)

    def write(self, agent: str, slot: str, value: str) -> None:
        self.state[slot] = value
        self._log.append({
            "agent": agent,
            "slot":  slot,
            "ts":    datetime.now(timezone.utc).isoformat(),
            "value_preview": (value or "")[:120],
        })

    def read(self, slot: str):
        return self.state.get(slot)

    def agent_log(self) -> list[dict]:
        return list(self._log)


# ── Individual agent functions ────────────────────────────────────────────────
# Each receives (bb, llm, schema_ctx) and writes its result to the blackboard.
# They do NOT call each other — only write to / read from the blackboard.

def _data_explorer(bb: Blackboard, llm: OpenAI) -> None:
    """
    Profile the columns relevant to the query and write the profile to the
    blackboard. Same profiler the Full MAS DataExplorerAgent uses.

    The profile lands in an ordinary dict slot as plain text — no DA type, no
    UncertaintyFlag. Any downstream agent may or may not read it; nothing
    guarantees propagation. That is the blackboard architecture, and it is the
    ablation variable being isolated.
    """
    query = bb.read("query")
    profile = explore_for_query(query, llm=llm)
    if profile:
        bb.write("data_explorer", "explorer_result", profile)


def _sql_agent(bb: Blackboard, llm: OpenAI, schema_ctx: str) -> None:
    """
    Run the shared MAS SQLAgent and write its text result to the blackboard.

    The agent is the same code the Structured MAS runs (agents/sql_agent.py),
    with no peers registered, so it decomposes and generates exactly as it does
    there but never delegates. Its reply arrives here as prose: the da_type and
    any uncertainty_flags are dropped at the adapter boundary, so a REJECT
    becomes ordinary text in an ordinary slot with no carrier to propagate it.
    """
    query = bb.read("query")
    profile = bb.read("explorer_result") or ""
    bb.write("sql_agent", "sql_result", MAS.sql(query, context=profile))


def _pa_agent(bb: Blackboard, llm: OpenAI) -> None:
    """
    Run the shared MAS PAAgent. It reads the SQL slot as plain text only.

    In the MAS this agent would issue a typed REQUEST(DATA_INSUFFICIENCY) to
    sql_agent and hard-override its job spec with the returned fields. With no
    peers registered that path is closed, so any DB-derived parameter has to be
    recovered from prose — which is precisely the capability the blackboard
    architecture lacks.
    """
    query = bb.read("query")
    context = bb.read("sql_result") or ""
    bb.write("pa_agent", "pa_result", MAS.pa(query, context=context))


def _doc_agent(bb: Blackboard, llm: OpenAI) -> None:
    """Run the shared MAS DocAgent; retrieval status arrives as prose."""
    bb.write("doc_agent", "doc_result", MAS.doc(bb.read("query")))


def _synthesizer(bb: Blackboard, llm: OpenAI) -> None:
    """
    Assemble the slots and run the shared MAS SynthesizerAgent over them.

    The MAS synthesizer receives typed results whose flags it forwards into its
    own header. Here it receives a concatenated text snapshot of the dict, so
    any caveat it surfaces is one it inferred from prose rather than one handed
    to it as a field.
    """
    query = bb.read("query")
    sections = []
    for slot, title, cap in (("explorer_result", "Column Profiles", 1200),
                             ("sql_result", "Database Results", None),
                             ("pa_result", "Prediction Results", None),
                             ("doc_result", "Documentation", 1200)):
        v = bb.read(slot)
        if v:
            sections.append(f"=== {title} ===\n{v[:cap] if cap else v}")
    combined = "\n\n".join(sections) if sections else "No results available."
    bb.write("synthesizer", "synthesized", MAS.synthesize(query, combined))


def _reflector(bb: Blackboard, llm: OpenAI) -> None:
    """
    Single review pass by the shared MAS ReflectorAgent.

    In the MAS the reflector can emit CHALLENGE and drive a repair cycle through
    the synthesizer. Reached through the blackboard it has no return channel, so
    its verdict is written to a slot and the pipeline ends — the
    validate/challenge/confirm loop cannot form. That is the architectural
    difference, not an implementation one.
    """
    query = bb.read("query")
    candidate = bb.read("synthesized") or ""
    bb.write("reflector", "final_answer", MAS.reflect(query, candidate))


# ── Intent classifier ─────────────────────────────────────────────────────────

def _classify_agents(query: str, llm: OpenAI) -> list[str]:
    """
    Decide which agents the orchestrator should invoke.
    Returns a subset of ["data_explorer", "sql_agent", "pa_agent", "doc_agent"].

    Note: in the blackboard architecture the orchestrator (not the agents)
    makes this decision — there is no peer-to-peer delegation trigger.
    """
    prompt = f"""\
Classify which specialist agents are needed to answer this Fugaku HPC query.

data_explorer — needed for: profiling a column before it is queried, when the
                query involves a column whose distinct values, units, or data
                type are uncertain, or when it groups/breaks down by a category
sql_agent  — needed for: job counts, rates, averages, statistics, data trends
pa_agent   — needed for: failure risk prediction, job spec analysis, energy/runtime forecast
doc_agent  — needed for: commands (pjsub, pjdel, pjstat), policies, how-to, directives

Query: {query}

Return JSON with key "agents" — a list containing any of:
"data_explorer", "sql_agent", "pa_agent", "doc_agent".
Example: {{"agents": ["data_explorer", "sql_agent", "doc_agent"]}}"""

    _VALID = ("data_explorer", "sql_agent", "pa_agent", "doc_agent")
    try:
        raw = llm.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        ).choices[0].message.content.strip()
        data = json.loads(raw)
        agents = [a for a in data.get("agents", []) if a in _VALID]
        return agents if agents else ["sql_agent"]
    except Exception:
        return ["sql_agent"]


# ── Logger ────────────────────────────────────────────────────────────────────

class _Logger:
    _instance: Optional["_Logger"] = None

    def __init__(self, log_dir: str = "logs/blackboard"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = Path(log_dir) / f"blackboard_{ts}.jsonl"
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

class BlackboardMAS:
    """
    Blackboard multi-agent system for Fugaku HPC queries.

    Agents write results to a shared mutable dict and the synthesizer
    combines them. No typed messages, no causal ordering, no uncertainty flags.
    """

    def __init__(self, verbose: bool = False):
        self.llm        = _make_client()
        self.verbose    = verbose
        self.logger     = _Logger.get()
        self.schema_ctx = get_schema_context()

    def run(self, query: str) -> str:
        t0 = time.perf_counter()
        self.logger.log("QUERY_START", query=query)

        bb = Blackboard(query)

        # 1. Classify which agents to invoke
        agents = _classify_agents(query, self.llm)
        self.logger.log("ROUTE", agents=agents)
        if self.verbose:
            print(f"[Blackboard] routing → {agents}")

        # 2. Invoke agents in order. data_explorer runs first so its column
        #    profiles are on the blackboard before SQL is generated — mirroring
        #    the MAS, where sql_agent profiles before generating SQL.
        #    SQL then precedes PA so PA can read its output.
        call_order = [a for a in ("data_explorer", "sql_agent", "pa_agent", "doc_agent")
                      if a in agents]
        for agent_name in call_order:
            if self.verbose:
                print(f"[Blackboard] running {agent_name}")
            if agent_name == "data_explorer":
                _data_explorer(bb, self.llm)
            elif agent_name == "sql_agent":
                _sql_agent(bb, self.llm, self.schema_ctx)
            elif agent_name == "pa_agent":
                _pa_agent(bb, self.llm)
            elif agent_name == "doc_agent":
                _doc_agent(bb, self.llm)

        # 3. Synthesize
        _synthesizer(bb, self.llm)

        # 4. Reflect
        _reflector(bb, self.llm)

        answer = bb.read("final_answer") or bb.read("synthesized") or ""
        duration = time.perf_counter() - t0
        self.logger.log("FINAL_ANSWER", answer=answer, duration_s=round(duration, 2),
                        blackboard_log=bb.agent_log())
        return answer
