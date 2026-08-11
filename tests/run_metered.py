"""
tests/run_metered.py

Instrumented runner: same 55 queries and same systems as run_n_queries.py, but
records per-query LLM cost and effort alongside the answer.

Why a separate runner. run_n_queries.py is the validated harness that produced
the reported numbers, and the systems under test should not be edited to measure
them. This wraps both from the outside: it reuses run_n_queries' query lists and
BM25 patch verbatim, installs the shared UsageMeter (which patches the OpenAI
SDK, so every system is counted identically including cheap gates), and counts
agent activations by decorating the baseline module functions at runtime. No
system's own source is modified.

Output:
  stdout                       — identical format to run_n_queries.py, so
                                 score_run.py / generate_sigdial_json.py parse it
  logs/usage_<system>.jsonl    — per-query llm_calls, turns, tokens

Usage:
    python3 tests/run_metered.py mas all
    python3 tests/run_metered.py blackboard all
    python3 tests/run_metered.py unstructured all
    python3 tests/run_metered.py a2a all
    python3 tests/run_metered.py single all
"""
import asyncio
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PR   = os.path.join(_ROOT, "research")
_SA   = os.path.join(_PR,   "single_agent_baseline")
_SH   = os.path.join(_PR,   "shared")
_AN   = os.path.join(_ROOT, "analytics")
_TE   = os.path.join(_ROOT, "tests")
for _p in (_ROOT, _PR, _SA, _SH, _AN, _TE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from run_n_queries import (            # noqa: E402 — reuse the validated harness
    ALL_QUERIES, BATCH4_QUERIES, BATCH5_QUERIES, BATCH6_QUERIES,
    NEW_QUERIES, BATCH2_QUERIES, BATCH3_QUERIES, DOMAIN_QUERIES,
    _patch_rag_with_bm25, _header, _print_q, _print_ans,
)
from usage_meter import METER          # noqa: E402


def _count_activations(module, names):
    """Wrap module-level agent functions so each invocation counts as one turn."""
    for n in names:
        fn = getattr(module, n, None)
        if fn is None:
            continue

        def make(f):
            def wrapped(*a, **kw):
                METER.note_turn()
                return f(*a, **kw)
            return wrapped

        setattr(module, n, make(fn))


# ── Runners ───────────────────────────────────────────────────────────────────

def _force_mas_model(model: str) -> None:
    """
    Make every MAS LLM call use `model`, overriding the per-call tier choice.

    The MAS normally splits work across two tiers: _llm_bool() defaults to
    gpt-4o-mini and ~20 call sites pass model="gpt-4o-mini" explicitly for
    extraction, parsing and delegation gates, leaving only _generate_sql,
    doc _synthesize and the synthesizer on gpt-4o. That is ~77% of calls on the
    cheap tier.

    This patches BaseAgent._llm at the boundary rather than editing the agents,
    so the system under test is unmodified and the override is visible in one
    place. _llm_bool routes through _llm, so pinning _llm covers both paths.
    """
    from agents.base_agent import BaseAgent

    original = BaseAgent._llm

    def _llm(self, prompt, **kwargs):
        kwargs["model"] = model
        return original(self, prompt, **kwargs)

    BaseAgent._llm = _llm
    print(f"[model] MAS pinned to {model} for every call (tiering disabled)")


def run_mas(queries):
    forced = os.getenv("MAS_FORCE_MODEL", "").strip()
    if forced:
        _force_mas_model(forced)
    from orchestrator import HpcOrchestrator
    agent = HpcOrchestrator(verbose=True)

    async def _go():
        for q in queries:
            _print_q(q)
            sid = f"n_{q['id']}"
            t0 = time.perf_counter()
            with METER.query(q["id"]):
                try:
                    ans = await agent.gateway.run(q["query"], sid)
                except Exception as e:
                    ans = f"[SYSTEM ERROR — query skipped: {type(e).__name__}: {e}]"
                # MAS turns are NOT taken from the shared log. SharedLog.append
                # is called on several paths per message (sender, recipient, and
                # forwarding), so a row count overstates hops roughly threefold.
                # The authoritative hop definition is the printed DA trace —
                # every "DA=" line except USER_QUERY/TERMINATE, per
                # sigdial_json/gen_hop_breakdown.py — so turns for the MAS are
                # derived from the run log by build_per_query_table.py instead.
                pass
            _print_ans(ans, time.perf_counter() - t0)

    asyncio.run(_go())


def _run_baseline(queries, build, module=None, fns=()):
    if module is not None:
        _count_activations(module, fns)
    agent = build()
    for q in queries:
        _print_q(q)
        t0 = time.perf_counter()
        with METER.query(q["id"]):
            try:
                ans = agent.run(q["query"])
            except Exception as e:
                ans = f"[SYSTEM ERROR — query skipped: {type(e).__name__}: {e}]"
        _print_ans(ans, time.perf_counter() - t0)


def run_blackboard(queries):
    _patch_rag_with_bm25()
    import blackboard_baseline.agent as M
    from blackboard_baseline.agent import BlackboardMAS
    _run_baseline(queries, BlackboardMAS, M,
                  ("_data_explorer", "_sql_agent", "_pa_agent", "_doc_agent",
                   "_synthesizer", "_reflector"))


def run_unstructured(queries):
    _patch_rag_with_bm25()
    import unstructured_baseline.agent as M
    from unstructured_baseline.agent import UnstructuredMAS
    _run_baseline(queries, UnstructuredMAS, M,
                  ("_data_explorer", "_sql_agent", "_pa_agent", "_doc_agent",
                   "_check_for_delegation", "_synthesizer", "_reflector"))


def run_a2a(queries):
    _patch_rag_with_bm25()
    import unstructured_a2a_baseline.agent as M
    from unstructured_a2a_baseline.agent import NaturalA2AMAS
    _run_baseline(queries, NaturalA2AMAS, M,
                  ("_data_explorer_full", "_data_explorer_respond",
                   "_sql_agent_full", "_sql_agent_respond",
                   "_doc_agent_full", "_doc_agent_respond",
                   "_pa_agent", "_synthesizer", "_reflector"))


def run_single(queries):
    _patch_rag_with_bm25()
    from single_agent_baseline.agent import SingleAgent
    agent = SingleAgent(verbose=False)
    for q in queries:
        _print_q(q)
        t0 = time.perf_counter()
        with METER.query(q["id"]):
            try:
                ans = agent.run(q["query"])
            except Exception as e:
                ans = f"[SYSTEM ERROR — query skipped: {type(e).__name__}: {e}]"
            # The single agent has no peer graph; its ReAct iteration count is
            # the comparable notion of a turn. Each iteration is exactly one
            # chat completion, so llm_calls is the iteration count.
            rec = METER.records.get(q["id"], {})
            METER.set_turns(rec.get("llm_calls", 0))
        _print_ans(ans, time.perf_counter() - t0)


SUITES = {
    "batch1": NEW_QUERIES, "batch2": BATCH2_QUERIES, "batch3": BATCH3_QUERIES,
    "batch4": BATCH4_QUERIES, "batch5": BATCH5_QUERIES, "batch6": BATCH6_QUERIES,
    "domain": DOMAIN_QUERIES,
    "all": ALL_QUERIES + BATCH4_QUERIES + BATCH5_QUERIES + DOMAIN_QUERIES + BATCH6_QUERIES,
}

RUNNERS = {
    "mas": run_mas, "blackboard": run_blackboard, "unstructured": run_unstructured,
    "a2a": run_a2a, "unstructured_a2a": run_a2a,
    "single": run_single, "sa": run_single, "single_agent": run_single,
}


if __name__ == "__main__":
    system = sys.argv[1].lower() if len(sys.argv) > 1 else "mas"
    suite  = sys.argv[2].lower() if len(sys.argv) > 2 else "all"

    queries = SUITES.get(suite)
    if queries is None:
        print(f"Unknown suite '{suite}'. Choose: {' | '.join(SUITES)}")
        sys.exit(1)
    runner = RUNNERS.get(system)
    if runner is None:
        print(f"Unknown system '{system}'. Choose: {' | '.join(sorted(set(RUNNERS)))}")
        sys.exit(1)

    METER.install()
    _header(system, suite)
    try:
        runner(queries)
    finally:
        os.makedirs(os.path.join(_ROOT, "logs"), exist_ok=True)
        tag = os.getenv("RUN_TAG", "")
        out = os.path.join(_ROOT, "logs", f"usage_{system}{tag}.jsonl")
        METER.dump(out)
        print(f"\n[usage] wrote {out}  ({len(METER.records)} queries)")
