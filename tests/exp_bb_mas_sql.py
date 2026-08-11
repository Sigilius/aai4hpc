"""
tests/exp_bb_mas_sql.py

Ablation probe: swap the MAS SQLAgent into the Blackboard system.

Why
---
HISTORICAL. This probe measured a confound that has since been removed: every
configuration now runs the shared agents from agents/ via
research/shared/mas_agents.py, so this experiment can no longer be reproduced
against the current tree. It is retained because its result motivated that
change.

At the time it ran, the MAS SQL role was agents/sql_agent.py (770 lines, 9 LLM
call sites — sub-question extraction, delegation gates, profiling, generation,
sufficiency check) while the Blackboard SQL role was a 50-line function with a
single LLM call. The MAS-vs-Blackboard fact-recall gap therefore confounded the
communication architecture (the intended variable) with the SQL agent
implementation.

This isolates them. Blackboard keeps its architecture — the mutable dict, the
orchestrator-fixed call order, no typed messages, no uncertainty propagation —
and only the SQL *implementation* is replaced with the MAS one. The MAS agent is
run standalone with no synthesizer peer registered, so it replies INFORM
directly and its text lands in the ordinary `sql_result` slot as plain prose.
Nothing typed crosses into the blackboard.

If FR rises, part of the reported architectural gap is implementation.
If it does not, the gap is attributable to the architecture as claimed.

Usage:
    python3 tests/exp_bb_mas_sql.py                 # default 5-query probe
    python3 tests/exp_bb_mas_sql.py N28,N32,N34     # explicit ids
"""
import asyncio
import os
import sys
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "research"),
           os.path.join(_ROOT, "research", "single_agent_baseline"),
           os.path.join(_ROOT, "research", "shared"),
           os.path.join(_ROOT, "analytics"),
           os.path.join(_ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from run_n_queries import (                      # noqa: E402
    ALL_QUERIES, BATCH4_QUERIES, BATCH5_QUERIES, BATCH6_QUERIES,
    DOMAIN_QUERIES, _patch_rag_with_bm25, _print_q, _print_ans,
)

ALL = (ALL_QUERIES + BATCH4_QUERIES + BATCH5_QUERIES
       + DOMAIN_QUERIES + BATCH6_QUERIES)

# Queries where MAS most outperforms Blackboard on facts — the probe is most
# informative where the gap to explain is largest.
DEFAULT_IDS = ["N28", "N54", "N34", "N32", "N26"]


def install_mas_sql_agent():
    """Replace blackboard_baseline._sql_agent with the MAS SQLAgent."""
    import blackboard_baseline.agent as BB
    from core.shared_log import SharedLog
    from core.message_schema import A2AMessage, DialogueActType
    from agents.sql_agent import SQLAgent

    log = SharedLog(os.path.join(tempfile.mkdtemp(), "probe.db"))
    mas_sql = SQLAgent(log, verbose=False)      # no peers registered

    def _sql_agent(bb, llm, schema_ctx):
        query = bb.read("query")
        msg = A2AMessage(
            sender="orchestrator", recipient="sql_agent",
            da_type=DialogueActType.REQUEST, content=query,
            session_id=f"probe_{abs(hash(query)) % 10**8}", turn=1,
        )
        try:
            resp = asyncio.run(mas_sql.handle(msg))
            text = resp.content or ""
        except Exception as exc:
            text = f"SQL agent error: {exc}"
        # Plain text into the ordinary slot — no DA type, no flags carried over.
        bb.write("sql_agent", "sql_result", text)

    BB._sql_agent = _sql_agent
    print("[probe] blackboard _sql_agent  ->  MAS agents/sql_agent.py SQLAgent")


if __name__ == "__main__":
    ids = (sys.argv[1].split(",") if len(sys.argv) > 1 else DEFAULT_IDS)
    queries = [q for q in ALL if q["id"] in ids]
    if len(queries) != len(ids):
        missing = set(ids) - {q["id"] for q in queries}
        print(f"unknown ids: {missing}")
        sys.exit(1)

    _patch_rag_with_bm25()
    install_mas_sql_agent()

    from blackboard_baseline.agent import BlackboardMAS
    agent = BlackboardMAS()

    print(f"\n{'='*72}")
    print("  BLACKBOARD + MAS SQL AGENT  —  probe")
    print(f"{'='*72}")

    for q in queries:
        _print_q(q)
        t0 = time.perf_counter()
        try:
            ans = agent.run(q["query"])
        except Exception as e:
            ans = f"[SYSTEM ERROR — query skipped: {type(e).__name__}: {e}]"
        _print_ans(ans, time.perf_counter() - t0)
