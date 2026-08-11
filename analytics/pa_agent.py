"""
pa_agent.py  —  Predictive Analytics Agent (EPIC / Fugaku)
Wires the Predictor (failure/cost/runtime/energy) into the EPIC
LangChain agentic loop as a registered tool.

Architecture (matches ISC-2026 paper §III-D):
  QP orchestrator  →  function-call  →  pa_tool  →  Predictor  →  response

Usage (standalone):
    python analytics/pa_agent.py

Usage (from QP orchestrator):
    from analytics.pa_agent import pa_predict_tool, pa_agent_invoke
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
from typing import Optional
from predict import Predictor

# ── langchain_core.tools is enough — no AgentExecutor needed ─────
try:
    from langchain_core.tools import tool as lc_tool
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    # Minimal stub so the rest of the file works without langchain
    def lc_tool(fn):
        fn.name        = fn.__name__
        fn.description = fn.__doc__ or ""
        return fn

# ── lazy-load the predictor once ─────────────────────────────────
_predictor: Optional[Predictor] = None

def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


# ═══════════════════════════════════════════════════════════════════
# LangChain Tool — callable by any tool-calling LLM in the QP loop
# ═══════════════════════════════════════════════════════════════════

@lc_tool
def pa_predict_tool(job_json: str) -> str:
    """
    Predict failure risk, expected runtime, and energy for a Fugaku HPC job.

    Input: JSON string with at minimum:
      {"nnumr": <int>, "elpl": <int>, "pclass": "compute-bound"|"memory-bound"}

    Optional keys: nnuma, cnumr, mszl, msza, freq_req, pri,
                   jobenv_req, usr, jnam, qdt

    Returns a plain-English prediction summary including:
      - Risk level (OK / CAUTION / WARNING) and failure probability
      - Expected runtime and energy if the job completes
      - Top risk factors (user history, node anomaly, job-name patterns)
      - Recommendation if risk is elevated
    """
    try:
        job = json.loads(job_json)
    except json.JSONDecodeError as e:
        return f"Error: could not parse job_json — {e}"
    return _run_pa_direct(job)


# ═══════════════════════════════════════════════════════════════════
# Core: format the Predictor output into plain English
# ═══════════════════════════════════════════════════════════════════

def _run_pa_direct(job: dict) -> str:
    result = get_predictor().predict(job)
    result.pop("_raw", None)

    risk    = result["risk_level"]
    p_fail  = result["p_fail"]
    runtime = result["expected_runtime"]
    energy  = result["expected_energy"]
    reasons = result["top_reasons"]
    fail_t  = result["fail_type_if_fails"]
    wasted  = result["wasted_node_hrs_if_slow"]

    lines = [
        f"[{risk}]  Failure probability : {100*p_fail:.1f}%",
        f"Expected runtime      : {runtime}",
        f"Expected energy       : {energy}",
        f"If it fails           : {fail_t}",
        f"Node-hrs wasted (slow): {wasted}",
        "",
        "Top risk factors:",
    ]
    for r in reasons:
        lines.append(f"  • {r}")

    if risk == "WARNING":
        lines += [
            "",
            "⚠️  RECOMMENDATION: Elevated failure risk.",
            "   Review your job script, memory settings, and node count",
            "   before submitting to avoid wasting compute allocation.",
        ]
    elif risk == "CAUTION":
        lines += [
            "",
            "⚡ CAUTION: Mild failure risk. Consider a short test run on",
            "   fewer nodes before committing the full allocation.",
        ]

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Public API — used by QP orchestrator and Chainlit UI
# ═══════════════════════════════════════════════════════════════════

def pa_agent_invoke(job: dict) -> dict:
    """
    Entry point for the QP orchestrator's function-call routing.
    Mirrors AgentExecutor.invoke() interface: returns {"output": str}.

    Args:
        job: dict with job parameters (same schema as Predictor.predict)

    Returns:
        {"output": "<plain-English summary>"}
    """
    return {"output": _run_pa_direct(job)}


def run_pa_from_query(natural_language_query: str, llm=None) -> str:
    """
    Full QI → Prediction pipeline.

    If `llm` is provided (any LangChain ChatModel that supports tool-calling),
    the LLM parses the query and calls pa_predict_tool automatically.

    If `llm` is None, prints instructions for manual param extraction.

    Args:
        natural_language_query: Free-text user query, e.g.:
            "Will a 512-node MD simulation running for 24 hours fail?"
        llm: Optional LangChain ChatModel

    Returns:
        Prediction summary string
    """
    if llm is None:
        return (
            "No LLM provided. Call pa_agent_invoke(job_dict) directly, "
            "or pass a LangChain ChatModel to use the full QI pipeline.\n"
            "Example:\n"
            '  from langchain_openai import ChatOpenAI\n'
            '  llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)\n'
            '  run_pa_from_query("Will my 512-node job fail?", llm=llm)'
        )

    # Build a minimal tool-calling loop without AgentExecutor
    from langchain_core.messages import HumanMessage, SystemMessage

    PA_SYSTEM = (
        "You are the Predictive Analytics agent for the EPIC HPC system. "
        "When asked about job failure risk or runtime, call pa_predict_tool "
        "with a JSON string containing the job parameters extracted from the query. "
        "Required fields: nnumr (int), elpl (int, seconds), pclass (string). "
        "Use defaults if missing: nnumr=1, elpl=3600, pclass='compute-bound'."
    )

    llm_with_tools = llm.bind_tools([pa_predict_tool])

    messages = [
        SystemMessage(content=PA_SYSTEM),
        HumanMessage(content=natural_language_query),
    ]

    # Step 1: LLM decides to call the tool
    response = llm_with_tools.invoke(messages)

    if not response.tool_calls:
        return response.content  # LLM answered without a tool call

    # Step 2: Execute tool calls
    from langchain_core.messages import ToolMessage
    results = []
    for tc in response.tool_calls:
        tool_result = pa_predict_tool.invoke(tc["args"])
        results.append(ToolMessage(content=tool_result, tool_call_id=tc["id"]))

    # Step 3: LLM synthesizes the tool result into a final answer
    messages += [response] + results
    final = llm.invoke(messages)
    return final.content


# ═══════════════════════════════════════════════════════════════════
# Self-test (no LLM required)
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 64)
    print("  PA Agent — Direct mode self-test  (no LLM required)")
    print("=" * 64)

    tests = [
        {
            "label": "Large MD job (512 nodes, 24hr, compute-bound, known user)",
            "job":   dict(nnumr=512, elpl=86400, pclass="compute-bound",
                          usr="usr_1229", jnam="md_water",
                          qdt="2024-01-15 22:00:00"),
        },
        # {
        #     "label": "Small postprocess (4 nodes, 1hr, memory-bound)",
        #     "job":   dict(nnumr=4, elpl=3600, pclass="memory-bound",
        #                   usr="user_0100", jnam="postprocess",
        #                   qdt="2024-01-15 09:00:00"),
        # },
        # {
        #     "label": "New user, overnight large job (cold start)",
        #     "job":   dict(nnumr=256, elpl=43200, pclass="compute-bound",
        #                   usr="brand_new_xyz", jnam="lammps_sim",
        #                   qdt="2024-01-16 23:00:00"),
        # },
    ]

    for t in tests:
        print(f"\n{'─'*64}")
        print(f"  {t['label']}")
        print(f"{'─'*64}")
        print(pa_agent_invoke(t["job"])["output"])

    # Print the tool JSON schema so QP orchestrator knows the call signature
    print("\n" + "="*64)
    print("  Tool schema for QP orchestrator")
    print("="*64)
    if _HAS_LANGCHAIN:
        try:
            schema = pa_predict_tool.args_schema.model_json_schema()
            print(json.dumps(schema, indent=2))
        except Exception:
            print("(schema not available — langchain_core not fully installed)")
    else:
        print("(langchain_core not installed — tool runs in stub mode)")
