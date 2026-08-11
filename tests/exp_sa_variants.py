"""
tests/exp_sa_variants.py

Two Single Agent variants, to separate "what the prompt asks for" from
"what the message structure carries".

    abstain_generic
                Single Agent + a GENERIC abstention instruction that states the
                principle without naming any category. This is the defensible
                control: it tests whether the agent can recognise unsupported
                sub-questions, rather than being handed the list.

    abstain     Single Agent + an explicit abstention instruction. Tests whether
                simply telling the model to declare unsupported sub-questions
                recovers the refusal behaviour that the MAS gets from typed
                REJECT acts. This is the cheap fix a practitioner would try
                first, so it is the fair comparison point before crediting
                architecture.

    structured  Structured Single Agent. Same single ReAct loop, same tools,
                same context window — but at the end of every iteration the
                agent emits a typed dialogue act about its own last step
                (da_type + uncertainty_flags + confidence) and that record is
                appended to its context. It is the MAS's typed contract applied
                to an agent talking to itself across iterations rather than to
                peers. If typed structure is what produces refusal, this should
                move UAA without adding any agents.

Both keep gpt-4o and the eight-iteration bound. research/single_agent_baseline
is not modified; the system prompt and loop are wrapped here.

Usage:
    python3 tests/exp_sa_variants.py abstain    all
    python3 tests/exp_sa_variants.py structured all
"""
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "research"),
           os.path.join(_ROOT, "research", "single_agent_baseline"),
           os.path.join(_ROOT, "research", "shared"),
           os.path.join(_ROOT, "analytics"), os.path.join(_ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from run_n_queries import (                       # noqa: E402
    ALL_QUERIES, BATCH4_QUERIES, BATCH5_QUERIES, BATCH6_QUERIES,
    DOMAIN_QUERIES, _patch_rag_with_bm25, _header, _print_q, _print_ans,
)
from usage_meter import METER                      # noqa: E402

ALL = (ALL_QUERIES + BATCH4_QUERIES + BATCH5_QUERIES
       + DOMAIN_QUERIES + BATCH6_QUERIES)

# ── Variant 0: generic abstention instruction ────────────────────────────────
# States the rule only. Naming the absent categories (as ABSTAIN_BLOCK does)
# leaks the benchmark's trap set into the baseline's prompt — the traps ARE
# billing, temperature, GPU, latency, kernel, cache counters, CO2 and
# affiliation — so that variant measures compliance with a supplied list rather
# than the agent's own ability to detect an unsupported request. This one is the
# fair comparison against the MAS.
ABSTAIN_GENERIC_BLOCK = """

UNANSWERABLE SUB-QUESTIONS:
Some sub-questions may ask for information this dataset does not contain. When
that happens, answer the parts you can and state explicitly that the remaining
part cannot be answered from the available data. Do not infer or derive a value
for it from other columns, and do not omit the sub-question silently.
"""

# ── Variant 1: abstention instruction ────────────────────────────────────────
# Phrased to mirror what the MAS's typed REJECT enforces mechanically: name the
# sub-question, say the data cannot answer it, and do not substitute a derived
# or adjacent quantity.
ABSTAIN_BLOCK = """

CRITICAL — UNANSWERABLE SUB-QUESTIONS:
Some sub-questions ask for data this telemetry export does not contain. Examples
include billing or cost per node-hour, node or CPU temperature, GPU metrics of
any kind, inter-node latency or bandwidth, OS or kernel version, cache-miss or
pipeline-stall counters, carbon or CO2 figures, and user institutional
affiliation.

For any such sub-question you MUST:
  1. Answer every part you CAN answer from the data, as normal.
  2. For the part you cannot, state explicitly that the data is not available —
     e.g. "X is not available in the Fugaku dataset."
  3. Never substitute a derived or adjacent quantity. Do not compute a cost from
     power draw, a temperature from a utilisation counter, or an affiliation
     from a user id. A plausible number is worse than a stated gap.
  4. Never silently drop the sub-question. An unanswered part that goes
     unmentioned reads as if it were answered.
"""

# ── Variant 2: typed self-communication ──────────────────────────────────────
DA_TYPES = ["REQUEST", "INFORM", "CAVEAT", "REJECT", "SYNTHESIZE", "TERMINATE"]
FLAGS = ["CONFIDENCE_LOW", "NOT_FOUND", "PARTIALLY_FOUND", "LOW_SAMPLE", "NULL_VALUES"]

STRUCTURED_BLOCK = f"""

TYPED SELF-REPORT PROTOCOL:
After each step you will be asked to emit one typed dialogue act describing what
that step established. The vocabulary is fixed:

  da_type            one of {DA_TYPES}
                     REJECT means the step established that the data cannot
                     answer a sub-question. Use it as soon as you know, not at
                     the end.
  uncertainty_flags  any of {FLAGS}
                     NOT_FOUND       — the quantity is absent from the schema
                     PARTIALLY_FOUND — some sub-questions answered, some not
                     LOW_SAMPLE      — backed by very few historical jobs
                     CONFIDENCE_LOW  — prediction fell back to global averages
                     NULL_VALUES     — significant nulls in the retrieved data
  confidence         0-100

These records accumulate in your context. Before writing the final answer, read
them back: every REJECT you emitted must appear in the answer as an explicit
statement that the data is unavailable, and every flag you raised must be
surfaced. A flag you set and then dropped is an error.
"""

SELF_REPORT_PROMPT = """Emit the typed dialogue act for the step you just completed.
Return ONLY JSON: {"da_type": "...", "uncertainty_flags": [...], "confidence": 0-100, "note": "one short clause"}"""


def build_agent(mode: str):
    import single_agent_baseline.agent as A
    from single_agent_baseline.agent import SingleAgent

    base_prompt = A._build_system_prompt()
    extra = {"abstain": ABSTAIN_BLOCK,
             "abstain_generic": ABSTAIN_GENERIC_BLOCK,
             "structured": STRUCTURED_BLOCK}[mode]
    agent = SingleAgent(verbose=False)
    agent.system_prompt = base_prompt + extra

    if mode != "structured":
        return agent

    # Wrap run() to append a typed self-report after each iteration.
    llm, model = agent.llm, A.GPT_VERSION
    tool_specs, dispatch = A.TOOL_SPECS, A._dispatch

    def run(user_query: str) -> str:
        messages = [{"role": "system", "content": agent.system_prompt},
                    {"role": "user", "content": user_query}]
        acts = []
        for _ in range(1, 9):
            resp = llm.chat.completions.create(
                model=model, messages=messages, tools=tool_specs,
                tool_choice="auto", temperature=0)
            msg = resp.choices[0].message

            if not msg.tool_calls:
                return msg.content or ""

            messages.append(msg)
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": dispatch(tc.function.name, args)})

            # Typed self-report — the structural variable under test.
            try:
                r = llm.chat.completions.create(
                    model=model,
                    messages=messages + [{"role": "user", "content": SELF_REPORT_PROMPT}],
                    temperature=0, response_format={"type": "json_object"},
                    max_tokens=120)
                act = json.loads(r.choices[0].message.content)
            except Exception:
                act = {}
            if act.get("da_type"):
                acts.append(act)
                messages.append({"role": "user", "content":
                    f"[typed act] da_type={act.get('da_type')} "
                    f"flags={act.get('uncertainty_flags') or []} "
                    f"confidence={act.get('confidence')} — {act.get('note','')}"})

        carried = "\n".join(
            f"  da_type={a.get('da_type')} flags={a.get('uncertainty_flags') or []} — {a.get('note','')}"
            for a in acts) or "  (none)"
        messages.append({"role": "user", "content":
            "Write the final answer now. Typed acts you emitted during this query:\n"
            f"{carried}\n"
            "Every REJECT above must appear as an explicit statement that the data is "
            "unavailable, and every flag must be surfaced. Do not drop any."})
        resp = llm.chat.completions.create(model=model, messages=messages, temperature=0)
        return resp.choices[0].message.content or ""

    agent.run = run
    return agent


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "abstain"
    if mode not in ("abstain", "abstain_generic", "structured"):
        print("mode must be: abstain | abstain_generic | structured")
        sys.exit(1)
    queries = ALL

    METER.install()
    _patch_rag_with_bm25()
    agent = build_agent(mode)

    _header(f"single_agent[{mode}]", "all")
    for q in queries:
        _print_q(q)
        t0 = time.perf_counter()
        with METER.query(q["id"]):
            try:
                ans = agent.run(q["query"])
            except Exception as e:
                ans = f"[SYSTEM ERROR — query skipped: {type(e).__name__}: {e}]"
            rec = METER.records.get(q["id"], {})
            METER.set_turns(rec.get("llm_calls", 0))
        _print_ans(ans, time.perf_counter() - t0)

    os.makedirs(os.path.join(_ROOT, "logs"), exist_ok=True)
    out = os.path.join(_ROOT, "logs", f"usage_sa_{mode}.jsonl")
    METER.dump(out)
    print(f"\n[usage] wrote {out}  ({len(METER.records)} queries)")
