# Results

## Provenance

**The code in this repository and the numbers in this folder do not yet match.**

Every configuration now runs the same agent implementations (`agents/`, reached
through `research/shared/mas_agents.py`). These result files predate that
change: when they were produced, each baseline still ran a separate, much
smaller agent of its own — a ~50-line SQL function for Blackboard against the
MAS's 770-line SQLAgent — so the baseline numbers below confound agent
implementation with communication architecture. A targeted probe showed the MAS
SQL agent alone closed 100% of the fact-recall gap on the five queries where
that gap was largest.

Re-running the four baselines against the current code is the outstanding work;
until then, treat these baseline figures as lower bounds.

Re-running the four baselines with the shared agents is the outstanding work;
the numbers below will move when that lands.

## What is current

- Scores use the corrected checker: `score_query()` unwraps display
  line-wrapping before phrase matching. Without it a correct refusal split
  across a line break scores zero, which cost several traps per configuration.
- Structured MAS is the all-GPT-4o configuration, matching the stated control
  that every configuration uses GPT-4o. The two-tier variant routes ~77% of its
  calls to gpt-4o-mini.
- Unstructured-Blackboard uses Blackboard's own context budgets and a neutral
  routing prompt, so that comparison isolates shared-state structure rather than
  context budget.

## Files

| Path | Contents |
|---|---|
| `json/` | per-query scored output, one file per configuration |
| `traces/` | per-query execution traces, Appendix E format |
| `tables/` | FR/UAA, turn-and-cost, trap-subtype, DA distribution |
