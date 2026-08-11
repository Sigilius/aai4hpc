# AAI4HPC — Multi-Agent Architectures for HPC Operational Analytics

Code and results for the biaxial ablation over agent-to-agent communication
structure and shared-state organisation, evaluated on 55 multi-intent queries
against the Fugaku job telemetry export (25.8M records).

## Configurations

All five run the **same specialist agent implementations** (`agents/`) — the same
SQL, PA, Doc, DataExplorer, Synthesizer and Reflector code, with the same
prompts, tools and base model. No configuration has an agent of its own. Only
the substrate carrying results between agents differs, and that is the ablation
variable.

| Configuration | Substrate | Typed fields |
|---|---|---|
| Structured MAS | typed A2A over a causally-ordered log | `da_type`, `uncertainty_flags`, `delegation_trigger` |
| Unstructured MAS | directed peer calls, causal log | stripped — replies rewritten as bare `INFORM` |
| Blackboard MAS | mutable dict, orchestrator-fixed order | none; no peer channel |
| Unstructured-Blackboard MAS | unordered free-text log | none |
| Single Agent | one ReAct loop holding all tools | n/a |

`research/shared/mas_agents.py` is the boundary: agents are constructed with an
empty peer registry, so no `has_peer()` guard fires and only `.content` crosses
into a baseline. A refusal therefore arrives as prose, and the baseline's own
dict / log / peer channel carries it or fails to.

## Layout

```
agents/            MAS specialist agents (shared by every configuration)
core/              typed message schema, causally-ordered shared log
research/          the four baseline configurations
research/shared/   mas_agents.py, data_explorer.py, usage_meter.py
tests/             benchmark runners and the ground-truth checkers
results/json/      per-query scored output, one file per configuration
results/traces/    per-query execution traces (Appendix E format)
results/tables/    FR/UAA, turn-and-cost, trap-subtype, DA distribution
```

## Metrics

- **FR** — Fact Recall: ground-truth facts recovered (119 facts; 92 single-hop, 27 multi-hop)
- **UAA** — Unanswerability Acknowledgment Accuracy: planted unanswerable
  sub-questions correctly refused (25 traps; 20 schema-absence, 5 missing-detail)

Scoring is deterministic — checker functions against database-derived ground
truth, no LLM judge. `score_query()` unwraps display line-wrapping before phrase
matching; without that, a correct refusal split across a line break scores zero.

## Running

```bash
make setup                 # virtualenv + pinned dependencies
cp .env.example .env       # fill in credentials and data paths
make check                 # verify data, models and LLM endpoint
make all                   # run all five, score, build tables and traces
```

`./run.sh` does the same without make; `./run.sh --parallel` runs the five
configurations at once. See **[REPRODUCING.md](REPRODUCING.md)** for what you
need to supply, what each step produces, and the one configuration flag that
changes the numbers.

Instrumentation (`research/shared/usage_meter.py`) patches the OpenAI SDK at one
point, so tokens, LLM calls and turns are counted identically for every
configuration including cheap gates.
