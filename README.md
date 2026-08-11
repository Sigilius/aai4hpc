# AAI4HPC — Multi-Agent Architectures for HPC Operational Analytics

Code and results for the biaxial ablation over agent-to-agent communication
structure and shared-state organisation, evaluated on 55 multi-intent queries
against the Fugaku job telemetry export (25.8M records).

## Configurations

All five run the **same specialist agent implementations** (`agents/`). Only the
substrate carrying results between agents differs — that is the ablation
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
cp .env.example .env          # add AZURE_OPENAI_API_KEY, FUGAKU_DATA_PATH, MODELS_PATH
pip install -r requirements.txt

python3 tests/run_metered.py mas all          # also: blackboard | unstructured | a2a | single
python3 score_run.py logs/metered_mas.log "Structured MAS" results/json/mas.json --mas
python3 build_results_sheet.py
```

Instrumentation (`research/shared/usage_meter.py`) patches the OpenAI SDK at one
point, so tokens, LLM calls and turns are counted identically for every
configuration including cheap gates.
