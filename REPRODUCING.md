# Reproducing the results

Five agent configurations, 55 multi-intent queries, scored deterministically
against the Fugaku job telemetry export. Measured end to end: **72 minutes**
sequential, about **25 minutes** if you run the configurations in parallel.

| Configuration | Wall-clock, 55 queries | Median per query |
|---|---|---|
| Structured MAS | 23.6 min | 23.7 s |
| Unstructured MAS | 17.6 min | 19.2 s |
| Blackboard MAS | 11.5 min | 12.2 s |
| Unstructured-Blackboard MAS | 12.6 min | 12.7 s |
| Single Agent | 6.8 min | 6.9 s |

---

## 1. What you need first

Three things are not in this repository and cannot be — they are large,
licensed, or secret.

| | What | Where it goes |
|---|---|---|
| **Fugaku telemetry** | the F-Data job export, parquet, 25.8M rows | `FUGAKU_DATA_PATH` |
| **Predictor artifacts** | pickled LightGBM models for the PA agent | `MODELS_PATH`, `PREPARED_PATH` |
| **An LLM endpoint** | Azure OpenAI (or OpenAI-compatible) with `gpt-4o` and `gpt-4o-mini` | `AZURE_OPENAI_*` |

Without the telemetry the SQL agent falls back to a small sample database and
every fact-recall number will be wrong. `make check` catches that case.

Documentation retrieval needs no vector database — it is BM25 over cached numpy
embeddings — but it does need two asset files, `data/doc_embeddings.npy` and the
chunked manual corpus. Without them DocAgent fails and every documentation
sub-question silently scores zero.

## 2. Setup

```bash
make setup                 # virtualenv + pinned dependencies
cp .env.example .env       # then fill it in
make check                 # verifies data, models and the LLM endpoint
```

`make check` is worth running. It confirms the jobs table has more than a
million rows (so you are on the real export, not the sample), loads the
predictor, and makes one live LLM call. It exits non-zero on any failure rather
than letting a long run fail at query 40.

Dependency versions are pinned because the predictor artifacts are pickled
sklearn and LightGBM models; a different major version will not unpickle.

## 3. Run

```bash
make run-all               # all five, sequential, ~72 min
make run-parallel          # all five at once, ~25 min, needs rate-limit headroom
```

Or one at a time:

```bash
make mas       # Structured MAS
make unmas     # Unstructured MAS
make bb        # Blackboard MAS
make unbb      # Unstructured-Blackboard MAS
make sa        # Single Agent
```

If you would rather not use make:

```bash
./run.sh                   # everything, sequential
./run.sh --parallel        # everything, at once
./run.sh mas unmas         # just these two
```

`run.sh` runs `make check` first and refuses to start if the environment is not
ready.

Each run writes `logs/metered_<system>.log` (the transcript, in the format the
scorer parses) and `logs/usage_<system>.jsonl` (per-query tokens, LLM calls and
turns).

**One configuration flag matters.** Structured MAS is reported with every call
on `gpt-4o`, which is what `MAS_FORCE_MODEL=gpt-4o` in the Makefile sets. Left
unset, the code routes roughly 77% of its calls to `gpt-4o-mini` — the cheap
tier used for extraction and delegation gates — and both FR and cost change
materially. The baselines are single-tier `gpt-4o` throughout.

## 4. Score and build the tables

```bash
make score                 # logs/ -> results/json/
make tables                # results/json/ -> results/tables/
make traces                # per-query execution traces -> results/traces/
```

Or `make all` to do the run, the scoring and both build steps in one go.

Scoring is deterministic. Each ground-truth fact has a checker function that
extracts a value from the free-text answer and compares it to the
database-derived truth within a calibrated tolerance; each planted trap has a
checker that looks for an explicit refusal and for fabricated values. No LLM
judge is involved, so scores are reproducible from a log alone.

## 5. What you get

```
results/json/     one scored file per configuration: per-query facts, traps,
                  the full answer, and the aggregate
results/tables/   FR/UAA, turn-and-cost, trap subtype, DA distribution
                  (results_tables.xlsx plus a CSV per sheet)
results/traces/   per-query execution traces for the queries analysed in the
                  paper's appendix
```

### Metrics

- **FR** — Fact Recall. 119 ground-truth facts: 92 answerable within one
  agent's tool scope, 27 requiring a prior agent's output.
- **UAA** — Unanswerability Acknowledgment Accuracy. 25 planted sub-questions
  the telemetry cannot answer: 20 schema-absence (billing, temperature, GPU,
  latency, kernel version, cache counters, CO2, affiliation) and 5
  missing-detail. A trap scores 1 only if the system explicitly declines; a
  hedge that neither answers nor declines scores 0.

Both are reported in aggregate and split by hop structure, which is a partition
of the same scores rather than a separate measure.

## 6. Expect run-to-run variation

Temperature is 0, but these systems are not deterministic in practice. Across
repeated Structured MAS runs of identical code we observed FR moving 82–93 of
119 and UAA 22–23 of 25. Treat a single run as one sample: for any comparison
that turns on a few facts or traps, run three times and report a mean with the
spread.

## 7. Notes for anyone reading the code

**All five configurations run the same agents.** `agents/` holds the specialist
implementations; `research/shared/mas_agents.py` exposes them to the baselines.
Each agent is constructed with an empty peer registry, so no `has_peer()` guard
fires, nothing delegates, and only `.content` crosses the boundary — `da_type`,
`uncertainty_flags` and `delegation_trigger` are dropped. What differs between
configurations is the substrate that carries a result from one agent to the
next, and nothing else.

The one exception is Unstructured MAS, whose defining property is directed peer
messaging. It gets the same PA agent with real `sql_agent` and `doc_agent`
peers, wrapped in a shim that rewrites every reply as a bare `INFORM` with no
flags. The wiring is identical to Structured MAS; the typed contract is not.

**Cost accounting is uniform by construction.**
`research/shared/usage_meter.py` patches the OpenAI SDK at its single entry
point, so tokens and calls are counted the same way everywhere, including the
cheap gates a per-agent counter would miss. `turns` is *not* comparable across
architectures — it is DA-trace hops for the MAS, agent activations for the MAS
baselines, and ReAct iterations for the single agent. Compare LLM calls instead.

**The scorer unwraps line breaks before matching.** Answers are printed through
`textwrap.fill(..., 72)`, and the checkers match literal phrases. A refusal that
happens to wrap between two words — `"...is not\navailable in the dataset"` —
scored zero before this was fixed. The bias only ever turned correct answers
into misses, so any figure computed without `unwrap()` is a floor.
