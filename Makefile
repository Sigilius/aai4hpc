# AAI4HPC — reproduction pipeline
#
#   make setup      one-time: virtualenv + dependencies
#   make check      verify credentials, dataset and model artifacts are reachable
#   make run-all    run all five configurations (sequential, ~3h)
#   make score      score every run log into results/json/
#   make tables     build results/tables/ from the scored JSON
#   make traces     build results/traces/ per-query execution traces
#   make all        run-all + score + tables + traces
#
# Individual configurations, if you want them one at a time:
#   make mas   make unmas   make bb   make unbb   make sa
#
# Each configuration is ~40 minutes. They are independent, so
# `make run-parallel` starts all five at once if your rate limit allows it.

PY      := .venv/bin/python
PIP     := .venv/bin/pip
RUNNER  := tests/run_metered.py
LOGS    := logs

# Structured MAS is reported with every call on gpt-4o. Without this the code
# routes roughly 77% of its calls to gpt-4o-mini, which changes the numbers.
MAS_ENV := MAS_FORCE_MODEL=gpt-4o

.PHONY: all setup check run-all run-parallel score tables traces clean \
        mas unmas bb unbb sa

all: run-all score tables traces

# ── Setup ─────────────────────────────────────────────────────────────────────

setup:
	python3 -m venv .venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt
	@echo "done. now: cp .env.example .env && edit it, then 'make check'"

check:
	@test -f .env || { echo "FAIL: no .env — cp .env.example .env and fill it in"; exit 1; }
	@$(PY) -c "import sys; sys.path[:0]=['.','research/shared','analytics']; \
from dotenv import load_dotenv; load_dotenv(); \
from shared.db import get_connection, data_source_label; \
c=get_connection(); n=c.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]; \
print(f'  data   : {data_source_label()}'); \
print(f'  rows   : {n:,}'); \
assert n > 1_000_000, 'jobs table looks like the sample DB, not the Fugaku export'"
	@$(PY) -c "import sys,os; sys.path[:0]=['.','analytics']; \
from dotenv import load_dotenv; load_dotenv(); \
from predict import Predictor; Predictor(); print('  models : loaded')"
	@$(PY) -c "import sys; sys.path.insert(0,'.'); \
from dotenv import load_dotenv; load_dotenv(); import config; from openai import OpenAI; \
r=OpenAI(base_url=config.AZURE_OPENAI_ENDPOINT, api_key=config.AZURE_OPENAI_API_KEY) \
 .chat.completions.create(model=config.MODEL, messages=[{'role':'user','content':'ok'}], max_tokens=3); \
print(f'  llm    : {config.MODEL} responding')"
	@echo "ready — 'make run-all' or 'make mas'"

# ── Configurations ────────────────────────────────────────────────────────────
# Log names are what score_all.py and the table builders expect; changing them
# means changing those too.

mas:
	@mkdir -p $(LOGS)
	$(MAS_ENV) $(PY) -u $(RUNNER) mas all 2>&1 | tee $(LOGS)/metered_mas.log

unmas:
	@mkdir -p $(LOGS)
	$(PY) -u $(RUNNER) a2a all 2>&1 | tee $(LOGS)/metered_a2a.log

bb:
	@mkdir -p $(LOGS)
	$(PY) -u $(RUNNER) blackboard all 2>&1 | tee $(LOGS)/metered_blackboard.log

unbb:
	@mkdir -p $(LOGS)
	$(PY) -u $(RUNNER) unstructured all 2>&1 | tee $(LOGS)/metered_unstructured.log

sa:
	@mkdir -p $(LOGS)
	$(PY) -u $(RUNNER) single all 2>&1 | tee $(LOGS)/metered_single.log

run-all: mas unmas bb unbb sa

run-parallel:
	@mkdir -p $(LOGS)
	@echo "starting five configurations at once — watch $(LOGS)/*.log"
	@$(MAS_ENV) $(PY) -u $(RUNNER) mas all          > $(LOGS)/metered_mas.log 2>&1 &
	@$(PY) -u $(RUNNER) a2a all                     > $(LOGS)/metered_a2a.log 2>&1 &
	@$(PY) -u $(RUNNER) blackboard all              > $(LOGS)/metered_blackboard.log 2>&1 &
	@$(PY) -u $(RUNNER) unstructured all            > $(LOGS)/metered_unstructured.log 2>&1 &
	@$(PY) -u $(RUNNER) single all                  > $(LOGS)/metered_single.log 2>&1 &
	@wait
	@echo "all five finished"

# ── Results ───────────────────────────────────────────────────────────────────

score:
	$(PY) score_all.py

tables:
	$(PY) build_results_sheet.py

traces:
	$(PY) build_appendix_e_traces.py
	$(PY) build_baseline_traces.py

clean:
	rm -rf $(LOGS)
	@echo "logs removed; results/ kept"
