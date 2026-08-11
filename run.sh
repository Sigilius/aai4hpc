#!/usr/bin/env bash
#
# run.sh — end-to-end reproduction, for reviewers who would rather not use make.
#
#   ./run.sh              all five configurations, sequential (~3h), then score
#   ./run.sh --parallel   all five at once (~40min), if your rate limit allows
#   ./run.sh mas unmas    just the named configurations
#
# Configuration names: mas | unmas | bb | unbb | sa
#
# Prerequisites: ./run.sh assumes `make setup` has been run and .env is filled
# in. It stops before doing anything expensive if the environment is not ready.

set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
RUNNER=tests/run_metered.py
mkdir -p logs

# name -> runner argument + log file
declare -A ARG=( [mas]=mas [unmas]=a2a [bb]=blackboard [unbb]=unstructured [sa]=single )
declare -A LOG=( [mas]=metered_mas [unmas]=metered_a2a [bb]=metered_blackboard \
                 [unbb]=metered_unstructured [sa]=metered_single )

die() { echo "error: $*" >&2; exit 1; }

[[ -x "$PY" ]] || die "no virtualenv — run 'make setup' first"
[[ -f .env  ]] || die "no .env — copy .env.example to .env and fill it in"

PARALLEL=0
TARGETS=()
for a in "$@"; do
  case "$a" in
    --parallel) PARALLEL=1 ;;
    mas|unmas|bb|unbb|sa) TARGETS+=("$a") ;;
    *) die "unknown argument '$a' (expected --parallel or a configuration name)" ;;
  esac
done
[[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=(mas unmas bb unbb sa)

echo "checking environment..."
make -s check || die "environment check failed"

run_one() {
  local name=$1
  local env_prefix=""
  # Structured MAS is reported with every call on gpt-4o; the code otherwise
  # routes ~77% of its calls to gpt-4o-mini.
  [[ "$name" == "mas" ]] && env_prefix="MAS_FORCE_MODEL=gpt-4o"
  echo "[$(date +%H:%M:%S)] $name -> logs/${LOG[$name]}.log"
  env $env_prefix $PY -u "$RUNNER" "${ARG[$name]}" all > "logs/${LOG[$name]}.log" 2>&1
  local n
  n=$(grep -c '^── ANSWER' "logs/${LOG[$name]}.log" || true)
  echo "[$(date +%H:%M:%S)] $name done — $n/55 queries answered"
}

if [[ $PARALLEL -eq 1 ]]; then
  for t in "${TARGETS[@]}"; do run_one "$t" & done
  wait
else
  for t in "${TARGETS[@]}"; do run_one "$t"; done
fi

echo
echo "scoring..."
$PY score_all.py
echo
echo "building tables and traces..."
$PY build_results_sheet.py
$PY build_appendix_e_traces.py
$PY build_baseline_traces.py

echo
echo "done."
echo "  results/json/    per-query scored output"
echo "  results/tables/  FR/UAA, turn-and-cost, trap-subtype, DA distribution"
echo "  results/traces/  per-query execution traces"
