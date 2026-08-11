#!/usr/bin/env python3
"""
score_all.py

Score every run log produced by `make run-all` into results/json/.

Reads the standard log names the Makefile writes and emits one scored JSON per
configuration, in the same schema as every other result file in this repo:
`system`, `log_file`, `queries[]` (each with facts, traps, fc, uaa) and
`aggregate`.

Skips configurations whose log is absent, so it is safe to run after scoring
only some of them.

Usage:
    python3 score_all.py
"""
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_ROOT, "results", "json")

# (log basename, display name, output file, is_mas)
RUNS = [
    ("metered_mas.log",          "Structured MAS",              "mas.json",                 True),
    ("metered_a2a.log",          "Unstructured MAS",            "unstructured.json",        False),
    ("metered_blackboard.log",   "Blackboard MAS",              "blackboard.json",          False),
    ("metered_unstructured.log", "Unstructured-Blackboard MAS", "unstructured_no_a2a.json", False),
    ("metered_single.log",       "Single Agent",                "sa.json",                  False),
]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    scored = skipped = 0

    for log, name, out, is_mas in RUNS:
        path = os.path.join(_ROOT, "logs", log)
        if not os.path.exists(path):
            print(f"  skip {name:28s} (no logs/{log})")
            skipped += 1
            continue

        cmd = [sys.executable, os.path.join(_ROOT, "score_run.py"),
               path, name, os.path.join(OUT, out)]
        if is_mas:
            cmd.append("--mas")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAIL {name}: {r.stderr.strip().splitlines()[-1] if r.stderr else '?'}")
            continue
        # score_run prints its own summary block; surface the two headline lines.
        for line in r.stdout.splitlines():
            if line.strip().startswith(("FR  (total)", "UAA (trap-lvl)", "Answered")):
                print(f"  {name:28s} {line.strip()}")
        scored += 1

    print(f"\nscored {scored}, skipped {skipped} -> {OUT}")
    if scored:
        print("next: python3 build_results_sheet.py   (writes results/tables/)")


if __name__ == "__main__":
    main()
