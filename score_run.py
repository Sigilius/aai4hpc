#!/usr/bin/env python3
"""
score_run.py — Score one run log against the 55-query benchmark ground truth.

Thin wrapper around generate_sigdial_json.py: reuses its log parser and
scoring logic (everything above the "# Main" section) but takes the log
path and output path from the command line instead of the hardcoded
/tmp/*.log constants.

Usage:
    python3 score_run.py <log_file> <system_display_name> <out.json> [--mas]
"""

import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_GEN = os.path.join(_ROOT, "generate_sigdial_json.py")

# Exec generate_sigdial_json.py up to (but not including) its "# Main" block,
# so we get parse_log / score_query / build_system_json without triggering the
# hardcoded 5-system run.
_src = open(_GEN, encoding="utf-8").read()
_cut = _src.index("# Main\n")
_cut = _src.rindex("# " + "─" * 5, 0, _cut)  # start of the Main banner comment
_ns = {"__file__": _GEN, "__name__": "_gen_sigdial"}
exec(compile(_src[:_cut], _GEN, "exec"), _ns)

build_system_json = _ns["build_system_json"]
ALL_Q = _ns["ALL_Q"]


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    log_file, display, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    is_mas = "--mas" in sys.argv[4:]

    data = build_system_json(display, log_file, is_mas)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    agg = data["aggregate"]
    missing = [q["query_id"] for q in data["queries"] if not q["answer"].strip()]

    print(f"\n{'='*62}")
    print(f"  {display}")
    print(f"  log: {log_file}")
    print(f"{'='*62}")
    print(f"  Answered      : {agg['completion_rate']['answered']}/{agg['completion_rate']['total']}"
          f"  ({agg['completion_rate']['pct']}%)")
    if missing:
        print(f"  MISSING       : {', '.join(missing)}")
    print(f"  FR  (indep)   : {agg['fc_indep']['correct']:.1f}/{agg['fc_indep']['total']}"
          f"  ({agg['fc_indep']['pct']}%)")
    print(f"  FR  (chain)   : {agg['fc_interdep']['correct']:.1f}/{agg['fc_interdep']['total']}"
          f"  ({agg['fc_interdep']['pct']}%)")
    print(f"  FR  (total)   : {agg['fc_total']['correct']:.1f}/{agg['fc_total']['total']}"
          f"  ({agg['fc_total']['pct']}%)")
    print(f"  UAA (trap-lvl): {agg['uaa']['correct']:.1f}/{agg['uaa']['total']}"
          f"  ({agg['uaa']['pct']}%)")
    print(f"  UA  (qry-lvl) : {agg['ua']['correct']:.1f}/{agg['ua']['total']}"
          f"  ({agg['ua']['pct']}%)")
    print(f"  written       : {out_path}\n")


if __name__ == "__main__":
    main()
