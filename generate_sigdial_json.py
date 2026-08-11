#!/usr/bin/env python3
"""
generate_sigdial_json.py — Generate 5 JSON evaluation files for SIGDIAL 2026.

Each file contains per-query answers, fact/trap scores, and aggregate metrics
for one of the 5 HPC systems evaluated on the Fugaku dataset.
"""

import sys
import re
import json
import os

# Import GT and scoring helpers from score_metrics.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests"))
from score_metrics import GT, _check_rate, _check_count, _check_contains

# ─────────────────────────────────────────────────────────────────────────────
# Query sets
# ─────────────────────────────────────────────────────────────────────────────

INDEP_IDS = (
    [f"N{i}" for i in range(1, 36)] +   # N1–N35
    [f"D{i}" for i in range(1, 6)]       # D1–D5
)
INTERDEP_IDS = [f"N{i}" for i in range(41, 56)]  # N41–N55
ALL_Q = INDEP_IDS + INTERDEP_IDS        # 55 total

# ─────────────────────────────────────────────────────────────────────────────
# Log parser
# ─────────────────────────────────────────────────────────────────────────────

SEP_RE   = re.compile(r'^={40,}')
QID_RE   = re.compile(r'\[([ND]\d+)\]')
ANS_RE   = re.compile(r'^── ANSWER \([\d\.]+s\) ──')


def parse_log(path: str) -> dict:
    """
    Parse a run log and return a dict keyed by query_id with:
        {
            "query_text": str,
            "da_trace":   [str, ...],
            "answer":     str,
        }
    """
    results = {}

    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    n = len(lines)
    i = 0

    while i < n:
        stripped = lines[i].rstrip()

        # Look for an opening separator line
        if SEP_RE.match(stripped):
            sep_start = i
            i += 1

            # Collect lines inside the separator block until the closing ===
            header_lines = []
            while i < n and not SEP_RE.match(lines[i].rstrip()):
                header_lines.append(lines[i].rstrip())
                i += 1

            if not header_lines:
                # Empty block (e.g. the very first global banner)
                continue

            # Check if this block contains a query ID tag
            first_nonempty = next((l for l in header_lines if l.strip()), "")
            m = QID_RE.search(first_nonempty)
            if not m:
                # Not a query block — skip
                continue

            qid = m.group(1)
            closing_sep = i  # index of the closing ===

            # Build query_text: skip the first line ([QID] label),
            # join remaining header lines
            text_lines = header_lines[1:]
            query_text = " ".join(l.strip() for l in text_lines if l.strip())
            # If query text is empty, include everything after the [QID] tag on line 1
            if not query_text:
                after_tag = re.sub(r'\[.*?\]', '', first_nonempty).strip()
                # Remove leading " — " separator if present
                after_tag = re.sub(r'^[-—\s]+', '', after_tag).strip()
                query_text = after_tag

            i += 1  # Move past the closing separator

            # Now collect DA trace lines and answer until the next === separator
            da_trace_lines = []
            answer_lines = []
            in_answer = False

            while i < n:
                line_raw = lines[i].rstrip()

                if SEP_RE.match(line_raw):
                    # Start of next block — stop here
                    break

                if ANS_RE.match(line_raw):
                    in_answer = True
                    i += 1
                    continue

                if in_answer:
                    answer_lines.append(line_raw)
                else:
                    da_trace_lines.append(line_raw)

                i += 1

            # Strip trailing blank lines from answer
            while answer_lines and not answer_lines[-1].strip():
                answer_lines.pop()

            # Strip trailing blank lines from da_trace
            while da_trace_lines and not da_trace_lines[-1].strip():
                da_trace_lines.pop()

            # Remove leading blank lines from da_trace
            while da_trace_lines and not da_trace_lines[0].strip():
                da_trace_lines.pop(0)

            results[qid] = {
                "query_text": query_text,
                "da_trace":   da_trace_lines,
                "answer":     "\n".join(answer_lines),
            }

        else:
            i += 1

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

_WS = re.compile(r"[ \t]*\n[ \t]*")


def unwrap(answer: str) -> str:
    """
    Undo the display line-wrapping before phrase matching.

    Answers are printed through textwrap.fill(..., 72) in tests/run_n_queries.py,
    which inserts hard newlines at column 72. The GT checkers then look for
    literal phrases ("not available", "1 node") with plain substring tests, so a
    phrase that straddles a wrap point silently fails to match and a correct
    answer is scored as a miss.

    Observed on the MAS all-4o run: D3 contains "...data is not\navailable in the
    Fugaku dataset", a clean refusal, which scored 0 purely because the wrap fell
    between "not" and "available". The bias is one-directional — wrapping can
    only turn hits into misses — so every FR and UAA figure computed without this
    is a floor rather than an estimate.

    Newlines become single spaces; paragraph structure is not needed by any
    checker, and the sentence-splitting helpers treat a space the same as the
    newline they would otherwise have seen.
    """
    return _WS.sub(" ", answer)


def score_query(qid: str, answer: str) -> tuple[list, list]:
    """Return (facts_list, traps_list) for a given query_id and answer text."""
    answer = unwrap(answer)
    spec = GT.get(qid, {"facts": [], "traps": []})
    facts = []
    for label, checker in spec["facts"]:
        try:
            s = float(checker(answer))
        except Exception:
            s = 0.0
        facts.append({"label": label, "score": round(s, 4)})

    traps = []
    for label, checker in spec["traps"]:
        try:
            s = float(checker(answer))
        except Exception:
            s = 0.0
        traps.append({"label": label, "score": round(s, 4)})

    return facts, traps


def pct_dict(correct: float, total: int) -> dict:
    pct = round(correct / total * 100, 1) if total > 0 else 0.0
    return {"correct": round(correct, 4), "total": total, "pct": pct}


# ─────────────────────────────────────────────────────────────────────────────
# Build one JSON structure for a system
# ─────────────────────────────────────────────────────────────────────────────

def build_system_json(system_display: str, log_file: str, is_mas: bool) -> dict:
    parsed = parse_log(log_file)

    queries = []
    answered_count = 0

    # Aggregate counters
    fc_indep_c = fc_indep_t = 0
    fc_inter_c = fc_inter_t = 0
    uaa_c = uaa_t = 0
    # UA = query-level: queries where ALL traps correctly rejected
    ua_trap_queries = 0
    ua_correct_queries = 0

    for qid in ALL_Q:
        data = parsed.get(qid, None)

        if data is None:
            # Missing query
            query_text = ""
            da_trace = []
            answer = ""
        else:
            query_text = data["query_text"]
            da_trace = data["da_trace"] if is_mas else []
            answer = data["answer"]
            if answer.strip():
                answered_count += 1

        facts, traps = score_query(qid, answer)

        fc_correct = sum(f["score"] for f in facts)
        fc_total   = len(facts)
        uaa_correct = sum(t["score"] for t in traps)
        uaa_total   = len(traps)

        fc_entry  = pct_dict(fc_correct, fc_total)
        uaa_entry = pct_dict(uaa_correct, uaa_total)

        # Accumulate aggregates
        if qid in INDEP_IDS:
            fc_indep_c += fc_correct
            fc_indep_t += fc_total
        elif qid in INTERDEP_IDS:
            fc_inter_c += fc_correct
            fc_inter_t += fc_total

        uaa_c += uaa_correct
        uaa_t += uaa_total

        if uaa_total > 0:
            ua_trap_queries += 1
            if uaa_correct >= uaa_total:
                ua_correct_queries += 1

        queries.append({
            "query_id":   qid,
            "query_text": query_text,
            "da_trace":   da_trace,
            "answer":     answer,
            "facts":      facts,
            "traps":      traps,
            "fc":         fc_entry,
            "uaa":        uaa_entry,
        })

    # Build aggregate
    fc_total_c = fc_indep_c + fc_inter_c
    fc_total_t = fc_indep_t + fc_inter_t

    total_q = len(ALL_Q)
    agg_completion = {
        "answered": answered_count,
        "total": total_q,
        "pct": round(answered_count / total_q * 100, 1) if total_q > 0 else 0.0,
    }

    aggregate = {
        "completion_rate": agg_completion,
        "fc_indep":        pct_dict(fc_indep_c, fc_indep_t),
        "fc_interdep":     pct_dict(fc_inter_c, fc_inter_t),
        "fc_total":        pct_dict(fc_total_c, fc_total_t),
        "uaa":             pct_dict(uaa_c, uaa_t),
        "ua":              pct_dict(ua_correct_queries, ua_trap_queries),
    }

    return {
        "system":    system_display,
        "log_file":  log_file,
        "queries":   queries,
        "aggregate": aggregate,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

SYSTEMS = [
    # (display_name,   log_file,                   output_name, is_mas)
    ("MAS v6",         "/tmp/all55_mas_v6.log",    "mas_v6.json",     True),
    ("MAS v5",         "/tmp/all55_mas_v5.log",    "mas_v5.json",     True),
    ("Single Agent",   "/tmp/all55_sa_v2.log",     "sa.json",         False),
    ("Blackboard",     "/tmp/all55_bb.log",         "blackboard.json", False),
    ("Unstructured",   "/tmp/all55_una2a.log",      "unstructured.json", False),
]

OUT_DIR = "/tmp/sigdial_json"
os.makedirs(OUT_DIR, exist_ok=True)

for display_name, log_file, out_name, is_mas in SYSTEMS:
    print(f"\n{'='*60}")
    print(f"  Processing: {display_name}")
    print(f"  Log:  {log_file}")
    print(f"  Out:  {OUT_DIR}/{out_name}")
    print(f"{'='*60}")

    data = build_system_json(display_name, log_file, is_mas)
    out_path = os.path.join(OUT_DIR, out_name)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    # Validate JSON round-trip
    with open(out_path, encoding="utf-8") as fh:
        loaded = json.load(fh)

    agg = loaded["aggregate"]
    print(f"  Queries parsed: {len(loaded['queries'])}")
    print(f"  Answered:       {agg['completion_rate']['answered']}/{agg['completion_rate']['total']}  ({agg['completion_rate']['pct']}%)")
    print(f"  FC (INDEP):     {agg['fc_indep']['correct']:.1f}/{agg['fc_indep']['total']}  ({agg['fc_indep']['pct']}%)")
    print(f"  FC (INTERDEP):  {agg['fc_interdep']['correct']:.1f}/{agg['fc_interdep']['total']}  ({agg['fc_interdep']['pct']}%)")
    print(f"  FC (TOTAL):     {agg['fc_total']['correct']:.1f}/{agg['fc_total']['total']}  ({agg['fc_total']['pct']}%)")
    print(f"  UAA:            {agg['uaa']['correct']:.1f}/{agg['uaa']['total']}  ({agg['uaa']['pct']}%)")
    print(f"  UA (q-level):   {agg['ua']['correct']:.1f}/{agg['ua']['total']}  ({agg['ua']['pct']}%)")
    print(f"  File size:      {os.path.getsize(out_path):,} bytes")
    print(f"  JSON valid:     YES")

print(f"\n{'='*60}")
print(f"  All 5 files written to {OUT_DIR}/")
print(f"{'='*60}\n")

# Final listing
for _, _, out_name, _ in SYSTEMS:
    p = os.path.join(OUT_DIR, out_name)
    print(f"  {p}  ({os.path.getsize(p):,} bytes)")
