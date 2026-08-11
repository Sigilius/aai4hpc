#!/usr/bin/env python3
"""
build_per_query_table.py

Join per-query scores with per-query LLM cost into one table across systems.

Columns
-------
query_id, system, hop           hop = single | multi (the benchmark's structural
                                partition: N1-N35/D1-D5 vs N41-N55)
n_facts, n_facts_ind,           NOTE: the benchmark labels hop structure per
n_facts_dep                     QUERY, not per fact. Every fact in a chain query
                                is a dependent fact and every fact in an
                                independent query is an independent one, so these
                                two columns are the fact count or zero. They are
                                emitted separately because the aggregate FR-Single
                                / FR-Multi split sums exactly these columns.
n_traps, n_traps_ind,           Same partition applied to traps.
n_traps_dep
n_traps_schema,                 Trap subtype. The 5 missing-detail traps are the
n_traps_missingdetail           client-platform ones (macOS/Windows); everything
                                else is schema-absence.
facts_recalled                  FR numerator for this query.
traps_acknowledged              UAA numerator for this query.
llm_calls, turns                turns = agent activations. For the MAS this is
                                the DA-trace hop count (every DA= line except
                                USER_QUERY/TERMINATE, per gen_hop_breakdown.py),
                                read from the run log. For the MAS baselines it
                                is the number of agent-function invocations. For
                                the single agent it is its ReAct iteration count.
                                Not architecture-neutral — compare llm_calls for
                                a like-for-like effort measure.
prompt_tokens, completion_tokens, total_tokens

Usage:
    python3 build_per_query_table.py            # writes CSV + prints summary
"""
import csv
import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))

INDEP = set([f"N{i}" for i in range(1, 36)] + [f"D{i}" for i in range(1, 6)])
MISSING_DETAIL = {
    ("N16", "macOS_watch_REJECT"), ("N17", "WinTaskMgr_REJECT"),
    ("N21", "Windows_dir_REJECT"), ("N24", "macOS_watch_REJECT"),
    ("N31", "macOS_AM_REJECT"),
}

# system tag -> (scored json, usage jsonl, run log for DA-trace hops)
SYSTEMS = [
    ("MAS",   "sigdial_json_repro/mas_metered.json",   "logs/usage_mas.jsonl",          "logs/metered_mas.log"),
    ("UNMAS", "sigdial_json_repro/una2a_metered.json", "logs/usage_a2a.jsonl",          "logs/metered_a2a.log"),
    ("BB",    "sigdial_json_repro/bb_metered.json",    "logs/usage_blackboard.jsonl",   "logs/metered_blackboard.log"),
    ("UNBB",  "sigdial_json_repro/unbb_eq.json",  "logs/usage_unstructured_eq.jsonl", "logs/metered_unbb_eq.log"),
    ("SA",    "sigdial_json_repro/sa_metered.json",    "logs/usage_single.jsonl",       "logs/metered_single.log"),
]

_DA = re.compile(r"\bDA=([A-Z_]+)\b")
_EXCLUDE_DA = {"USER_QUERY", "TERMINATE"}
_DUR = re.compile(r"^── ANSWER \(([\d.]+)s\) ──", re.M)


def wall_clock(log_path: str) -> dict[str, float]:
    """Per-query wall-clock seconds, read from the run log's ANSWER banner."""
    if not os.path.exists(log_path):
        return {}
    txt = open(log_path, encoding="utf-8", errors="replace").read()
    parts = re.split(r"^  \[([ND]\d+)\]", txt, flags=re.M)
    out = {}
    for i in range(1, len(parts), 2):
        m = _DUR.search(parts[i + 1])
        if m:
            out[parts[i]] = float(m.group(1))
    return out


def da_trace_hops(log_path: str) -> dict[str, int]:
    """Hop count per query from a MAS run log — the published hop definition."""
    if not os.path.exists(log_path):
        return {}
    txt = open(log_path, encoding="utf-8", errors="replace").read()
    parts = re.split(r"^  \[([ND]\d+)\]", txt, flags=re.M)
    out = {}
    for i in range(1, len(parts), 2):
        qid, body = parts[i], parts[i + 1]
        out[qid] = sum(1 for m in _DA.finditer(body) if m.group(1) not in _EXCLUDE_DA)
    return out


def load_usage(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out[r["query_id"]] = r
    return out


def main() -> None:
    rows = []
    for tag, score_f, usage_f, log_f in SYSTEMS:
        score_p = os.path.join(_ROOT, score_f)
        if not os.path.exists(score_p):
            print(f"  skip {tag}: missing {score_f}")
            continue

        data = json.load(open(score_p, encoding="utf-8"))
        usage = load_usage(os.path.join(_ROOT, usage_f))
        hops = da_trace_hops(os.path.join(_ROOT, log_f)) if tag == "MAS" else {}
        durs = wall_clock(os.path.join(_ROOT, log_f))

        for q in data["queries"]:
            qid = q["query_id"]
            ind = qid in INDEP
            nf, nt = len(q["facts"]), len(q["traps"])
            md = sum(1 for t in q["traps"] if (qid, t["label"]) in MISSING_DETAIL)
            u = usage.get(qid, {})
            turns = hops.get(qid, u.get("turns", "")) if tag == "MAS" else u.get("turns", "")

            rows.append({
                "query_id": qid, "system": tag, "hop": "single" if ind else "multi",
                "n_facts": nf,
                "n_facts_ind": nf if ind else 0,
                "n_facts_dep": 0 if ind else nf,
                "n_traps": nt,
                "n_traps_ind": nt if ind else 0,
                "n_traps_dep": 0 if ind else nt,
                "n_traps_schema": nt - md,
                "n_traps_missingdetail": md,
                "facts_recalled": round(sum(x["score"] for x in q["facts"]), 2),
                "traps_acknowledged": round(sum(t["score"] for t in q["traps"]), 2),
                "llm_calls": u.get("llm_calls", ""),
                "turns": turns,
                "prompt_tokens": u.get("prompt_tokens", ""),
                "completion_tokens": u.get("completion_tokens", ""),
                "total_tokens": u.get("total_tokens", ""),
                "tokens_per_call": (round(u["total_tokens"] / u["llm_calls"], 1)
                                    if u.get("llm_calls") else ""),
                "wall_clock_s": durs.get(qid, ""),
            })

    if not rows:
        print("No scored runs found — run score_run.py first.")
        sys.exit(1)

    out = os.path.join(_ROOT, "sigdial_json_repro", "per_query_full.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} — {len(rows)} rows\n")

    # Per-system totals
    print(f"{'system':7s}{'FR':>12s}{'UAA':>11s}{'llm_calls':>11s}{'turns':>9s}"
          f"{'tokens':>12s}{'tok/query':>11s}")
    print("-" * 63)
    for tag, *_ in SYSTEMS:
        rs = [r for r in rows if r["system"] == tag]
        if not rs:
            continue
        fr = sum(r["facts_recalled"] for r in rs)
        frt = sum(r["n_facts"] for r in rs)
        ua = sum(r["traps_acknowledged"] for r in rs)
        uat = sum(r["n_traps"] for r in rs)
        num = lambda k: [r[k] for r in rs if isinstance(r[k], (int, float))]
        calls, turns, toks = num("llm_calls"), num("turns"), num("total_tokens")
        print(f"{tag:7s}{fr:6.0f}/{frt:<5d}{ua:5.0f}/{uat:<5d}"
              f"{sum(calls):11d}{sum(turns):9d}{sum(toks):12,d}"
              f"{(sum(toks)//len(toks) if toks else 0):11,d}")


if __name__ == "__main__":
    main()
