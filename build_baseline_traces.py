#!/usr/bin/env python3
"""
build_baseline_traces.py

Appendix E per-query traces for the four baseline configurations.

Appendix E shows a dialogue-act trace for Structured MAS. The baselines have no
dialogue acts — that absence is the ablation variable — so a DA trace cannot be
produced for them. What each one does have is its own native execution record,
and showing those side by side is the point: it makes visible what each
architecture can and cannot represent.

  Unstructured MAS  causally-ordered log with directed peer exchanges. Entries
                    are sender/content, or sender="A→B" with request/response.
                    No da_type, no uncertainty_flags — a refusal is prose.
  Blackboard MAS    orchestrator routing decision, then the ordered sequence of
                    slot writes. Agents never address each other; each write
                    overwrites its key. There is no message history.
  Unstr.-Blackboard free-text log entries, sender/content only.
  Single Agent      ReAct iterations: tool calls and results inside one context
                    window. No inter-agent messages exist at all.

Sources are each system's own JSONL logger, matched to the run behind the
reported numbers — not the newest file on disk, since later probe runs share the
same directories.

Output: sigdial_json_repro/appendix_e_traces_baselines.txt
"""
import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, os.path.join(_ROOT, "research"),
           os.path.join(_ROOT, "research", "single_agent_baseline"),
           os.path.join(_ROOT, "research", "shared"),
           os.path.join(_ROOT, "analytics"), os.path.join(_ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = os.path.join(_ROOT, "sigdial_json_repro", "appendix_e_traces_baselines.txt")

APPENDIX_E = [
    ("N9",  "Energy footprint and billing-cost trap"),
    ("N52", "User profile and core-hour billing trap"),
    ("N47", "Year-over-year spike and thermal/OS kernel traps"),
    ("N24", "VQE benchmark and macOS watch trap"),
    ("N26", "CB vs. MB joint risk at 96 nodes"),
    ("N41", "CB 192-node 2023 spike (chain query)"),
    ("N46", "usr_1898 profile and institutional affiliation trap"),
]

# label -> (jsonl log, scored json, note on what the record can represent)
SYSTEMS = [
    ("Unstructured MAS",
     "logs/unstructured_a2a/unstructured_a2a_20260808_144324.jsonl",
     "una2a_metered.json",
     "causal log + directed peer exchanges; free-text, no typed fields"),
    ("Blackboard MAS",
     "logs/blackboard/blackboard_20260808_144323.jsonl",
     "bb_metered.json",
     "orchestrator routing + slot writes; no peer messaging, no history"),
    ("Unstr.-Blackboard MAS",
     "logs/unstructured/unstructured_20260808_194929.jsonl",
     "unbb_eq.json",
     "free-text log entries; equalized context budget and routing"),
    ("Single Agent",
     "logs/single_agent/both_4_1/single_agent_20260808_144324.jsonl",
     "sa_metered.json",
     "ReAct iterations in one context window; no inter-agent messages"),
]


def clip(s, n=110):
    return re.sub(r"\s+", " ", str(s))[:n]


def load_queries():
    from run_n_queries import (ALL_QUERIES, BATCH4_QUERIES, BATCH5_QUERIES,
                               BATCH6_QUERIES, DOMAIN_QUERIES)
    allq = (ALL_QUERIES + BATCH4_QUERIES + BATCH5_QUERIES
            + DOMAIN_QUERIES + BATCH6_QUERIES)
    return {q["id"]: q["query"] for q in allq}


def index_by_query(path, kind):
    """Map query-text prefix -> record, from a system's own JSONL."""
    p = os.path.join(_ROOT, path)
    if not os.path.exists(p):
        return {}
    out, cur = {}, None
    if kind == "single":
        steps = []
        for line in open(p, encoding="utf-8"):
            try:
                e = json.loads(line)
            except Exception:
                continue
            ev = e.get("event")
            if ev == "QUERY_START":
                if cur:
                    out[cur[:60]] = steps
                cur, steps = e.get("query", ""), []
            elif ev in ("TOOL_CALL", "TOOL_RESULT", "LLM_STEP"):
                steps.append(e)
        if cur:
            out[cur[:60]] = steps
        return out

    for line in open(p, encoding="utf-8"):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") == "QUERY_START":
            cur = e.get("query", "")
        elif e.get("event") == "FINAL_ANSWER" and cur:
            out[cur[:60]] = e
        elif e.get("event") == "ROUTE" and cur:
            out.setdefault(cur[:60] + "::route", e.get("agents"))
    return out


def render(label, rec, route, kind):
    L = []
    if rec is None:
        return ["  [no record found for this query in the source log]"]

    if kind == "blackboard":
        L.append(f"  orchestrator  ROUTE -> {route}")
        for w in rec.get("blackboard_log", []):
            L.append(f"  {w['agent']:14s} WRITE {w['slot']:14s} {clip(w.get('value_preview',''), 88)}")
        L.append("  (no peer messaging; each write overwrites its key, no history retained)")
    elif kind in ("a2a", "log"):
        for m in rec.get("conversation_log", []):
            if "request" in m:
                frm, to = (m["sender"].split("→") + [""])[:2]
                L.append(f"  {frm} --> {to}   REQUEST  {clip(m['request'], 84)}")
                L.append(f"  {to} --> {frm}   RESPONSE {clip(m['response'], 84)}")
            else:
                L.append(f"  {m['sender']:14s} {clip(m.get('content',''), 92)}")
        if kind == "a2a":
            L.append("  (directed peer exchanges present; content is free text, no da_type)")
        else:
            L.append("  (append-only free text; no directed messaging, no typed fields)")
    else:  # single agent
        step = 0
        for e in rec:
            if e.get("event") == "LLM_STEP":
                step += 1
                L.append(f"  -- iteration {step}")
            elif e.get("event") == "TOOL_CALL":
                L.append(f"     CALL   {e.get('tool_name')}({clip(e.get('arguments'), 74)})")
            elif e.get("event") == "TOOL_RESULT":
                L.append(f"     RESULT {clip(e.get('result'), 88)}")
        L.append("  (single context window; no inter-agent messages exist)")
    return L


def main():
    qtext = load_queries()
    scored = {}
    for label, _, sf, _ in SYSTEMS:
        p = os.path.join(_ROOT, "sigdial_json_repro", sf)
        if os.path.exists(p):
            scored[label] = {q["query_id"]: q
                             for q in json.load(open(p, encoding="utf-8"))["queries"]}

    idx = {}
    for label, log, _, _ in SYSTEMS:
        kind = ("single" if "single_agent" in log else
                "blackboard" if "blackboard" in log else
                "a2a" if "a2a" in log else "log")
        idx[label] = (index_by_query(log, kind), kind)

    lines = []
    w = lines.append
    w("=" * 78)
    w("APPENDIX E — PER-QUERY TRACES, BASELINE CONFIGURATIONS")
    w("=" * 78)
    w("")
    w("The baselines emit no dialogue acts; that absence is the ablation")
    w("variable. Shown instead is each system's own execution record, in the")
    w("form its architecture can represent.")
    w("")
    for label, log, sf, note in SYSTEMS:
        w(f"  {label:24s} {note}")
        w(f"  {'':24s} source: {log}")
        w(f"  {'':24s} scores: sigdial_json_repro/{sf}")
    w("")

    for qid, title in APPENDIX_E:
        w("")
        w("=" * 78)
        w(f"{qid}: {title}")
        w("=" * 78)
        w("")
        w("QUERY")
        for c in re.findall(r".{1,74}(?:\s|$)", qtext.get(qid, "")):
            if c.strip():
                w("  " + c.strip())
        w("")
        for label, log, sf, _ in SYSTEMS:
            recs, kind = idx[label]
            key = qtext.get(qid, "")[:60]
            rec = recs.get(key)
            route = recs.get(key + "::route")
            q = scored.get(label, {}).get(qid)
            fr = sum(x["score"] for x in q["facts"]) if q else 0
            ua = sum(t["score"] for t in q["traps"]) if q else 0
            nf = len(q["facts"]) if q else 0
            nt = len(q["traps"]) if q else 0
            w("-" * 78)
            score = f"FR {fr:.0f}/{nf}" + (f", UAA {ua:.0f}/{nt}" if nt else "")
            w(f"{label}   ({score})")
            w("")
            for l in render(label, rec, route, kind):
                w(l)
            w("")

    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"wrote {OUT}  ({len(lines)} lines, {os.path.getsize(OUT):,} bytes)")
    print()
    print(f"{'query':7s}" + "".join(f"{l[:14]:>16s}" for l, *_ in SYSTEMS))
    for qid, _ in APPENDIX_E:
        row = f"{qid:7s}"
        for label, *_ in SYSTEMS:
            q = scored.get(label, {}).get(qid)
            if q:
                fr = sum(x["score"] for x in q["facts"]); nf = len(q["facts"])
                ua = sum(t["score"] for t in q["traps"]); nt = len(q["traps"])
                row += f"{f'{fr:.0f}/{nf}' + (f' {ua:.0f}/{nt}' if nt else ''):>16s}"
        print(row)


if __name__ == "__main__":
    main()
