#!/usr/bin/env python3
"""
build_appendix_e_traces.py

Regenerate the Appendix E per-query traces for the all-GPT-4o Structured MAS run.

Appendix E of the SIGDial EPIC2 draft shows, for seven queries, the ground-truth
facts and traps, a shaded box with the Structured MAS dialogue-act trace, and a
per-system comparison. This emits the same seven traces from
logs/metered_mas_4o.log — the configuration where every call uses GPT-4o, which
is what the paper states as its control — together with the score that run
achieved, so each trace can be checked against the appendix version.

Traces are the DA= lines as printed by the orchestrator, with the message
preview kept so the delegation content is visible. Scores come from the
post-fix checker (display line-wrapping is unwrapped before phrase matching).

Output: sigdial_json_repro/appendix_e_traces_mas_4o.txt
"""
import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "tests"))

LOG = os.path.join(_ROOT, "logs", "metered_mas_4o.log")
SCORED = os.path.join(_ROOT, "sigdial_json_repro", "mas_4o.json")
OUT = os.path.join(_ROOT, "sigdial_json_repro", "appendix_e_traces_mas_4o.txt")

# The seven queries given per-query analysis in Appendix E, with the appendix's
# own section titles and the figures it reports for Structured MAS.
APPENDIX_E = [
    ("N9",  "Energy footprint and billing-cost trap",            "FR 1/1, UAA 1/1"),
    ("N52", "User profile and core-hour billing trap",           "FR 2/2, UAA 1/1"),
    ("N47", "Year-over-year spike and thermal/OS kernel traps",  "FR 2/2, UAA 2/2"),
    ("N24", "VQE benchmark and macOS watch trap",                "FR 2/2, UAA 1/1"),
    ("N26", "CB vs. MB joint risk at 96 nodes",                  "FR 3/3"),
    ("N41", "CB 192-node 2023 spike (chain query)",              "FR 1/1"),
    ("N46", "usr_1898 profile and institutional affiliation trap", "FR 2/2, UAA 1/1"),
]

_DA_LINE = re.compile(r"^\S+ (?:──▶|◀──) \S+\s+DA=[A-Z_]+")
_PREVIEW = re.compile(r"^\s{2,}(?:──|◀──)")


def blocks(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    parts = re.split(r"^  \[([ND]\d+)\]", txt, flags=re.M)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


def trace_lines(body: str) -> list[str]:
    """DA lines plus their one-line message previews, in causal order."""
    out = []
    for raw in body.splitlines():
        line = raw.rstrip()
        if _DA_LINE.match(line.strip()) or line.strip().startswith(("user ◀──", "gateway ──▶")):
            if "DA=" in line:
                out.append(line.strip())
                continue
        if out and (_PREVIEW.match(line) or line.strip().startswith(("──", "◀──", "classified"))):
            frag = line.strip()
            if frag:
                out.append("      " + frag[:150])
    return out


def query_text(body: str) -> str:
    head = body.split("=" * 40)[0].splitlines()
    return " ".join(l.strip() for l in head[1:] if l.strip())


def main():
    if not os.path.exists(LOG):
        print(f"missing {LOG}")
        sys.exit(1)
    B = blocks(LOG)
    scored = {q["query_id"]: q for q in json.load(open(SCORED, encoding="utf-8"))["queries"]}

    lines = []
    w = lines.append
    w("=" * 78)
    w("APPENDIX E — PER-QUERY TRACES")
    w("Structured MAS, all-GPT-4o configuration")
    w("=" * 78)
    w("")
    w("Source run : logs/metered_mas_4o.log")
    w("Scores     : sigdial_json_repro/mas_4o.json (post-fix checker)")
    w("Config     : every LLM call on gpt-4o; the shipped MAS routes ~77% of")
    w("             calls to gpt-4o-mini, so this is the run that matches the")
    w("             paper's stated control that all configurations use GPT-4o.")
    w("Aggregate  : FR 93/119 (78.2%)   UAA 23/25 (92.0%)")
    w("")
    w("Each block gives the appendix's reported Structured MAS result, the")
    w("result this run achieved, and the full dialogue-act trace.")
    w("")

    for qid, title, reported in APPENDIX_E:
        body = B.get(qid)
        q = scored.get(qid)
        w("")
        w("=" * 78)
        w(f"{qid}: {title}")
        w("=" * 78)
        w("")
        if body is None or q is None:
            w("  [query not present in this run]")
            continue

        w("QUERY")
        for chunk in re.findall(r".{1,74}(?:\s|$)", query_text(body)):
            if chunk.strip():
                w("  " + chunk.strip())
        w("")

        fr_c = sum(x["score"] for x in q["facts"])
        ua_c = sum(t["score"] for t in q["traps"])
        w("GROUND TRUTH")
        for f in q["facts"]:
            w(f"  fact  {f['label']:28s} -> {'RECOVERED' if f['score'] else 'missed'}")
        for t in q["traps"]:
            w(f"  trap  {t['label']:28s} -> {'REFUSED' if t['score'] else 'not refused'}")
        if not q["facts"] and not q["traps"]:
            w("  (none)")
        w("")
        w(f"APPENDIX REPORTS : {reported}")
        w(f"THIS RUN         : FR {fr_c:.0f}/{len(q['facts'])}"
          + (f", UAA {ua_c:.0f}/{len(q['traps'])}" if q["traps"] else ""))
        w("")

        tl = trace_lines(body)
        das = [l for l in tl if "DA=" in l]
        w(f"STRUCTURED MAS TRACE  ({len(das)} typed messages)")
        w("  " + "-" * 74)
        for l in tl:
            w("  " + l)
        w("  " + "-" * 74)
        w("")
        w("FINAL ANSWER")
        for chunk in re.findall(r".{1,74}(?:\s|$)", re.sub(r"\s+", " ", q["answer"])):
            if chunk.strip():
                w("  " + chunk.strip())
        w("")

    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    print(f"  {len(APPENDIX_E)} queries, {len(lines)} lines, {os.path.getsize(OUT):,} bytes")
    for qid, title, reported in APPENDIX_E:
        q = scored.get(qid)
        if q:
            fr = sum(x["score"] for x in q["facts"])
            ua = sum(t["score"] for t in q["traps"])
            print(f"  {qid:5s} appendix: {reported:16s} | this run: FR {fr:.0f}/{len(q['facts'])}"
                  + (f", UAA {ua:.0f}/{len(q['traps'])}" if q["traps"] else ""))


if __name__ == "__main__":
    main()
