"""
analytics/da_analysis.py

All measurements required for the SIGDIAL 2026 paper, computed from SharedLog.

Sections:
  1. DA type distribution per agent
  2. DA type distribution per agent-agent pair
  3. DA type diff — simple vs complex (multi-intent) sessions
  4. DA type diff — terminal agents vs core agents
  5. Baseline #1 — null-hypothesis DA distribution vs observed
  6. Uncertainty flag lifecycle (preservation rate, mortality, type survival)
  7. Delegation trigger typing (frequency per pair, trigger → chain length)
  8. DA sequence entropy (bigram transition entropy per agent)

Usage:
    python analytics/da_analysis.py
    python analytics/da_analysis.py --db /path/to/conversation_log.db
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
import os
from collections import Counter, defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_DB = os.path.join(os.path.dirname(__file__), "..", "logs", "conversation_log.db")

TERMINAL_AGENTS = {"gateway", "synthesizer", "reflector"}
CORE_AGENTS     = {"sql_agent", "doc_agent", "pa_agent", "data_explorer"}
ALL_AGENTS      = TERMINAL_AGENTS | CORE_AGENTS

# Null-hypothesis: what DA types should each agent emit by design?
NULL_HYPOTHESIS: dict[str, list[str]] = {
    "gateway":       ["USER_QUERY", "REQUEST", "SYNTHESIZE", "TERMINATE"],
    "sql_agent":     ["INFORM", "REJECT"],
    "doc_agent":     ["INFORM", "REJECT"],
    "pa_agent":      ["INFORM", "CAVEAT", "REJECT"],
    "data_explorer": ["INFORM", "CAVEAT"],
    "synthesizer":   ["VALIDATE", "SYNTHESIZE", "CAVEAT"],
    "reflector":     ["CONFIRM", "CHALLENGE"],
}


def _agent(full_name: str) -> str:
    """Strip version suffix: 'sql_agent/2.0.0' → 'sql_agent'."""
    return full_name.split("/")[0]


# ── DB loader ─────────────────────────────────────────────────────────────────

def load_messages(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT session_id, turn, sender, recipient, da_type, flags, payload "
        "FROM messages ORDER BY session_id, turn"
    ).fetchall()
    conn.close()

    msgs = []
    for session_id, turn, sender, recipient, da_type, flags_json, payload_json in rows:
        payload = json.loads(payload_json) if payload_json else {}
        msgs.append({
            "session_id":         session_id,
            "turn":               turn,
            "sender":             _agent(sender),
            "sender_full":        sender,
            "recipient":          _agent(recipient),
            "da_type":            da_type,
            "flags":              json.loads(flags_json) if flags_json else [],
            "delegation_trigger": payload.get("delegation_trigger"),
            "metadata":           payload.get("metadata", {}),
        })
    return msgs


def group_by_session(msgs: list[dict]) -> dict[str, list[dict]]:
    sessions: dict[str, list[dict]] = defaultdict(list)
    for m in msgs:
        sessions[m["session_id"]].append(m)
    return dict(sessions)


# ── Session classifier ────────────────────────────────────────────────────────

def classify_session(session_msgs: list[dict]) -> str:
    """
    'simple'  — only one core agent fires, no delegation triggers
    'complex' — multiple core agents fire OR any delegation trigger present
    """
    triggers  = {m["delegation_trigger"] for m in session_msgs if m["delegation_trigger"]}
    core_used = {m["sender"] for m in session_msgs if m["sender"] in CORE_AGENTS}
    if triggers or len(core_used) > 1:
        return "complex"
    return "simple"


# ── Section 1: DA distribution per agent ─────────────────────────────────────

def section_da_per_agent(msgs: list[dict]) -> None:
    print("\n" + "="*80)
    print("  SECTION 1 — DA Type Distribution per Agent")
    print("="*80)

    counts: dict[str, Counter] = defaultdict(Counter)
    for m in msgs:
        counts[m["sender"]][m["da_type"]] += 1

    for agent in sorted(counts):
        total = sum(counts[agent].values())
        print(f"\n  {agent}  (n={total})")
        for da, c in counts[agent].most_common():
            pct = 100 * c / total
            bar = "█" * int(pct / 2)
            print(f"    {da:<15} {c:>5}  {pct:5.1f}%  {bar}")


# ── Section 2: DA distribution per agent-agent pair ──────────────────────────

def section_da_per_pair(msgs: list[dict]) -> None:
    print("\n" + "="*80)
    print("  SECTION 2 — DA Type Distribution per Agent-Agent Pair")
    print("="*80)

    pair_counts: dict[tuple, Counter] = defaultdict(Counter)
    for m in msgs:
        if m["sender"] == "user" or m["recipient"] == "user":
            continue
        pair = (m["sender"], m["recipient"])
        pair_counts[pair][m["da_type"]] += 1

    for (src, dst) in sorted(pair_counts):
        total = sum(pair_counts[(src, dst)].values())
        da_str = "  ".join(
            f"{da}={c}" for da, c in pair_counts[(src, dst)].most_common()
        )
        print(f"  {src:<18} → {dst:<18}  n={total:>4}  |  {da_str}")


# ── Section 3: DA diff simple vs complex ─────────────────────────────────────

def section_simple_vs_complex(msgs: list[dict], sessions: dict[str, list[dict]]) -> None:
    print("\n" + "="*80)
    print("  SECTION 3 — DA Type Diff: Simple vs Complex (Multi-Intent) Sessions")
    print("="*80)

    simple_counts: Counter = Counter()
    complex_counts: Counter = Counter()
    n_simple = n_complex = 0

    for sid, smgs in sessions.items():
        kind = classify_session(smgs)
        if kind == "simple":
            n_simple += 1
            for m in smgs:
                simple_counts[(m["sender"], m["da_type"])] += 1
        else:
            n_complex += 1
            for m in smgs:
                complex_counts[(m["sender"], m["da_type"])] += 1

    print(f"\n  Simple sessions  : {n_simple}")
    print(f"  Complex sessions : {n_complex}")

    # normalise to rates
    s_total = sum(simple_counts.values())  or 1
    c_total = sum(complex_counts.values()) or 1

    # find largest deltas
    all_keys = set(simple_counts) | set(complex_counts)
    deltas = []
    for key in all_keys:
        s_rate = simple_counts[key]  / s_total
        c_rate = complex_counts[key] / c_total
        deltas.append((key, s_rate, c_rate, c_rate - s_rate))

    deltas.sort(key=lambda x: abs(x[3]), reverse=True)

    print(f"\n  {'Agent+DA':<35} {'Simple%':>8}  {'Complex%':>9}  {'Δ':>8}")
    print("  " + "-"*65)
    for (agent, da), sr, cr, delta in deltas[:20]:
        marker = "▲" if delta > 0 else "▼"
        print(f"  {agent+'.'+da:<35} {100*sr:>7.2f}%  {100*cr:>8.2f}%  "
              f"{marker}{abs(100*delta):>6.2f}%")


# ── Section 4: Terminal vs Core agent DA diff ─────────────────────────────────

def section_terminal_vs_core(msgs: list[dict]) -> None:
    print("\n" + "="*80)
    print("  SECTION 4 — DA Type Diff: Terminal Agents vs Core Agents")
    print("="*80)

    terminal_counts: Counter = Counter()
    core_counts: Counter = Counter()

    for m in msgs:
        if m["sender"] in TERMINAL_AGENTS:
            terminal_counts[m["da_type"]] += 1
        elif m["sender"] in CORE_AGENTS:
            core_counts[m["da_type"]] += 1

    t_total = sum(terminal_counts.values()) or 1
    c_total = sum(core_counts.values())     or 1

    all_da = sorted(set(terminal_counts) | set(core_counts))

    print(f"\n  {'DA Type':<18} {'Terminal%':>10}  {'Core%':>8}  {'Δ':>8}")
    print("  " + "-"*50)
    for da in all_da:
        t_pct = 100 * terminal_counts[da] / t_total
        c_pct = 100 * core_counts[da]     / c_total
        delta = t_pct - c_pct
        marker = "▲T" if delta > 0 else "▲C"
        print(f"  {da:<18} {t_pct:>9.2f}%  {c_pct:>7.2f}%  {marker}{abs(delta):>5.2f}%")

    print(f"\n  Terminal total messages: {sum(terminal_counts.values())}")
    print(f"  Core total messages    : {sum(core_counts.values())}")


# ── Section 5: Baseline #1 — null-hypothesis DA distribution ─────────────────

def section_null_hypothesis(msgs: list[dict]) -> None:
    print("\n" + "="*80)
    print("  SECTION 5 — Baseline #1: Null-Hypothesis DA Distribution vs Observed")
    print("="*80)
    print("  Null hypothesis: each agent emits only its DESIGNED DA types, uniformly.")

    observed: dict[str, Counter] = defaultdict(Counter)
    for m in msgs:
        if m["sender"] in ALL_AGENTS:
            observed[m["sender"]][m["da_type"]] += 1

    for agent in sorted(NULL_HYPOTHESIS):
        expected_das  = NULL_HYPOTHESIS[agent]
        obs           = observed.get(agent, Counter())
        obs_total     = sum(obs.values()) or 1
        exp_uniform   = 100 / len(expected_das)

        print(f"\n  {agent}")
        print(f"  {'DA':<18} {'Expected%':>10}  {'Observed%':>10}  {'Δ':>8}  Note")
        print("  " + "-"*65)

        # expected DAs
        for da in expected_das:
            obs_pct = 100 * obs[da] / obs_total
            delta   = obs_pct - exp_uniform
            note    = "as designed" if abs(delta) < 5 else ("MORE than expected" if delta > 0 else "LESS than expected")
            print(f"  {da:<18} {exp_uniform:>9.1f}%  {obs_pct:>9.1f}%  "
                  f"{'▲' if delta>0 else '▼'}{abs(delta):>5.1f}%  {note}")

        # unexpected DAs (observed but not in null)
        unexpected = {da: c for da, c in obs.items() if da not in expected_das}
        for da, c in sorted(unexpected.items(), key=lambda x: -x[1]):
            obs_pct = 100 * c / obs_total
            print(f"  {da:<18} {'0.0':>10}%  {obs_pct:>9.1f}%  ▲{obs_pct:>5.1f}%  UNEXPECTED")


# ── Section 6: Uncertainty flag lifecycle ────────────────────────────────────

def section_flag_lifecycle(msgs: list[dict], sessions: dict[str, list[dict]]) -> None:
    print("\n" + "="*80)
    print("  SECTION 6 — Uncertainty Flag Lifecycle")
    print("="*80)

    introduced_total = 0
    survived_total   = 0
    mortality: Counter  = Counter()   # agent where flag last appears
    type_introduced: Counter = Counter()
    type_survived:   Counter = Counter()

    for sid, smgs in sessions.items():
        # find all flags introduced (first appearance per flag type)
        session_flags_introduced: list[tuple[str, str]] = []  # (flag, introducer)
        for m in smgs:
            for flag in m["flags"]:
                session_flags_introduced.append((flag, m["sender"]))

        if not session_flags_introduced:
            continue

        # find flags in final SYNTHESIZE/TERMINATE
        final_flags: set[str] = set()
        for m in smgs:
            if m["da_type"] in ("SYNTHESIZE", "TERMINATE") and m["recipient"] == "user":
                for f in m["flags"]:
                    final_flags.add(f)

        # find mortality: last turn where each flag appears
        for flag, introducer in session_flags_introduced:
            introduced_total   += 1
            type_introduced[flag] += 1
            if flag in final_flags:
                survived_total   += 1
                type_survived[flag] += 1
            else:
                # find last agent that carried it
                last_carrier = introducer
                for m in smgs:
                    if flag in m["flags"]:
                        last_carrier = m["sender"]
                mortality[last_carrier] += 1

    preservation_rate = 100 * survived_total / introduced_total if introduced_total else 0

    print(f"\n  Flags introduced : {introduced_total}")
    print(f"  Flags survived   : {survived_total}  ({preservation_rate:.1f}% preservation rate)")
    print(f"  Flags killed     : {introduced_total - survived_total}")

    print(f"\n  Mortality points (where flags die):")
    for agent, count in mortality.most_common():
        print(f"    {agent:<25} kills {count} flag(s)")

    print(f"\n  Flag type survival differential:")
    all_flag_types = set(type_introduced) | set(type_survived)
    print(f"  {'Flag type':<25} {'Introduced':>12}  {'Survived':>9}  {'Survival%':>10}")
    print("  " + "-"*60)
    for ft in sorted(all_flag_types):
        intro  = type_introduced[ft]
        surv   = type_survived[ft]
        rate   = 100 * surv / intro if intro else 0
        print(f"  {ft:<25} {intro:>12}  {surv:>9}  {rate:>9.1f}%")


# ── Section 7: Delegation trigger typing ─────────────────────────────────────

def section_trigger_typing(msgs: list[dict], sessions: dict[str, list[dict]]) -> None:
    print("\n" + "="*80)
    print("  SECTION 7 — Delegation Trigger Typing")
    print("="*80)

    # trigger frequency per agent-agent pair
    pair_trigger: dict[tuple, Counter] = defaultdict(Counter)
    for m in msgs:
        if m["delegation_trigger"]:
            pair_trigger[(m["sender"], m["recipient"])][m["delegation_trigger"]] += 1

    print(f"\n  Trigger frequency per agent-agent pair:")
    print(f"  {'Pair':<40} {'Trigger':<25} Count")
    print("  " + "-"*70)
    for (src, dst) in sorted(pair_trigger):
        for trig, count in pair_trigger[(src, dst)].most_common():
            print(f"  {src+' → '+dst:<40} {trig:<25} {count}")

    # overall trigger frequency
    trigger_total: Counter = Counter()
    for m in msgs:
        if m["delegation_trigger"]:
            trigger_total[m["delegation_trigger"]] += 1

    print(f"\n  Overall trigger distribution:")
    total_triggers = sum(trigger_total.values())
    for trig, count in trigger_total.most_common():
        pct = 100 * count / total_triggers
        print(f"    {trig:<30} {count:>5}  ({pct:.1f}%)")

    # does trigger predict chain length?
    # chain length = number of messages in the session after the trigger fires
    trigger_chain_lengths: dict[str, list[int]] = defaultdict(list)
    for sid, smgs in sessions.items():
        for i, m in enumerate(smgs):
            if m["delegation_trigger"]:
                remaining = len(smgs) - i
                trigger_chain_lengths[m["delegation_trigger"]].append(remaining)

    print(f"\n  Trigger → delegation chain length (remaining turns after trigger):")
    print(f"  {'Trigger':<30} {'n':>5}  {'mean':>7}  {'min':>5}  {'max':>5}")
    print("  " + "-"*55)
    for trig in sorted(trigger_chain_lengths):
        lengths = trigger_chain_lengths[trig]
        mean = sum(lengths) / len(lengths)
        print(f"  {trig:<30} {len(lengths):>5}  {mean:>7.2f}  {min(lengths):>5}  {max(lengths):>5}")


# ── Section 8: DA sequence entropy ───────────────────────────────────────────

def section_da_entropy(msgs: list[dict], sessions: dict[str, list[dict]]) -> None:
    print("\n" + "="*80)
    print("  SECTION 8 — DA Sequence Entropy (Bigram Transition Entropy per Agent)")
    print("="*80)
    print("  Measures: how predictable is each agent's next DA given its current DA?")
    print("  Low entropy = highly predictable role. High entropy = flexible/adaptive.")

    # build per-agent DA sequences across all sessions
    agent_sequences: dict[str, list[str]] = defaultdict(list)
    for sid, smgs in sessions.items():
        # per session, build per-agent ordered DA sequence
        for agent in ALL_AGENTS | {"user"}:
            seq = [m["da_type"] for m in smgs if m["sender"] == agent]
            agent_sequences[agent].extend(seq)

    print(f"\n  {'Agent':<20} {'n_msgs':>7}  {'n_states':>9}  {'H_bigram':>10}  Profile")
    print("  " + "-"*70)

    agent_entropy = {}
    for agent in sorted(agent_sequences):
        seq = agent_sequences[agent]
        if len(seq) < 2:
            continue

        # bigram counts
        bigrams: Counter = Counter()
        unigrams: Counter = Counter()
        for i in range(len(seq) - 1):
            bigrams[(seq[i], seq[i+1])] += 1
            unigrams[seq[i]] += 1

        # conditional entropy H(next | current)
        H = 0.0
        for (a, b), count in bigrams.items():
            p_ab = count / unigrams[a]
            p_a  = unigrams[a] / sum(unigrams.values())
            H   -= p_a * p_ab * math.log2(p_ab)

        n_states = len(set(seq))
        agent_entropy[agent] = H
        profile = _entropy_profile(agent, H)
        print(f"  {agent:<20} {len(seq):>7}  {n_states:>9}  {H:>10.4f}  {profile}")

    # bigram transition matrix for top agents
    print(f"\n  Bigram transition probabilities (most active agents):")
    for agent in ["synthesizer", "reflector", "pa_agent", "sql_agent", "gateway"]:
        seq = agent_sequences.get(agent, [])
        if len(seq) < 2:
            continue
        bigrams: Counter = Counter()
        unigrams: Counter = Counter()
        for i in range(len(seq) - 1):
            bigrams[(seq[i], seq[i+1])] += 1
            unigrams[seq[i]] += 1

        print(f"\n  {agent}:")
        for (a, b) in sorted(bigrams, key=lambda k: -bigrams[k]):
            p = bigrams[(a, b)] / unigrams[a]
            print(f"    {a:<15} → {b:<15}  P={p:.3f}  n={bigrams[(a,b)]}")


def _entropy_profile(agent: str, H: float) -> str:
    if agent in ("reflector",):
        return "narrow role (VALIDATE in → CONFIRM/CHALLENGE out)"
    if agent in ("data_explorer",):
        return "narrow role (REQUEST in → INFORM/CAVEAT out)"
    if H < 0.3:
        return "very predictable — strict role adherence"
    if H < 0.7:
        return "predictable — mostly follows designed role"
    if H < 1.2:
        return "moderate flexibility — adapts to context"
    return "high flexibility — rich communicative repertoire"


# ── Main ──────────────────────────────────────────────────────────────────────

def run(db_path: str | None = None) -> None:
    db = db_path or DEFAULT_DB
    print(f"\nLoading messages from: {db}")
    msgs     = load_messages(db)
    sessions = group_by_session(msgs)
    print(f"  {len(msgs)} messages across {len(sessions)} sessions")

    section_da_per_agent(msgs)
    section_da_per_pair(msgs)
    section_simple_vs_complex(msgs, sessions)
    section_terminal_vs_core(msgs)
    section_null_hypothesis(msgs)
    section_flag_lifecycle(msgs, sessions)
    section_trigger_typing(msgs, sessions)
    section_da_entropy(msgs, sessions)

    print("\n" + "="*80)
    print("  ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    db_path = None
    if len(sys.argv) > 1:
        if sys.argv[1] == "--db" and len(sys.argv) > 2:
            db_path = sys.argv[2]
        else:
            db_path = sys.argv[1]
    run(db_path)
