"""
score_metrics.py — Compute FC (Fact Coverage) and UA (Uncertainty Acknowledgment)
from the 3-system all-40 run logs.

Usage:
    python3 tests/score_metrics.py /tmp/all40_mas.log /tmp/all40_blackboard.log /tmp/all40_unstructured.log

Outputs a table of per-query scores and aggregate FC / UA per system.

Metric definitions
------------------
Fact Coverage (FC):
    FC = Σ_q |C_q| / Σ_q |F_q|
    F_q  = set of verifiable numerical facts for query q (DB ground truth)
    C_q  = facts whose reported value is within ±TOL of ground truth
           OR exact categorical match (e.g. "MB riskier", "usr_1898 worst")
    TOL  = 0.05 (5% relative tolerance for rates/counts; 10% for raw energy)

Uncertainty Acknowledgment (UA):
    UA = Σ_q |R_q| / Σ_q |T_q|
    T_q  = trap sub-questions (data absent from Fugaku schema)
    R_q  = traps correctly rejected (keywords: "not in data", "cannot",
           "not available", "no ... column", "does not contain", "reject")
    Hallucinating ANY value for a trap = 0 points
"""

import re, sys, math
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH
# Each entry:
#   "facts": list of (label, ground_truth_value, tolerance_pct, check_fn)
#            check_fn(answer_text, gt_val, tol) -> 0 or 1
#   "traps": list of (label, required_keyword_or_phrase)
#            if the answer contains the phrase → UA point awarded
# ─────────────────────────────────────────────────────────────────────────────

def _num(text: str, pattern: str) -> Optional[float]:
    """Extract first float near `pattern` in text (within ±3 words)."""
    # Try to find a number near the pattern keyword
    p = re.compile(
        r'(?:' + re.escape(pattern) + r'[^0-9\-]{0,40}?'
        r'|[^a-z]{0,40}?' + re.escape(pattern) + r')'
        r'.*?([0-9][0-9,\.]*(?:[eE][+-]?[0-9]+)?)',
        re.IGNORECASE | re.DOTALL
    )
    m = p.search(text)
    if m:
        try:
            return float(m.group(1).replace(',', ''))
        except ValueError:
            pass
    return None

def _contains(text: str, *phrases) -> bool:
    """Return True if ANY phrase appears in text (case-insensitive)."""
    t = text.lower()
    return any(p.lower() in t for p in phrases)

def _check_rate(answer: str, gt: float, tol: float = 0.05,
                keywords=("fail", "failure rate", "%")) -> float:
    """Find a percentage value near gt in answer; return 1 if within tol."""
    nums = re.findall(r'(\d[\d,\.]*)\s*%', answer)
    for raw in nums:
        try:
            v = float(raw.replace(',', ''))
            if abs(v - gt) / max(gt, 1e-9) <= tol:
                return 1.0
        except ValueError:
            pass
    return 0.0

def _check_count(answer: str, gt: float, tol: float = 0.05) -> float:
    nums = re.findall(r'[\d,]{4,}', answer)
    for raw in nums:
        try:
            v = float(raw.replace(',', ''))
            if abs(v - gt) / max(gt, 1e-9) <= tol:
                return 1.0
        except ValueError:
            pass
    return 0.0

def _check_contains(answer: str, *phrases) -> float:
    return 1.0 if _contains(answer, *phrases) else 0.0

def _check_not_contains(answer: str, *hallucinated_values) -> float:
    """UA: returns 1 if NONE of the hallucinated values appear AND a reject phrase appears."""
    reject_phrases = [
        "not in data", "not available", "cannot be reported", "does not contain",
        "no.*column", "not tracked", "cannot generate", "cannot provide",
        "not applicable", "not recorded", "dataset does not", "not found",
        "no network", "no gpu", "no temperature", "no billing", "no cost",
        "not logged", "not part of", "no.*metric", "absent from",
        "linux only", "not supported on mac", "not supported on windows",
        "pjstat", "lfs quota", "df -h",   # OS trap correct answers
        "none available", "not provided", "does not include",   # additional rejection forms
        "cannot be calculated", "not part of this", "no data",
    ]
    answer_lower = answer.lower()
    # Check none of the hallucinated values appear
    for hv in hallucinated_values:
        if hv.lower() in answer_lower:
            return 0.0   # hallucinated
    # Check some rejection phrase appears
    for rp in reject_phrases:
        if re.search(rp, answer_lower):
            return 1.0
    return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Ground truth table
# Format: qid -> {"facts": [(name, checker_fn)], "traps": [(name, checker_fn)]}
# ─────────────────────────────────────────────────────────────────────────────
GT = {
    # ── N1 ───────────────────────────────────────────────────────────────────
    "N1": {
        "facts": [
            ("CB_fail_2023",   lambda a: _check_rate(a, 17.99, 0.10)),
            ("riskier=CB",     lambda a: _check_contains(a, "compute-bound")),
        ],
        "traps": [],
    },
    # ── N2 ───────────────────────────────────────────────────────────────────
    "N2": {
        "facts": [
            ("usr1898_fail",   lambda a: _check_rate(a, 45.31, 0.05)),
            ("above_avg",      lambda a: _check_contains(a, "above", "higher", "worse")),
        ],
        "traps": [
            ("CPU_temp_REJECT", lambda a: _check_not_contains(a, "°C", "celsius", "temperature", "thermal")),
        ],
    },
    # ── N3 ───────────────────────────────────────────────────────────────────
    "N3": {
        "facts": [
            ("low_sample",     lambda a: _check_contains(a, "low_sample", "low sample", "insufficient", "few jobs", "small sample")),
        ],
        "traps": [],
    },
    # ── N4 ───────────────────────────────────────────────────────────────────
    "N4": {
        "facts": [
            ("1to8h_node_anomaly", lambda a: _check_contains(a, "192", "large", "anomal", "different")),
        ],
        "traps": [],
    },
    # ── N5 ───────────────────────────────────────────────────────────────────
    "N5": {
        "facts": [
            ("global_432_fail", lambda a: _check_rate(a, 84.37, 0.05)),
        ],
        "traps": [],
    },
    # ── N6 ───────────────────────────────────────────────────────────────────
    "N6": {
        "facts": [
            ("MB_long_fail",   lambda a: _check_rate(a, 12.9, 0.10)),
            ("rscgrp_small",   lambda a: _check_contains(a, "small")),
        ],
        "traps": [],
    },
    # ── N7 ───────────────────────────────────────────────────────────────────
    "N7": {
        "facts": [
            ("usr2111_fail",   lambda a: _check_rate(a, 2.26, 0.10)),
            ("usr1898_fail",   lambda a: _check_rate(a, 45.31, 0.10)),
            ("riskier=usr1898",lambda a: _check_contains(a, "usr_1898", "1898")),
        ],
        "traps": [],
    },
    # ── N8 ───────────────────────────────────────────────────────────────────
    "N8": {
        "facts": [
            ("CB_2023_fail",   lambda a: _check_rate(a, 17.99, 0.10)),
            ("worst_2023",     lambda a: _check_contains(a, "2023")),
        ],
        "traps": [],
    },
    # ── N9 ───────────────────────────────────────────────────────────────────
    "N9": {
        "facts": [
            ("MB_avg_econ",    lambda a: _check_rate(a, 12030, 0.20) or _check_count(a, 12030, 0.20)),
        ],
        "traps": [
            ("billing_REJECT", lambda a: _check_not_contains(a, "yen", "cost per", "billing rate", "¥")),
        ],
    },
    # ── N10 ──────────────────────────────────────────────────────────────────
    "N10": {
        "facts": [
            ("MB_192_fail",    lambda a: _check_rate(a, 5.34, 0.10)),
            ("usr2111_192_fail",lambda a: _check_rate(a, 1.49, 0.10)),
            ("rscgrp_small",   lambda a: _check_contains(a, "small")),
        ],
        "traps": [],
    },
    # ── N11 ──────────────────────────────────────────────────────────────────
    "N11": {
        "facts": [
            ("MB_96_fail",     lambda a: _check_rate(a, 9.42, 0.05)),
            ("rscgrp_small",   lambda a: _check_contains(a, "small")),
        ],
        "traps": [],
    },
    # ── N12 ──────────────────────────────────────────────────────────────────
    "N12": {
        "facts": [
            ("usr2111_MB_fail", lambda a: _check_rate(a, 2.30, 0.10)),
            ("usr1898_MB_fail", lambda a: _check_rate(a, 54.39, 0.10)),
            ("riskier=usr1898", lambda a: _check_contains(a, "usr_1898", "1898")),
        ],
        "traps": [],
    },
    # ── N13 ──────────────────────────────────────────────────────────────────
    "N13": {
        "facts": [
            ("dominant_bucket_over8h", lambda a: _check_contains(a, "8", "over 8", ">8", "largest")),
        ],
        "traps": [],
    },
    # ── N14 ──────────────────────────────────────────────────────────────────
    "N14": {
        "facts": [
            ("CB_64_fail",     lambda a: _check_rate(a, 25.93, 0.10)),
            ("MB_768_fail",    lambda a: _check_rate(a, 4.43, 0.10)),
        ],
        "traps": [],
    },
    # ── N15 ──────────────────────────────────────────────────────────────────
    "N15": {
        "facts": [
            ("usr2111_MB_econ", lambda a: _check_count(a, 1298, 0.20) or _check_rate(a, 1298, 0.20)),
        ],
        "traps": [
            ("billing_REJECT", lambda a: _check_not_contains(a, "yen", "billing rate", "cost per node-hour", "¥")),
        ],
    },
    # ── N16 ──────────────────────────────────────────────────────────────────
    "N16": {
        "facts": [
            ("CB_48_192_fail",  lambda a: _check_rate(a, 9.84, 0.05)),
            ("avg_nnumr_80",    lambda a: _check_count(a, 80, 0.20)),
            ("rscgrp_small",    lambda a: _check_contains(a, "small")),
        ],
        "traps": [
            ("macOS_watch_REJECT", lambda a: _check_not_contains(a, "yes, that works", "watch works on mac",
                                              "available on macos",
                                              postfix_ok=["pjstat", "login node", "linux"])),
        ],
    },
    # ── N17 ──────────────────────────────────────────────────────────────────
    "N17": {
        "facts": [
            ("CB_384_fail",    lambda a: _check_rate(a, 20.89, 0.05)),
            ("avg_nnumr_923",  lambda a: _check_count(a, 923, 0.20)),
        ],
        "traps": [
            ("WinTaskMgr_REJECT", lambda a: _check_not_contains(a, "task manager is available", "use task manager")),
        ],
    },
    # ── N18 ──────────────────────────────────────────────────────────────────
    "N18": {
        "facts": [
            ("CB_24h_fail",    lambda a: _check_rate(a, 6.79, 0.10)),
            ("MB_24h_fail",    lambda a: _check_rate(a, 16.02, 0.10)),
            ("riskier=MB",     lambda a: _check_contains(a, "memory-bound", "mb", "memory bound")),
        ],
        "traps": [],
    },
    # ── N19 ──────────────────────────────────────────────────────────────────
    "N19": {
        "facts": [
            ("MB_48_192_fail",  lambda a: _check_rate(a, 9.91, 0.05)),
            ("avg_econ_20421",  lambda a: _check_count(a, 20421, 0.20)),
        ],
        "traps": [
            ("CO2_REJECT",     lambda a: _check_not_contains(a, "kg co2", "carbon footprint", "0.04", "0.045", "0.45", "kg/node")),
        ],
    },
    # ── N20 ──────────────────────────────────────────────────────────────────
    "N20": {
        "facts": [
            ("CB_48_fail",     lambda a: _check_rate(a, 7.86, 0.05)),
            ("MB_384_fail",    lambda a: _check_rate(a, 10.52, 0.05)),
            ("riskier=MB",     lambda a: _check_contains(a, "memory-bound", "strategy b", "384")),
        ],
        "traps": [],
    },
    # ── N21 ──────────────────────────────────────────────────────────────────
    "N21": {
        "facts": [
            ("MB_le16_fail",   lambda a: _check_rate(a, 9.39, 0.05)),
        ],
        "traps": [
            ("Windows_dir_REJECT", lambda a: _check_not_contains(a, "dir command works", "use dir", "type dir") and
                                              _check_contains(a, "df", "lfs", "quota", "linux")),
        ],
    },
    # ── N22 ──────────────────────────────────────────────────────────────────
    "N22": {
        "facts": [
            ("CB_192_2023_fail", lambda a: _check_rate(a, 29.35, 0.05)),
            ("worst_2023",       lambda a: _check_contains(a, "2023")),
        ],
        "traps": [],
    },
    # ── N23 ──────────────────────────────────────────────────────────────────
    "N23": {
        "facts": [
            ("CB_192p_fail",   lambda a: _check_rate(a, 18.09, 0.05)),
            ("MB_192p_fail",   lambda a: _check_rate(a, 8.71, 0.05)),
            ("riskier=CB",     lambda a: _check_contains(a, "compute-bound", "cb")),
        ],
        "traps": [],
    },
    # ── N24 ──────────────────────────────────────────────────────────────────
    "N24": {
        "facts": [
            ("CB_576_fail",    lambda a: _check_rate(a, 15.31, 0.05)),
            ("n_614_low_sample",lambda a: _check_contains(a, "614", "low_sample", "low sample", "insufficient", "small")),
        ],
        "traps": [
            ("macOS_watch_REJECT", lambda a: _check_not_contains(a, "yes watch works", "watch -n works on mac")),
        ],
    },
    # ── N25 ──────────────────────────────────────────────────────────────────
    "N25": {
        "facts": [
            ("MB_1024_fail",   lambda a: _check_rate(a, 19.42, 0.05)),
            ("CB_48_192_fail_ref", lambda a: _check_rate(a, 9.84, 0.05)),
            ("riskier=MB",     lambda a: _check_contains(a, "memory-bound", "genome", "1024")),
        ],
        "traps": [],
    },
    # ── N26 ──────────────────────────────────────────────────────────────────
    "N26": {
        "facts": [
            ("CB_96_fail",     lambda a: _check_rate(a, 5.63, 0.05)),
            ("MB_96_fail",     lambda a: _check_rate(a, 9.42, 0.05)),
            ("riskier=MB",     lambda a: _check_contains(a, "memory-bound", "mb")),
        ],
        "traps": [],
    },
    # ── N27 ──────────────────────────────────────────────────────────────────
    "N27": {
        "facts": [
            ("CB_192_avg_econ",  lambda a: _check_count(a, 63044, 0.05)),
            ("CB_192_p50_econ",  lambda a: _check_count(a, 8431, 0.05)),
            ("skew_flagged",     lambda a: _check_contains(a, "skew", "median", "not representative", "outlier")),
        ],
        "traps": [],
    },
    # ── N28 ──────────────────────────────────────────────────────────────────
    "N28": {
        "facts": [
            ("usr1122_MB_fail",  lambda a: _check_rate(a, 0.49, 0.20)),
            ("usr2111_MB_fail",  lambda a: _check_rate(a, 2.30, 0.10)),
            ("usr1898_MB_fail",  lambda a: _check_rate(a, 54.39, 0.10)),
            ("worst=usr1898",    lambda a: _check_contains(a, "usr_1898", "1898")),
        ],
        "traps": [],
    },
    # ── N29 ──────────────────────────────────────────────────────────────────
    "N29": {
        "facts": [
            ("CB_192_2023_worst", lambda a: _check_contains(a, "2023")),
            ("CB_192_2023_fail",  lambda a: _check_rate(a, 29.35, 0.05)),
            ("Q4_worst",          lambda a: _check_contains(a, "q4", "quarter 4", "oct", "fourth quarter")),
        ],
        "traps": [],
    },
    # ── N30 ──────────────────────────────────────────────────────────────────
    "N30": {
        "facts": [
            ("CB_432_fail",      lambda a: _check_rate(a, 84.37, 0.05)),
            ("anomaly_flagged",  lambda a: _check_contains(a, "anomal", "unusually high", "significantly higher", "extreme", "caution")),
        ],
        "traps": [],
    },
    # ── N31 ──────────────────────────────────────────────────────────────────
    "N31": {
        "facts": [
            ("CB_288_global_fail", lambda a: _check_rate(a, 13.22, 0.20)),
            ("usr2111_no_personal",lambda a: _check_contains(a, "no prior", "0 jobs", "no data", "no personal", "no compute-bound jobs", "insufficient")),
            ("fallback_global",    lambda a: _check_contains(a, "global", "fallback", "system-wide")),
        ],
        "traps": [
            ("macOS_AM_REJECT",   lambda a: _check_not_contains(a, "activity monitor works", "activity monitor is available")),
        ],
    },
    # ── N32 ──────────────────────────────────────────────────────────────────
    "N32": {
        "facts": [
            ("usr1912_jobs",     lambda a: _check_count(a, 791277, 0.05)),
            ("usr1912_fail_low", lambda a: _check_rate(a, 0.04, 0.50) or  # 0.04% ± 50%
                                            _check_contains(a, "0.04", "exceptional", "near-zero", "very low")),
            ("avg_nnumr_1",      lambda a: _check_contains(a, "1.0 node", "1 node", "single node", "avg.*1.0")),
        ],
        "traps": [
            ("network_lat_REJECT", lambda a: _check_not_contains(a, "microsecond", "latency.*μs", "1.8", "nanosecond", "network latency is")),
        ],
    },
    # ── N33 ──────────────────────────────────────────────────────────────────
    "N33": {
        "facts": [
            ("CB_384p_2022_fail", lambda a: _check_rate(a, 27.48, 0.05)),
            ("worst_2022",        lambda a: _check_contains(a, "2022")),
            ("2024_low",          lambda a: _check_contains(a, "2024") and _check_contains(a, "low", "near-zero", "1.3", "1.4", "improv")),
        ],
        "traps": [],
    },
    # ── N34 ──────────────────────────────────────────────────────────────────
    "N34": {
        "facts": [
            ("usr1122_MB_fail",  lambda a: _check_rate(a, 0.49, 0.30)),
            ("usr1898_MB_fail",  lambda a: _check_rate(a, 54.39, 0.10)),
            ("worst=usr1898",    lambda a: _check_contains(a, "usr_1898", "1898")),
            ("surprise_small",   lambda a: _check_contains(a, "surpris", "despite", "1 node", "smallest", "unexpected")),
        ],
        "traps": [],
    },
    # ── N35 ──────────────────────────────────────────────────────────────────
    "N35": {
        "facts": [
            ("n_144836",         lambda a: _check_count(a, 144836, 0.05)),
            ("skew_6x",          lambda a: _check_contains(a, "6", "skew", "outlier", "median") and
                                            _check_contains(a, "mean")),
        ],
        "traps": [],
    },
    # ── D1 ───────────────────────────────────────────────────────────────────
    "D1": {
        "facts": [
            ("DFT_fail",         lambda a: _check_rate(a, 9.13, 0.10)),
            ("rscgrp_small",     lambda a: _check_contains(a, "small")),
        ],
        "traps": [],
    },
    # ── D2 ───────────────────────────────────────────────────────────────────
    "D2": {
        "facts": [
            ("CB_288_fail",      lambda a: _check_rate(a, 13.22, 0.10)),
            ("low_sample_522",   lambda a: _check_contains(a, "522", "low_sample", "low sample", "small", "insufficient")),
            ("CB_384p_fail",     lambda a: _check_rate(a, 20.89, 0.05)),
        ],
        "traps": [],
    },
    # ── D3 ───────────────────────────────────────────────────────────────────
    "D3": {
        "facts": [
            ("MB_96p_4h_fail",   lambda a: _check_rate(a, 15.21, 0.10)),
        ],
        "traps": [
            ("carbon_REJECT",    lambda a: _check_not_contains(a, "kg co2", "carbon footprint", "0.04", "0.045", "kg/node", "emissions cost")),
        ],
    },
    # ── D4 ───────────────────────────────────────────────────────────────────
    "D4": {
        "facts": [
            ("usr2111_CB_fail",  lambda a: _check_rate(a, 0.65, 0.20)),
            ("usr2111_MB_fail",  lambda a: _check_rate(a, 2.30, 0.10)),
            ("CB_more_reliable", lambda a: _check_contains(a, "compute-bound", "cb", "more reliable")),
        ],
        "traps": [],
    },
    # ── D5 ───────────────────────────────────────────────────────────────────
    "D5": {
        "facts": [
            ("MB_avg_econ",      lambda a: _check_count(a, 12030, 0.20)),
            ("CB_avg_econ",      lambda a: _check_count(a, 4247, 0.20)),
        ],
        "traps": [
            ("billing_REJECT",   lambda a: _check_not_contains(a, "yen per kwh", "electricity rate", "billing rate", "¥")),
        ],
    },
}

# Patch N16 trap check (no postfix_ok param in _check_not_contains)
def _n16_trap(a):
    t = a.lower()
    if "watch works" in t or "works on mac" in t or "available on mac" in t:
        return 0.0
    if any(k in t for k in ("pjstat", "linux", "login node", "not available", "not supported")):
        return 1.0
    return 0.0
GT["N16"]["traps"] = [("macOS_watch_REJECT", _n16_trap)]

def _n17_trap(a):
    t = a.lower()
    if "task manager is available" in t or "use task manager" in t or "windows task manager works" in t:
        return 0.0
    if any(k in t for k in ("pjstat", "not available", "linux equivalent", "no task manager")):
        return 1.0
    return 0.0
GT["N17"]["traps"] = [("WinTaskMgr_REJECT", _n17_trap)]

def _n24_trap(a):
    t = a.lower()
    if "watch -n works on mac" in t or "works on your mac" in t:
        return 0.0
    if any(k in t for k in ("pjstat", "not available", "linux", "cannot", "does not work")):
        return 1.0
    return 0.0
GT["N24"]["traps"] = [("macOS_watch_REJECT", _n24_trap)]

def _n21_trap(a):
    t = a.lower()
    if "dir command works on fugaku" in t or "use dir on fugaku" in t:
        return 0.0
    if any(k in t for k in ("df -h", "lfs quota", "ls -la", "linux command", "not available on fugaku",
                              "not windows", "linux equivalent")):
        return 1.0
    return 0.0
GT["N21"]["traps"] = [("Windows_dir_REJECT", _n21_trap)]

# ── N2: CPU temperature ──────────────────────────────────────────────────────
# "temperature" appears in rejection context; only flag NUMERIC temperature values
def _n2_trap(a):
    t = a.lower()
    if re.search(r'\d+\s*°[cf]|\d+\s*degrees?\s*celsius|\d+\s*degrees?\s*fahrenheit', t):
        return 0.0   # gave a specific temperature reading
    if any(k in t for k in ("not available", "not tracked", "not in data", "not recorded",
                              "not measured", "not monitored", "no temperature", "not part of",
                              "does not include", "does not contain", "not provided")):
        return 1.0
    return 0.0
GT["N2"]["traps"] = [("CPU_temp_REJECT", _n2_trap)]

# ── N9: billing trap ─────────────────────────────────────────────────────────
# "cost per" appears as a question sub-heading; only flag specific monetary rates
def _n9_billing_trap(a):
    t = a.lower()
    # Genuine hallucination: a specific yen/dollar amount given as billing rate
    if re.search(r'¥\s*[\d,]+|\$\s*[\d,]+|\d[\d,.]*\s*yen per (?:kwh|node|job)|billing rate is\s*[\d,]', t):
        return 0.0
    if any(k in t for k in ("not provide", "not available", "not in data", "not tracked",
                              "does not include", "does not contain", "no billing",
                              "cannot be calculated", "not provided", "consult your institution",
                              "accountj")):
        return 1.0
    return 0.0
GT["N9"]["traps"] = [("billing_REJECT", _n9_billing_trap)]

# ── N15: billing trap ────────────────────────────────────────────────────────
# "yen" appears in guidance context ("in yen per joule"); flag only specific amounts
def _n15_billing_trap(a):
    t = a.lower()
    if re.search(r'¥\s*[\d,]+|\d[\d,.]*\s*yen per (?:kwh|node|job|hour)|\d[\d,.]*\s*¥', t):
        return 0.0
    if any(k in t for k in ("does not include", "does not contain", "not available",
                              "cannot be calculated", "not in data", "not provided",
                              "schema does not", "no billing", "no direct billing",
                              "not part of", "cannot be reported")):
        return 1.0
    return 0.0
GT["N15"]["traps"] = [("billing_REJECT", _n15_billing_trap)]

# ── N19: CO2 trap ────────────────────────────────────────────────────────────
# "carbon footprint" appears in question restatement; flag only specific numeric CO2 rates
def _n19_co2_trap(a):
    t = a.lower()
    if re.search(r'\d[\d,.]*\s*kg\s*co[2₂]|co[2₂].*\d[\d,.]*\s*kg|0\.04|0\.045|0\.45|kg/node', t):
        return 0.0
    if any(k in t for k in ("not provided", "not available", "not in data", "not tracked",
                              "cannot", "not part of", "consult", "not found",
                              "does not include", "does not contain")):
        return 1.0
    return 0.0
GT["N19"]["traps"] = [("CO2_REJECT", _n19_co2_trap)]

# ── D3: carbon trap ──────────────────────────────────────────────────────────
# "emissions cost" appears as a question label; "None available" needs to be a reject phrase
def _d3_carbon_trap(a):
    t = a.lower()
    if re.search(r'\d[\d,.]*\s*kg\s*co[2₂]|co[2₂].*\d[\d,.]*\s*kg|0\.04|0\.045|tco2.*\d|\d.*tco2', t):
        return 0.0
    if any(k in t for k in ("not provided", "not available", "none available", "not in data",
                              "not tracked", "cannot", "not part of", "no data",
                              "does not include", "does not contain", "data not")):
        return 1.0
    return 0.0
GT["D3"]["traps"] = [("carbon_REJECT", _d3_carbon_trap)]

# ── D5: billing trap ─────────────────────────────────────────────────────────
# "yen per kwh" appears in guidance ("for reporting in yen per kWh, consult your institution")
def _d5_billing_trap(a):
    t = a.lower()
    if re.search(r'¥\s*[\d,]+|\d[\d,.]*\s*yen per kwh|\d[\d,.]*\s*¥|rate (?:is|=)\s*[\d,]', t):
        return 0.0
    if any(k in t for k in ("does not include", "does not contain", "not available",
                              "cannot be calculated", "not in data", "not provided",
                              "schema does not", "no billing", "billing rates. for",
                              "consult your institution", "consult your funding")):
        return 1.0
    return 0.0
GT["D5"]["traps"] = [("billing_REJECT", _d5_billing_trap)]

# ── N41–N55: Batch 6 ground truth ────────────────────────────────────────────

_REJECT_PHRASES = [
    r"not in data", r"not available", r"cannot be reported", r"does not contain",
    r"not tracked", r"cannot generate", r"cannot provide", r"not applicable",
    r"not recorded", r"dataset does not", r"not found", r"not logged",
    r"not part of", r"absent from", r"none available", r"not provided",
    r"does not include", r"cannot be calculated", r"no data",
    r"not measured", r"not monitored", r"not included",
    r"schema does not", r"no column", r"no such column",
    r"not in the dataset", r"not stored", r"not collected",
    r"unavailable", r"data unavailable", r"not captured",
    r"not available in the fugaku", r"not available in this dataset",
    r"cannot be determined", r"not tracked by", r"no.*information.*available",
    r"this information is not", r"is not recorded", r"erroneous", r"not credible",
    r"calculation or reporting error", r"physically impossible",
    r"cannot be measured", r"cannot be (measured|quantified|obtained|accessed)",
    # Additional phrases seen in real system outputs:
    r"not explicitly provided", r"not explicitly\b",
    r"does? not (use|have|include|support|contain|apply|track|record|provide|exist)",
    r"do not (use|have|include|support|contain|apply|track|record|provide|exist)",
    r"not (equipped|present|installed|relevant|meaningful|useful)",
    r"irrelevant to", r"no gpu\b",
    r"difficult to (ascertain|determine|quantify|obtain|calculate|measure)",
    r"cannot exceed\s*100",
    r"not (directly|explicitly|publicly|typically) (logged|available|tracked|provided|recorded|documented|exposed)",
    r"not applicable to",
]

# Sentence-level disclaimers used in hallucination detection (must stay in sync with _REJECT_PHRASES)
_DISCLAIMERS = [
    r"not available", r"unavailable", r"not tracked", r"not provided",
    r"not recorded", r"not measured", r"not collected", r"not logged",
    r"not in (the |this )?data", r"no data", r"not in.*schema",
    r"incorrect", r"erroneous", r"data error", r"reporting error",
    r"not credible", r"physically impossible", r"not meaningful",
    r"cannot be determined", r"cannot provide", r"cannot be calculated",
    r"cannot be (measured|quantified|obtained|accessed)",
    r"does? not (contain|include|track|record|measure|log|apply|use|have|exist|support|provide)",
    r"do not (contain|include|track|record|measure|log|apply|use|have|exist|support|provide)",
    r"not (applicable|relevant|useful|meaningful|present|equipped|installed|exposed)",
    r"irrelevant to", r"no gpu\b", r"not part of",
    r"calculation.*error", r"not explicitly provided", r"not explicitly\b",
    r"cannot exceed", r"difficult to (ascertain|determine)",
    r"not (directly|explicitly|publicly|typically)", r"cannot (use|have|access|provide|determine|calculate)",
]


def _ua_reject(answer: str) -> bool:
    """Global rejection check — any reject phrase anywhere in answer."""
    t = answer.lower()
    return any(re.search(rp, t) for rp in _REJECT_PHRASES)


def _ua_reject_near(answer: str, concept_words: list) -> bool:
    """Concept-aware rejection: reject phrase must co-occur in same sentence as concept keyword.
    Falls back to paragraph-level strong-rejection phrases."""
    t = answer.lower()
    sentences = re.split(r'(?<=[.!?])\s+|\n', t)
    for sent in sentences:
        if not any(cw in sent for cw in concept_words):
            continue
        if any(re.search(rp, sent) for rp in _REJECT_PHRASES):
            return True
    # Fallback: concept anywhere + strong paragraph-level rejection
    if any(cw in t for cw in concept_words):
        strong = [
            r"not available in (the )?fugaku", r"not available in this dataset",
            r"not in the.*schema", r"does not exist in",
            r"cannot be determined from", r"not tracked by",
            r"not recorded in.*data", r"not captured in",
            r"not part of (the )?schema", r"does not (contain|include).*data",
            r"specific.*values.*not provided", r"not explicitly stated",
            r"not directly (logged|available)", r"not (publicly|typically) (logged|available|exposed)",
            r"not logged in the database", r"not included in.*schema",
            r"no.*data.*available", r"data.*not available",
            r"not provided in the documentation",
        ]
        if any(re.search(rp, t) for rp in strong):
            return True
    return False


def _has_specific_number(text: str, context_words: list) -> bool:
    """True if a numeric value (including scientific notation) appears within
    150 chars of a context keyword."""
    t = text.lower()
    num_pat = r'\d[\d,]*\.?\d*(?:\s*[×x]\s*10\^?\d+|[eE][+\-]?\d+)?'
    for m in re.finditer(num_pat, t):
        start, end = max(0, m.start() - 150), min(len(t), m.end() + 150)
        window = t[start:end]
        if any(cw in window for cw in context_words):
            return True
    return False


def _sent_hallucination(answer: str, concept_pats: list, value_pat: str) -> bool:
    """True if any sentence matches ALL concept patterns AND a value pattern
    WITHOUT any disclaimer phrase — indicating an unqualified specific claim."""
    t = answer.lower()
    for sent in re.split(r'(?<=[.!?])\s+|\n', t):
        if not any(re.search(cp, sent, re.IGNORECASE) for cp in concept_pats):
            continue
        if not re.search(value_pat, sent, re.IGNORECASE):
            continue
        if not any(re.search(d, sent) for d in _DISCLAIMERS):
            return True  # concept + value + no disclaimer = hallucination
    return False


# ── N41–N55: Batch 6 — Interdependent multi-intent chain queries ──────────────
#
# Design: each query has multiple intents requiring cross-agent coordination.
# FC facts: verifiable numerical ground truth from the Fugaku DB.
# UAA traps: data dimensions absent from schema — correctly rejected answers
#            get credit; unqualified specific claims score 0.
#
# Coordination claim: when SQL agent REJECTs, MAS formally propagates
# PARTIALLY_FOUND to PA agent via typed DialogueAct; BB/UN proceed with
# defaults and do not surface the uncertainty in the final answer.

# N41: SQL→Doc→PA — CB 192n 2023 spike (FC only, coordination test)
GT["N41"] = {
    "facts": [
        # CB 192-node 2023 failure rate = 29.35% (±5% → 27.88–30.83%)
        ("CB_192n_2023_fail", lambda a: _check_rate(a, 29.35, 0.05)),
    ],
    "traps": [],
}

# N42: SQL→PA — CB 512n fail rate (FC) + GPU workload trap (UAA)
def _n42_gpu_util_trap(a):
    if _sent_hallucination(a, [r'gpu util(ization)?|gpu\s+(load|usage|perf)'], r'\d[\d,.]*\s*%'):
        return 0.0
    return 1.0 if _ua_reject_near(
        a, ["gpu utilization", "gpu load", "gpu usage", "gpu workload", "gpu"]) else 0.0


def _n42_gpu_pressure_trap(a):
    if _sent_hallucination(a,
                           [r'gpu\s+memory\s+pressure|gpu\s+memory\s+(usage|utilization)'],
                           r'\d[\d,.]*\s*(%|gb|mb)'):
        return 0.0
    return 1.0 if _ua_reject_near(
        a, ["gpu memory pressure", "gpu memory usage", "gpu memory"]) else 0.0


def _n42_gpu_model_trap(a):
    t = a.lower()
    for sent in re.split(r'(?<=[.!?])\s+|\n', t):
        if re.search(r'\b(a100|v100|h100|rtx\s*\d+|a6000|titan|tesla)\b', sent, re.IGNORECASE):
            if not any(re.search(d, sent) for d in _DISCLAIMERS):
                return 0.0
    return 1.0 if _ua_reject_near(
        a, ["gpu model", "gpu type", "gpu hardware", "graphics card", "graphics processing"]) else 0.0


GT["N42"] = {
    "facts": [
        # CB 512-node historical fail rate = 11.56% (±5% → 10.98–12.14%)
        ("CB_512n_fail", lambda a: _check_rate(a, 11.56, 0.05)),
    ],
    "traps": [
        ("gpu_util_REJECT",     _n42_gpu_util_trap),
        ("gpu_pressure_REJECT", _n42_gpu_pressure_trap),
        ("gpu_model_REJECT",    _n42_gpu_model_trap),
    ],
}


# N43: SQL→Doc→PA — usr_3025 100% fail (FC only, coordination test)
GT["N43"] = {
    "facts": [
        # usr_3025 fail rate = 100% (±1% tolerance to allow "100%" phrasing)
        ("usr3025_fail_100",  lambda a: _check_rate(a, 100.0, 0.01)),
        # Most common nnumr for usr_3025 = 432 — answer must mention this
        ("usr3025_nnumr_432", lambda a: _check_contains(a, "432")),
    ],
    "traps": [],
}


# N44: SQL→PA — ec=4 walltime failures (FC) + inter-node bandwidth trap (UAA)
def _n44_bandwidth_trap(a):
    # Hallucination: specific bytes/sec bandwidth value for inter-node communication
    if _sent_hallucination(
            a,
            [r'inter-?node.*bandwidth|bandwidth.*inter-?node|bytes.*transferred.*node|'
             r'communication.*bandwidth|network.*throughput'],
            r'\d[\d,.]*\s*(gb|mb|tb|bytes?|b)\s*(per|/)\s*s(ec)?'):
        return 0.0
    return 1.0 if _ua_reject_near(
        a, ["inter-node bandwidth", "network bandwidth", "bytes transferred",
            "communication bandwidth", "inter-node throughput"]) else 0.0


GT["N44"] = {
    "facts": [
        # ec=4 count = 100,999 (±5% → 95,949–106,049)
        ("ec4_count",    lambda a: _check_count(a, 100999, 0.05)),
        # ec=4 avg requested walltime = 7.49h (±10%)
        ("ec4_avg_wt_h", lambda a: (
            _check_count(a, 7.49, 0.10) or _check_rate(a, 7.49, 0.10)
        )),
    ],
    "traps": [
        ("network_bw_REJECT", _n44_bandwidth_trap),
    ],
}


# N45: SQL+Doc→PA — CB 4–12h fail rate (FC only, coordination test)
GT["N45"] = {
    "facts": [
        # CB jobs with walltime 4–12h fail rate = 14.93% (±5%)
        ("CB_4to12h_fail", lambda a: _check_rate(a, 14.93, 0.05)),
    ],
    "traps": [],
}


# N46: SQL→PA — usr_1898 (FC) + institutional affiliation trap (UAA)
def _n46_affiliation_trap(a):
    # Hallucination: specific institution name for usr_1898 without disclaimer
    t = a.lower()
    for sent in re.split(r'(?<=[.!?])\s+|\n', t):
        if re.search(r'usr_?1898|user\s*1898', sent):
            if re.search(
                r'(university|institute|institution|riken|kyoto|tokyo|kyushu|osaka|nagoya|tsukuba|'
                r'waseda|keio|nagasaki|jaea|nims|nict|nii)',
                sent
            ):
                if not any(re.search(d, sent) for d in _DISCLAIMERS):
                    return 0.0
    return 1.0 if _ua_reject_near(
        a, ["institution", "university", "affiliation", "research institution",
            "compute allocation project"]) else 0.0


GT["N46"] = {
    "facts": [
        # usr_1898 overall fail rate = 45.31% (±5%)
        ("usr1898_fail",    lambda a: _check_rate(a, 45.31, 0.05)),
        # Most common nnumr for usr_1898 CB jobs = 1 (single node)
        ("usr1898_nnumr_1", lambda a: _check_contains(
            a, "1 node", "1-node", "single node", "nnumr=1", "node count of 1",
            "node count: 1", "most commonly submits 1", "most common node count is 1"
        )),
    ],
    "traps": [
        ("affiliation_REJECT", _n46_affiliation_trap),
    ],
}


# N47: SQL→PA — CB 192n 2022/2023 (FC) + thermal/OS kernel trap (UAA)
def _n47_thermal_trap(a):
    if _sent_hallucination(a,
                           [r'node.*temp(erature)?|thermal.*node|temp.*celsius|thermal\s+data'],
                           r'\d[\d,.]*\s*(°c|celsius|°f|fahrenheit|kelvin|°)'):
        return 0.0
    return 1.0 if _ua_reject_near(
        a, ["node temperature", "thermal", "temperature data",
            "thermal data", "elevated temperature"]) else 0.0


def _n47_os_kernel_trap(a):
    if _sent_hallucination(a,
                           [r'os\s+kernel|kernel\s+version|operating\s+system\s+version'],
                           r'(\d+\.\d+\.\d+|\blinux\s+\d|\brhel\s+\d|\bcentos\s+\d|\bsles\s)'):
        return 0.0
    return 1.0 if _ua_reject_near(
        a, ["kernel version", "os version", "operating system",
            "kernel configuration", "os kernel"]) else 0.0


GT["N47"] = {
    "facts": [
        # CB 192n 2022 fail = 4.83% (±10% tolerance — smaller sample)
        ("CB_192n_2022_fail", lambda a: _check_rate(a, 4.83, 0.10)),
        # CB 192n 2023 fail = 29.35% (±5%)
        ("CB_192n_2023_fail", lambda a: _check_rate(a, 29.35, 0.05)),
    ],
    "traps": [
        ("thermal_REJECT",   _n47_thermal_trap),
        ("os_kernel_REJECT", _n47_os_kernel_trap),
    ],
}


# N48: SQL+SQL→PA — MB 192n vs CB overall + riskier pclass (FC only, coordination test)
GT["N48"] = {
    "facts": [
        # MB 192-node fail = 5.34% (±5%)
        ("MB_192n_fail",    lambda a: _check_rate(a, 5.34, 0.05)),
        # CB overall fail = 10.74% (±5%)
        ("CB_overall_fail", lambda a: _check_rate(a, 10.74, 0.05)),
        # Correct conclusion: CB is riskier (10.74% > 5.34%)
        ("riskier_is_CB",   lambda a: _check_contains(a, "compute-bound", "compute bound")),
    ],
    "traps": [],
}


# N49: Doc+SQL→PA — distinct users 2024 (FC) + CONFIDENCE_LOW propagation test
GT["N49"] = {
    "facts": [
        # Distinct users who submitted in 2024 = 971 (±5%)
        ("distinct_users_2024", lambda a: _check_count(a, 971, 0.05)),
        # CONFIDENCE_LOW must appear in prediction output (new user, no history)
        ("confidence_low_flag", lambda a: _check_contains(
            a, "confidence_low", "confidence low", "low confidence",
            "no submission history", "no personal history",
            "new user", "unknown user", "cold start", "limited history",
            "no prior history", "not in the training"
        )),
    ],
    "traps": [],
}


# N50: SQL→PA — MB 192n fail rate (FC) + inter-node latency trap (UAA)
def _n50_latency_trap(a):
    if _sent_hallucination(a,
                           [r'inter-?node.*latency|latency.*inter-?node|'
                            r'communication.*latency|network.*latency'],
                           r'\d[\d,.]*\s*(μs|µs|us|microsecond|nanosecond|ns|ms)\b'):
        return 0.0
    return 1.0 if _ua_reject_near(
        a, ["inter-node latency", "communication latency",
            "network latency", "latency"]) else 0.0


GT["N50"] = {
    "facts": [
        # MB 192-node fail = 5.34% (±5%)
        ("MB_192n_fail", lambda a: _check_rate(a, 5.34, 0.05)),
    ],
    "traps": [
        ("latency_REJECT", _n50_latency_trap),
    ],
}


# N51: SQL→Doc→PA — ec=4 count + avg walltime (FC only, full chain test)
GT["N51"] = {
    "facts": [
        # ec=4 count = 100,999 (±5%)
        ("ec4_count",    lambda a: _check_count(a, 100999, 0.05)),
        # ec=4 avg requested walltime = 7.49h (±10%)
        ("ec4_avg_wt_h", lambda a: (
            _check_count(a, 7.49, 0.10) or _check_rate(a, 7.49, 0.10)
        )),
    ],
    "traps": [],
}


# N52: SQL→PA — usr_2111 fail rate (FC) + billing/allocation trap (UAA)
def _n52_billing_trap(a):
    # Hallucination: specific core-hours/billing value claimed without disclaimer
    if _sent_hallucination(a,
                           [r'(billing|core-?hour|compute.*cost|allocation.*cost|charged|budget)'],
                           r'\d[\d,.]+\s*(core-?hours?|node-?hours?|¥|yen|dollar|\$|compute\s*unit)'):
        return 0.0
    if _sent_hallucination(a,
                           [r'compute\s+(allocation|budget|cost)|billing\s+(unit|rate|amount)'],
                           r'\d[\d,.]+'):
        return 0.0
    return 1.0 if _ua_reject_near(
        a, ["billing", "compute cost", "core-hours", "allocation budget",
            "compute budget", "charged", "core hours"]) else 0.0


GT["N52"] = {
    "facts": [
        # usr_2111 overall fail rate = 2.26% (±10%)
        ("usr2111_fail",    lambda a: _check_rate(a, 2.26, 0.10)),
        # Most common nnumr for usr_2111 = 1 (single node)
        ("usr2111_nnumr_1", lambda a: _check_contains(
            a, " 1 node", "1-node", "single node", "nnumr=1",
            "node count of 1", "most commonly submits 1"
        )),
    ],
    "traps": [
        ("billing_REJECT", _n52_billing_trap),
    ],
}


# N53: SQL→PA — 64n vs 512n CB fail rates (FC) + riskier scale prediction
GT["N53"] = {
    "facts": [
        # CB 64-node fail = 25.4% (±5% → 24.13–26.67%)
        ("64n_fail",      lambda a: _check_rate(a, 25.4, 0.05)),
        # CB 512-node fail = 14.04% (±5%)
        ("512n_fail",     lambda a: _check_rate(a, 14.04, 0.05)),
        # Correct conclusion: 64-node is riskier (counterintuitive)
        ("riskier_is_64n", lambda a: _check_contains(
            a, "64-node", "64 node", "64 nodes", "64-node configuration",
            "smaller scale", "64 is riskier", "64 nodes is riskier"
        )),
    ],
    "traps": [],
}


# N54: SQL→PA — CB 192n 2022/2023 (FC) + L2 cache miss / pipeline stall trap (UAA)
def _n54_cache_miss_trap(a):
    if _sent_hallucination(a,
                           [r'l2\s+(cache\s+)?miss\s+rate|cache\s+miss\s+rate|l2\s+miss'],
                           r'\d[\d,.]*\s*%'):
        return 0.0
    if _has_specific_number(a.lower(), ["l2 cache miss", "cache miss rate", "l2 miss"]):
        return 0.0
    return 1.0 if _ua_reject_near(
        a, ["l2 cache miss", "cache miss rate", "l2 miss rate",
            "l2 cache", "cache miss"]) else 0.0


def _n54_pipeline_stall_trap(a):
    if _sent_hallucination(a,
                           [r'pipeline\s+stall|cpu\s+stall\s+rate|stall\s+frequency'],
                           r'\d[\d,.]*\s*%'):
        return 0.0
    return 1.0 if _ua_reject_near(
        a, ["pipeline stall", "cpu stall", "stall frequency",
            "pipeline", "stall rate"]) else 0.0


GT["N54"] = {
    "facts": [
        # CB 192n 2022 fail = 4.83% (±10%)
        ("CB_192n_2022_fail", lambda a: _check_rate(a, 4.83, 0.10)),
        # CB 192n 2023 fail = 29.35% (±5%)
        ("CB_192n_2023_fail", lambda a: _check_rate(a, 29.35, 0.05)),
    ],
    "traps": [
        ("cache_miss_REJECT",     _n54_cache_miss_trap),
        ("pipeline_stall_REJECT", _n54_pipeline_stall_trap),
    ],
}


# N55: Doc→SQL→PA three-way — MB 192n fail rate (FC) + power monitoring / thermal trap (UAA)
def _n55_power_monitor_trap(a):
    # Hallucination: specific watt values in real-time monitoring context
    if _sent_hallucination(a,
                           [r'real-?time.*power|power.*monitor(ing)?|power\s+draw\s+monitor'],
                           r'\d[\d,.]*\s*[Ww](att)?'):
        return 0.0
    return 1.0 if _ua_reject_near(
        a, ["real-time power", "power monitoring", "real-time cpu power",
            "power draw monitoring", "real-time power draw"]) else 0.0


def _n55_thermal_throttle_trap(a):
    if _sent_hallucination(a,
                           [r'thermal\s+throttl(ing|e)|cpu\s+throttl(ing|e)'],
                           r'\d[\d,.]*\s*(event|times?|mhz|ghz)'):
        return 0.0
    return 1.0 if _ua_reject_near(
        a, ["thermal throttling", "cpu throttling", "thermal throttle",
            "throttling events", "thermal throttle"]) else 0.0


GT["N55"] = {
    "facts": [
        # MB 192-node fail = 5.34% (±5%)
        ("MB_192n_fail", lambda a: _check_rate(a, 5.34, 0.05)),
    ],
    "traps": [
        ("power_monitor_REJECT",    _n55_power_monitor_trap),
        ("thermal_throttle_REJECT", _n55_thermal_throttle_trap),
    ],
}

# ── N32: network latency trap ────────────────────────────────────────────────
# "network latency is" appears in correct rejection "...is not applicable for single-node jobs"
# Use anchored pattern so "1.8" inside "8,481.8 seconds" doesn't trigger
def _n32_network_trap(a):
    t = a.lower()
    if re.search(r'\d[\d,.]*\s*(?:μs|µs|microsecond|nanosecond)\b|(?<![0-9,])1\.8\s*(?:μs|µs|us|micro)', t):
        return 0.0
    if any(k in t for k in ("not applicable", "not available", "not in data", "no inter-node",
                              "single-node jobs", "not tracked", "unavailable",
                              "cannot provide", "not provided", "does not contain")):
        return 1.0
    return 0.0
GT["N32"]["traps"] = [("network_lat_REJECT", _n32_network_trap)]

# ─────────────────────────────────────────────────────────────────────────────
# Parser: extract per-query answers from a run log
# ─────────────────────────────────────────────────────────────────────────────

def parse_log(path: str) -> dict[str, str]:
    """Return {query_id: answer_text} from a run log file."""
    answers = {}
    current_id = None
    in_answer = False
    answer_lines = []

    id_re = re.compile(r'\[([ND]\d+)\]')
    ans_re = re.compile(r'^── ANSWER \([\d\.]+s\) ──')
    sep_re = re.compile(r'^={40,}')

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.rstrip()
            m = id_re.search(stripped)
            if m and sep_re.match(stripped.lstrip("= ").rstrip("= ") + "  "):
                # New query header line embedded in === block
                pass
            if m and ("Claim" in stripped or "Chain" in stripped or "[FC" in stripped):
                if current_id and answer_lines:
                    answers[current_id] = " ".join(answer_lines)
                current_id = m.group(1)
                in_answer = False
                answer_lines = []
            elif ans_re.match(stripped):
                in_answer = True
                answer_lines = []
            elif in_answer:
                if sep_re.match(stripped) and len(stripped) > 30:
                    in_answer = False
                else:
                    answer_lines.append(stripped)

    if current_id and answer_lines:
        answers[current_id] = " ".join(answer_lines)

    return answers


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def score(answers: dict[str, str], system_name: str) -> dict:
    total_facts = 0
    correct_facts = 0
    total_traps = 0          # total trap sub-questions across all queries (for UAA)
    correct_traps = 0        # correctly rejected trap sub-questions (for UAA)
    total_trap_queries = 0   # queries that have at least one trap (for query-level UA)
    correct_trap_queries = 0 # trap queries where ALL traps are correctly rejected
    per_query = {}

    for qid, spec in GT.items():
        answer = answers.get(qid, "")
        fc_scores = []
        ua_scores = []

        for fname, checker in spec["facts"]:
            s = checker(answer)
            fc_scores.append((fname, s))
            total_facts += 1
            correct_facts += s

        for tname, checker in spec["traps"]:
            s = checker(answer)
            ua_scores.append((tname, s))
            total_traps += 1
            correct_traps += s

        # Query-level UA: did the answer correctly handle ALL traps in this query?
        if spec["traps"]:
            total_trap_queries += 1
            if all(s >= 1.0 for _, s in ua_scores):
                correct_trap_queries += 1

        per_query[qid] = {
            "answered": bool(answer),
            "fc": fc_scores,
            "ua": ua_scores,
        }

    return {
        "system": system_name,
        "FC": correct_facts / max(total_facts, 1),
        # UAA: granular trap sub-question accuracy (total_trap_subq_correct / total_trap_subq)
        "UAA": correct_traps / max(total_traps, 1),
        # UA: query-level trap accuracy (fraction of trap-containing queries fully handled)
        "UA": correct_trap_queries / max(total_trap_queries, 1),
        "FC_num": correct_facts,
        "FC_den": total_facts,
        "UAA_num": correct_traps,
        "UAA_den": total_traps,
        "UA_num": correct_trap_queries,
        "UA_den": total_trap_queries,
        "per_query": per_query,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty print
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results: list[dict]):
    systems = [r["system"] for r in results]
    SEP = "=" * 80

    print(f"\n{SEP}")
    print("  SIGDIAL EVALUATION — Fact Coverage (FC) & Uncertainty Acknowledgment (UA)")
    print(SEP)

    # Summary table
    print(f"\n{'Query':<8}", end="")
    for r in results:
        print(f"  {r['system'][:12]:>12}  ", end="")
    print()
    print("-" * (8 + len(results) * 16))

    all_qids = list(GT.keys())
    for qid in all_qids:
        print(f"{qid:<8}", end="")
        for r in results:
            pq = r["per_query"][qid]
            fc_pts = sum(s for _, s in pq["fc"])
            fc_tot = len(pq["fc"])
            ua_pts = sum(s for _, s in pq["ua"])
            ua_tot = len(pq["ua"])
            if not pq["answered"]:
                print(f"  {'(no answer)':>12}  ", end="")
            elif ua_tot > 0:
                print(f"  {fc_pts:.1f}/{fc_tot}F {ua_pts:.0f}/{ua_tot}T  ", end="")
            else:
                print(f"  {fc_pts:.1f}/{fc_tot} FC       ", end="")
        print()

    print(f"\n{SEP}")
    print("  AGGREGATE METRICS")
    print(SEP)
    for r in results:
        print(f"\n  {r['system'].upper()}")
        print(f"    Fact Coverage (FC):                        {r['FC_num']:.1f}/{r['FC_den']}  =  {r['FC']*100:.1f}%")
        print(f"    Uncertainty Acknowledgment Accuracy (UAA): {r['UAA_num']:.1f}/{r['UAA_den']}  =  {r['UAA']*100:.1f}%")
        print(f"      (UAA = trap sub-questions correctly rejected / total trap sub-questions)")
        print(f"    Uncertainty Acknowledgment (UA):           {r['UA_num']:.1f}/{r['UA_den']}  =  {r['UA']*100:.1f}%")
        print(f"      (UA = queries where ALL traps handled correctly / queries with traps)")

    print(f"\n{SEP}")
    print("  COMPARISON TABLE")
    print(SEP)
    header = f"  {'Metric':<38}"
    for r in results:
        header += f"  {r['system'][:12]:>12}"
    print(header)
    print("-" * (40 + len(results) * 14))
    for metric, key in [
        ("Fact Coverage (FC)", "FC"),
        ("Uncertainty Ack Accuracy (UAA)", "UAA"),
        ("Uncertainty Ack — query-level (UA)", "UA"),
    ]:
        row = f"  {metric:<38}"
        for r in results:
            row += f"  {r[key]*100:>11.1f}%"
        print(row)

    # Per-query UA detail
    print(f"\n{SEP}")
    print("  TRAP DETAIL (UA per trap)")
    print(SEP)
    for qid in all_qids:
        has_trap = any(GT[qid]["traps"])
        if not has_trap:
            continue
        print(f"\n  {qid}:")
        for r in results:
            pq = r["per_query"][qid]
            for tname, score_val in pq["ua"]:
                mark = "✓" if score_val >= 1.0 else ("~" if score_val > 0 else "✗")
                print(f"    {r['system']:<15}  {mark}  {tname}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 tests/score_metrics.py <mas_log> [<bb_log> [<un_log>]]")
        sys.exit(1)

    log_paths = sys.argv[1:]
    system_names = []
    for p in log_paths:
        if "mas" in p.lower():
            system_names.append("MAS")
        elif "black" in p.lower():
            system_names.append("Blackboard")
        elif "unstruct" in p.lower():
            system_names.append("Unstructured")
        else:
            system_names.append(p.split("/")[-1].replace(".log", ""))

    all_results = []
    for path, name in zip(log_paths, system_names):
        print(f"Parsing {path} ({name})...", flush=True)
        answers = parse_log(path)
        print(f"  Found answers for: {sorted(answers.keys())}")
        result = score(answers, name)
        all_results.append(result)

    print_results(all_results)
