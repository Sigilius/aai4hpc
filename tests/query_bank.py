"""
tests/query_bank.py  —  v2.0  (redesigned for SIGDIAL 2026)

50 queries redesigned around three principles:
  1. Natural language — queries an HPC operator would actually type
  2. Dependency-driven coordination — intent B's input comes from intent A's output
  3. SIGDIAL claim coverage — each core query maps to a specific paper claim

Structure
---------
  Q01–Q05  Type A  — Fully answerable, sanity checks (Claim 1 baseline)
  Q06–Q13  Type B  — Partially answerable, uncertainty propagation (Claims 1 & 2)
  Q14–Q20  Type C  — Unanswerable / low-confidence (Claims 2 & 3)
  Q21–Q50  Breadth — Natural HPC operator scenarios, mixed intents (core-3 checks)

Query types (from SIGDIAL paper evaluation schema)
--------------------------------------------------
  A — All facts in data, clear schema, no uncertainty; all systems should pass
  B — Some facts missing, inferred, or sparse; MAS must flag, baselines often assert
  C — Data absent, out-of-range, or model has too few samples; MAS rejects / caveats,
      baselines hallucinate — the "killer data point" for Claim 3

Ground-truth DB facts (verified against fugaku.duckdb)
------------------------------------------------------
  Total jobs:                  25,866,900
  Failed jobs:                  2,558,519   (~9.88%)
  Dataset range:                2021-03-01 → 2024-05-08
  compute-bound fail rate:     10.74%
  memory-bound fail rate:       9.38%
  avg nnumr of failed CB jobs: ~20 nodes
  avg elpl of failed CB jobs:  ~33,365s (~9.3 h)
  jobs > 24 h:                  335,252
  jobs > 12 h:                1,603,893
  jobs > 512 nodes:             419,273
  nnumr=128 MB fail rate:      10.74%   (5,938 / 55,260)
  nnumr=64  MB fail rate:      25.18%  (14,017 / 55,677)
  nnumr=972 CB jobs:           ~1 job  → LOW_SAMPLE guaranteed
  nnumr=914 total:              1 job  → LOW_SAMPLE guaranteed
  unique users:                 3,457
  usr_1898 total jobs:      1,252,185
  jobenv_req_8 count:              67
  jobenv_req_2 count:               5
  freq_req=1600 MHz:                2 jobs
  NOT in schema: OS, GPU, billing, network latency, CPU arch, temperature
"""
from __future__ import annotations

QUERY_BANK: list[dict] = [

    # ══════════════════════════════════════════════════════════════════════════
    # TYPE A  —  FULLY ANSWERABLE  (Q01–Q05)
    # All facts present, schema clear, no uncertainty flags expected.
    # Every system should pass these; they anchor the comparison baseline.
    # Strict evaluation (7 checks).
    # SIGDIAL Claim 1: MAS handles these correctly through coordination.
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "Q01", "tier": "type_a", "query_type": "A",
        "sigdial_claim": 1,
        "query": (
            "I want to understand the overall health of Fugaku's job queue. "
            "How many jobs ran in total and what percentage failed?"
        ),
        "intent_count": 1, "intent_types": ["sql"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "synthesizer", "reflector"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": [],
        "coordination_chain": "sql_agent → synthesizer → reflector(CONFIRM) → gateway",
        "checks": {
            "answer_min_length": 50,
            "must_contain": ["25,866", "2,558"],
            "must_not_contain": [],
        },
        "notes": "Type A sanity. Anchors fact coverage baseline for all 4 systems.",
    },
    {
        "id": "Q02", "tier": "type_a", "query_type": "A",
        "sigdial_claim": 1,
        "query": (
            "How do compute-bound and memory-bound jobs compare in terms of "
            "average duration and failure rate?"
        ),
        "intent_count": 1, "intent_types": ["sql"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "synthesizer", "reflector"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": [],
        "coordination_chain": "sql_agent(GROUP BY pclass) → synthesizer → reflector(CONFIRM)",
        "checks": {
            "answer_min_length": 80,
            "must_contain": ["compute-bound", "memory-bound"],
            "must_not_contain": [],
        },
        "notes": "Type A comparative SQL. Tests GROUP BY pclass on two metrics simultaneously.",
    },
    {
        "id": "Q03", "tier": "type_a", "query_type": "A",
        "sigdial_claim": 1,
        "query": (
            "Before I submit, I want to understand the risk profile and see the "
            "correct pjsub directives for a 64-node compute-bound job running 4 hours."
        ),
        "intent_count": 2, "intent_types": ["predict", "doc"],
        "entry_agent": "pa_agent",
        "expected_agents": ["pa_agent", "doc_agent", "synthesizer", "reflector"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": ["knowledge_gap"],
        "coordination_chain": "pa_agent(predict) → doc_agent(KNOWLEDGE_GAP: pjsub directives) → synthesizer",
        "checks": {
            "answer_min_length": 120,
            "must_contain": ["pjsub", "fail"],
            "must_not_contain": [],
        },
        "notes": (
            "Type A two-intent: prediction + documentation. PA predicts (23K similar "
            "jobs, reliable), then delegates KNOWLEDGE_GAP to doc for directives."
        ),
    },
    {
        "id": "Q04", "tier": "type_a", "query_type": "A",
        "sigdial_claim": 1,
        "query": "How many jobs ran with more than 512 nodes, and what's their failure rate?",
        "intent_count": 1, "intent_types": ["sql"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "synthesizer", "reflector"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": [],
        "coordination_chain": "sql_agent(COUNT + AVG WHERE nnumr>512) → synthesizer → reflector",
        "checks": {
            "answer_min_length": 50,
            "must_contain": ["419,273"],
            "must_not_contain": [],
        },
        "notes": "Type A filtered aggregation. Tests nnumr threshold filtering.",
    },
    {
        "id": "Q05", "tier": "type_a", "query_type": "A",
        "sigdial_claim": 1,
        "query": (
            "Who are the top 5 users by total number of jobs submitted? "
            "I'd also like to know what the pjstat command shows so I can monitor my own jobs."
        ),
        "intent_count": 2, "intent_types": ["sql", "doc"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "doc_agent", "synthesizer", "reflector"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": ["knowledge_gap"],
        "coordination_chain": "sql_agent(TOP-5 GROUP BY usr) → doc_agent(KNOWLEDGE_GAP: pjstat) → synthesizer",
        "checks": {
            "answer_min_length": 100,
            "must_contain": ["usr_1898", "pjstat"],
            "must_not_contain": [],
        },
        "notes": "Type A sql + doc. usr_1898 is the top user (1.25M jobs). Tests delegation to doc.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # TYPE B  —  PARTIALLY ANSWERABLE  (Q06–Q13)
    # Some facts answerable, some absent or sparse.
    # KEY CLAIM 2 evidence: MAS must flag partial facts; baselines often assert them.
    # KEY CLAIM 1 evidence: several require coordination for the answer to make sense.
    # Strict evaluation (7 checks).
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "Q06", "tier": "type_b", "query_type": "B",
        "sigdial_claim": 1,
        "query": (
            "I keep seeing failures in my group's compute-bound jobs. Can you look up "
            "the average node count and walltime of recently failed compute-bound jobs, "
            "then use those averages as parameters to predict whether that config is risky?"
        ),
        "intent_count": 2, "intent_types": ["sql", "predict"],
        "entry_agent": "pa_agent",
        "expected_agents": ["pa_agent", "sql_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": ["data_insufficiency"],
        "coordination_chain": (
            "pa_agent detects pre-predict param need → sql_agent(DATA_INSUFFICIENCY: "
            "avg nnumr/elpl of failed CB jobs) → pa_agent(predict with sql results) → synthesizer"
        ),
        "checks": {
            "answer_min_length": 100,
            "must_contain": ["average", "fail"],
            "must_not_contain": [],
        },
        "notes": (
            "COORDINATION LOAD-BEARING: SQL output IS the prediction input "
            "(avg_nnumr≈20, avg_elpl≈33,365s). Remove the PA→SQL message and the "
            "prediction has no valid parameters. Key evidence for Claim 1."
        ),
    },
    {
        "id": "Q07", "tier": "type_b", "query_type": "B",
        "sigdial_claim": 2,
        "query": (
            "We've had unusual failures in jobs using the jobenv_req_8 environment "
            "recently. Can you check the failure rate for that environment type and "
            "explain what it actually means?"
        ),
        "intent_count": 2, "intent_types": ["sql", "doc"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "doc_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": ["knowledge_gap"],
        "coordination_chain": (
            "sql_agent(jobenv_req_8 failure rate, 67 jobs) → "
            "doc_agent(KNOWLEDGE_GAP: what jobenv_req_8 means) → synthesizer"
        ),
        "checks": {
            "answer_min_length": 80,
            "must_contain": ["jobenv_req_8"],
            "must_not_contain": ["no data", "not found"],
        },
        "notes": (
            "Type B: 67 jobs — sparse but real. System must return actual count, "
            "not NOT_FOUND. Doc explains the environment type via KNOWLEDGE_GAP delegation."
        ),
    },
    {
        "id": "Q08", "tier": "type_b", "query_type": "B",
        "sigdial_claim": 2,
        "query": (
            "For my capacity planning report: what is the failure rate per job class, "
            "the average power draw per class, and the billing cost breakdown per class?"
        ),
        "intent_count": 3, "intent_types": ["sql", "sql", "sql_reject"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": ["partially_found"],
        "partial_available": True, "delegation_triggers": [],
        "coordination_chain": (
            "sql_agent decomposes → 2 SQL queries succeed (fail rate + power) + "
            "1 CANNOT_GENERATE (billing) → PARTIALLY_FOUND flag → synthesizer"
        ),
        "checks": {
            "answer_min_length": 80,
            "must_contain": ["compute-bound", "memory-bound"],
            "must_not_contain": ["billing cost is", "costs $", "¥", "per unit"],
        },
        "notes": (
            "CLAIM 2 key test: failure rate ✓, power ✓, billing ✗. "
            "MAS must say 'not tracked' for billing; baseline often asserts a cost figure."
        ),
    },
    {
        "id": "Q09", "tier": "type_b", "query_type": "B",
        "sigdial_claim": 2,
        "query": (
            "I'm a new researcher who just got access to Fugaku. What failure risk "
            "should I expect for a 64-node compute-bound job with 2-hour walltime, "
            "and how confident is that estimate given I have no submission history?"
        ),
        "intent_count": 1, "intent_types": ["predict"],
        "entry_agent": "pa_agent",
        "expected_agents": ["pa_agent", "synthesizer", "reflector"],
        "expected_reject": False, "expected_flags": ["confidence_low"],
        "partial_available": False, "delegation_triggers": [],
        "coordination_chain": (
            "pa_agent(predict, no usr → global rates) → CONFIDENCE_LOW flag → synthesizer"
        ),
        "checks": {
            "answer_min_length": 80,
            "must_contain": ["fail", "confidence"],
            "must_not_contain": [],
        },
        "notes": (
            "Type B: prediction runs but confidence is reduced — no user history means "
            "the model uses global rates. CONFIDENCE_LOW must reach the final answer."
        ),
    },
    {
        "id": "Q10", "tier": "type_b", "query_type": "B",
        "sigdial_claim": 1,
        "query": (
            "Can you predict the failure probability for a 256-node memory-bound job "
            "with 6-hour walltime, and then cross-check that against what the historical "
            "data actually shows for similar configurations?"
        ),
        "intent_count": 2, "intent_types": ["predict", "sql"],
        "entry_agent": "pa_agent",
        "expected_agents": ["pa_agent", "sql_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": ["data_insufficiency"],
        "coordination_chain": (
            "pa_agent(predict) → sql_agent(DATA_INSUFFICIENCY: "
            "historical rate for nnumr=256 MB) → synthesizer(compare model vs history)"
        ),
        "checks": {
            "answer_min_length": 100,
            "must_contain": ["historical", "fail"],
            "must_not_contain": [],
        },
        "notes": (
            "Type B: post-prediction SQL comparison. PA predicts, then explicitly "
            "verifies against historical data. Tests DATA_INSUFFICIENCY post-predict pattern."
        ),
    },
    {
        "id": "Q11", "tier": "type_b", "query_type": "B",
        "sigdial_claim": 1,
        "query": (
            "I'm planning a long overnight job. What does the walltime limit policy say "
            "for large jobs, and what do the statistics show for jobs that ran over 12 hours — "
            "how many were there and what was their failure rate?"
        ),
        "intent_count": 2, "intent_types": ["doc", "sql"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "doc_agent", "synthesizer", "reflector"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": ["knowledge_gap"],
        "coordination_chain": (
            "sql_agent(COUNT/fail rate WHERE duration>43200) → "
            "doc_agent(KNOWLEDGE_GAP: walltime policy) → synthesizer"
        ),
        "checks": {
            "answer_min_length": 100,
            "must_contain": ["1,603,893"],
            "must_not_contain": [],
        },
        "notes": (
            "Type B: sql gets the stats (1.6M jobs over 12h), doc gets the policy. "
            "Both needed for a complete answer — coordination makes the answer coherent."
        ),
    },
    {
        "id": "Q12", "tier": "type_b", "query_type": "B",
        "sigdial_claim": 2,
        "query": (
            "What is the average duration and failure rate for compute-bound jobs vs "
            "memory-bound jobs, and also what's the average GPU utilization per job class?"
        ),
        "intent_count": 3, "intent_types": ["sql", "sql", "sql_reject"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": ["partially_found"],
        "partial_available": True, "delegation_triggers": [],
        "coordination_chain": (
            "sql_agent decomposes → duration/fail rate ✓, GPU utilization ✗ (no GPU on Fugaku) "
            "→ PARTIALLY_FOUND → synthesizer"
        ),
        "checks": {
            "answer_min_length": 80,
            "must_contain": ["compute-bound", "memory-bound"],
            "must_not_contain": ["gpu utilization is", "% gpu", "gpu usage"],
        },
        "notes": (
            "CLAIM 2 + hallucination guard: Fugaku has no GPUs. "
            "Baseline often guesses a GPU utilization figure. "
            "MAS must say Fugaku has no GPU data."
        ),
    },
    {
        "id": "Q13", "tier": "type_b", "query_type": "B",
        "sigdial_claim": 1,
        "query": (
            "My team typically runs failed jobs with about 20 nodes and 9-hour walltimes. "
            "Based on that profile, is submitting a similar job actually risky, "
            "and what percentage of all jobs fit that profile?"
        ),
        "intent_count": 2, "intent_types": ["predict", "sql"],
        "entry_agent": "pa_agent",
        "expected_agents": ["pa_agent", "sql_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": ["data_insufficiency"],
        "coordination_chain": (
            "pa_agent(predict nnumr≈20, elpl≈32400) → "
            "sql_agent(DATA_INSUFFICIENCY: % jobs matching that profile) → synthesizer"
        ),
        "checks": {
            "answer_min_length": 100,
            "must_contain": ["fail"],
            "must_not_contain": [],
        },
        "notes": (
            "Type B: user provides parameters that match the average failed CB job profile "
            "(avg_nnumr≈20, avg_elpl≈33365s). PA predicts, SQL contextualizes the profile."
        ),
    },

    # ══════════════════════════════════════════════════════════════════════════
    # TYPE C  —  UNANSWERABLE / LOW-CONFIDENCE  (Q14–Q20)
    # Data absent, out of range, or model has too few samples.
    # KEY CLAIM 3: baselines hallucinate; MAS rejects or heavily caveats.
    # High-value for "uncertainty honesty ≠ correctness" argument.
    # Strict evaluation (7 checks).
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "Q14", "tier": "type_c", "query_type": "C",
        "sigdial_claim": 3,
        "query": (
            "How do failure rates compare between jobs running on GPU-accelerated "
            "nodes versus CPU-only nodes on Fugaku?"
        ),
        "intent_count": 1, "intent_types": ["sql_reject"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent"],
        "expected_reject": True, "expected_flags": [],
        "partial_available": False, "delegation_triggers": [],
        "coordination_chain": "sql_agent(CANNOT_GENERATE: no GPU column) → REJECT → gateway",
        "checks": {
            "answer_min_length": 40,
            "must_contain": [],
            "must_not_contain": ["gpu failure rate is", "% on gpu", "accelerated nodes had"],
        },
        "notes": (
            "CLAIM 3 key test: Fugaku has NO GPUs. Baseline confidently invents a "
            "GPU failure rate. MAS must reject. The fabricated answer from baselines "
            "is the clearest hallucination example in the corpus."
        ),
    },
    {
        "id": "Q15", "tier": "type_c", "query_type": "C",
        "sigdial_claim": 3,
        "query": "How many jobs were submitted to Fugaku last month, and is that typical?",
        "intent_count": 1, "intent_types": ["sql"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": ["not_found"],
        "partial_available": False, "delegation_triggers": [],
        "coordination_chain": (
            "sql_agent(query for last month → 0 rows, dataset ends 2024-05) "
            "→ NOT_FOUND flag → synthesizer explains range"
        ),
        "checks": {
            "answer_min_length": 40,
            "must_contain": ["2024"],
            "must_not_contain": [],
        },
        "notes": (
            "Type C temporal boundary: running in 2026, dataset ends May 2024. "
            "Baseline may return 0 silently or invent a recent count. "
            "MAS must surface NOT_FOUND and explain the data range."
        ),
    },
    {
        "id": "Q16", "tier": "type_c", "query_type": "C",
        "sigdial_claim": 3,
        "query": (
            "I'm completely new to Fugaku and want to run a large 972-node "
            "compute-bound job for 8 hours. What's the failure probability, "
            "and how reliable is that estimate?"
        ),
        "intent_count": 1, "intent_types": ["predict"],
        "entry_agent": "pa_agent",
        "expected_agents": ["pa_agent", "synthesizer", "reflector"],
        "expected_reject": False, "expected_flags": ["low_sample", "confidence_low"],
        "partial_available": False, "delegation_triggers": [],
        "coordination_chain": (
            "pa_agent(predict nnumr=972 CB) → n_similar=1 → LOW_SAMPLE flag + "
            "no user history → CONFIDENCE_LOW flag → gateway adds dual caution footnotes"
        ),
        "checks": {
            "answer_min_length": 80,
            "must_contain": ["caution"],
            "must_not_contain": [],
        },
        "notes": (
            "Type C double uncertainty: nnumr=972 CB has only 1 historical job (LOW_SAMPLE) "
            "AND new user has no history (CONFIDENCE_LOW). Both flags must reach the answer. "
            "Key evidence that uncertainty flags survive the full pipeline."
        ),
    },
    {
        "id": "Q17", "tier": "type_c", "query_type": "C",
        "sigdial_claim": 3,
        "query": (
            "For our monthly ops report, I need the average GPU temperature per job, "
            "the billing cost per department, and the failure rate broken down by "
            "operating system."
        ),
        "intent_count": 3, "intent_types": ["sql_reject", "sql_reject", "sql_reject"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent"],
        "expected_reject": True, "expected_flags": [],
        "partial_available": False, "delegation_triggers": [],
        "coordination_chain": (
            "sql_agent decomposes 3 sub-questions → all 3 CANNOT_GENERATE "
            "(no temp/billing/OS columns) → full REJECT"
        ),
        "checks": {
            "answer_min_length": 60,
            "must_contain": [],
            "must_not_contain": ["temperature is", "°c", "billing is", "costs $", "windows", "linux had"],
        },
        "notes": (
            "CLAIM 3 full-rejection test: all 3 dimensions absent. "
            "Baseline typically fabricates all three. MAS must reject all and explain. "
            "Rejection explanation quality is itself a metric."
        ),
    },
    {
        "id": "Q18", "tier": "type_c", "query_type": "C",
        "sigdial_claim": 2,
        "query": (
            "I'm doing a historical analysis. How many jobs were submitted in "
            "January 2020, March 2020, and October 2020?"
        ),
        "intent_count": 1, "intent_types": ["sql"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": ["not_found"],
        "partial_available": False, "delegation_triggers": [],
        "coordination_chain": (
            "sql_agent(queries for 2020 dates) → all return 0 (dataset starts 2021-03) "
            "→ NOT_FOUND flag → synthesizer explains dataset boundary"
        ),
        "checks": {
            "answer_min_length": 40,
            "must_contain": ["2021"],
            "must_not_contain": [],
        },
        "notes": (
            "Type C full temporal pre-boundary: dataset starts March 2021, all three "
            "dates are before it. System must state this clearly, not return silent zeros."
        ),
    },
    {
        "id": "Q19", "tier": "type_c", "query_type": "C",
        "sigdial_claim": 2,
        "query": (
            "My 914-node compute-bound job failed last week. Looking at historical data, "
            "how common is that configuration and what does it predict for my next submission?"
        ),
        "intent_count": 2, "intent_types": ["sql", "predict"],
        "entry_agent": "pa_agent",
        "expected_agents": ["pa_agent", "sql_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": ["low_sample"],
        "partial_available": False, "delegation_triggers": ["data_insufficiency"],
        "coordination_chain": (
            "pa_agent → sql_agent(DATA_INSUFFICIENCY: count nnumr=914) "
            "→ n_similar=1 → LOW_SAMPLE flag → synthesizer with caution"
        ),
        "checks": {
            "answer_min_length": 80,
            "must_contain": ["caution"],
            "must_not_contain": [],
        },
        "notes": (
            "Type C sparse: nnumr=914 has only 1 historical job. "
            "Natural framing — user had an actual failure and is asking about recurrence. "
            "LOW_SAMPLE must propagate to final answer."
        ),
    },
    {
        "id": "Q20", "tier": "type_c", "query_type": "C",
        "sigdial_claim": 3,
        "query": (
            "What failure rates does the system show for jobs broken down by "
            "network topology (fat-tree vs dragonfly), CPU architecture (ARM vs x86), "
            "and user department?"
        ),
        "intent_count": 3, "intent_types": ["sql_reject", "sql_reject", "sql_reject"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent"],
        "expected_reject": True, "expected_flags": [],
        "partial_available": False, "delegation_triggers": [],
        "coordination_chain": (
            "sql_agent decomposes → 3× CANNOT_GENERATE (no topology/arch/dept columns) "
            "→ full REJECT with explanation"
        ),
        "checks": {
            "answer_min_length": 60,
            "must_contain": [],
            "must_not_contain": ["fat-tree had", "dragonfly had", "arm failure", "x86 failure", "department had"],
        },
        "notes": (
            "Type C adversarial: all three dimensions sound plausible for an HPC system "
            "but none exist in the Fugaku schema. Tests breadth of hallucination prevention."
        ),
    },

    # ══════════════════════════════════════════════════════════════════════════
    # BREADTH  —  Q21–Q50
    # Natural HPC operator queries. Mixed intents, realistic scenarios.
    # Core-3 checks only (route + flags + reject).
    # ══════════════════════════════════════════════════════════════════════════

    # ── Single-intent breadth (Q21–Q27) ──────────────────────────────────────

    {
        "id": "Q21", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": "How many unique users have submitted jobs to Fugaku across the whole dataset?",
        "intent_count": 1, "intent_types": ["sql"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": [],
        "coordination_chain": "sql_agent(COUNT DISTINCT usr) → synthesizer",
        "checks": {"answer_min_length": 20, "must_contain": ["3,457"], "must_not_contain": []},
        "notes": "Simple distinct count. Ground truth: 3,457 unique users.",
    },
    {
        "id": "Q22", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": "What are the top 5 most common job names submitted by usr_1898?",
        "intent_count": 1, "intent_types": ["sql"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": [],
        "coordination_chain": "sql_agent(GROUP BY jnam WHERE usr=usr_1898 LIMIT 5) → synthesizer",
        "checks": {"answer_min_length": 30, "must_contain": [], "must_not_contain": []},
        "notes": "Top-N query for heaviest user. usr_1898 has 1.25M jobs.",
    },
    {
        "id": "Q23", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": "What does the pjdel command do and when should I use it?",
        "intent_count": 1, "intent_types": ["doc"],
        "entry_agent": "doc_agent",
        "expected_agents": ["doc_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": [],
        "coordination_chain": "doc_agent(RAG: pjdel) → synthesizer",
        "checks": {"answer_min_length": 60, "must_contain": ["pjdel"], "must_not_contain": []},
        "notes": "Basic doc retrieval. pjdel is a well-documented Fugaku command.",
    },
    {
        "id": "Q24", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": (
            "What is the failure risk for a 32-node compute-bound job with "
            "2-hour walltime submitted by user usr_1898?"
        ),
        "intent_count": 1, "intent_types": ["predict"],
        "entry_agent": "pa_agent",
        "expected_agents": ["pa_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": [],
        "coordination_chain": "pa_agent(predict, known user usr_1898) → synthesizer",
        "checks": {"answer_min_length": 60, "must_contain": ["fail"], "must_not_contain": []},
        "notes": "Prediction with known high-volume user. No CONFIDENCE_LOW expected.",
    },
    {
        "id": "Q25", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": "What is the average wait time between job submission and start in 2023?",
        "intent_count": 1, "intent_types": ["sql"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": [],
        "coordination_chain": "sql_agent(AVG wait time for 2023 jobs) → synthesizer",
        "checks": {"answer_min_length": 30, "must_contain": [], "must_not_contain": []},
        "notes": "Wait time derived metric: EPOCH(sdt) - EPOCH(qdt).",
    },
    {
        "id": "Q26", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": "How do I compile and run an OpenMP Fortran job on Fugaku?",
        "intent_count": 1, "intent_types": ["doc"],
        "entry_agent": "doc_agent",
        "expected_agents": ["doc_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": [],
        "coordination_chain": "doc_agent(RAG: OpenMP Fortran compilation) → synthesizer",
        "checks": {"answer_min_length": 60, "must_contain": [], "must_not_contain": []},
        "notes": "Compilation how-to from documentation.",
    },
    {
        "id": "Q27", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": "How many jobs ran at boosted frequency (freq_req not equal to 2000 MHz)?",
        "intent_count": 1, "intent_types": ["sql"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": [],
        "coordination_chain": "sql_agent(COUNT WHERE freq_req != 2000) → synthesizer",
        "checks": {"answer_min_length": 20, "must_contain": [], "must_not_contain": []},
        "notes": "freq_req has 2 values: 2000 MHz (normal) and other (boosted).",
    },

    # ── Dual-intent breadth (Q28–Q36) ────────────────────────────────────────

    {
        "id": "Q28", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": (
            "My job got stuck — how do I delete it, and separately, "
            "what's the overall failure rate for compute-bound jobs so I know "
            "if this is a one-off or a pattern?"
        ),
        "intent_count": 2, "intent_types": ["doc", "sql"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "doc_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": ["knowledge_gap"],
        "coordination_chain": "sql_agent(CB fail rate) → doc_agent(KNOWLEDGE_GAP: pjdel) → synthesizer",
        "checks": {"answer_min_length": 80, "must_contain": ["pjdel"], "must_not_contain": []},
        "notes": "Operator troubleshooting pattern: stats + action. Natural mixed query.",
    },
    {
        "id": "Q29", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": (
            "What's the average job duration for compute-bound jobs, and based "
            "on that, predict the failure risk if I submit a job with that typical duration "
            "and 64 nodes?"
        ),
        "intent_count": 2, "intent_types": ["sql", "predict"],
        "entry_agent": "pa_agent",
        "expected_agents": ["pa_agent", "sql_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": ["data_insufficiency"],
        "coordination_chain": (
            "pa_agent → sql_agent(DATA_INSUFFICIENCY: avg CB duration ~13602s) "
            "→ pa_agent(predict nnumr=64, elpl≈13602) → synthesizer"
        ),
        "checks": {"answer_min_length": 80, "must_contain": ["fail"], "must_not_contain": []},
        "notes": "SQL-derived walltime feeds into prediction. avg_cb_dur ≈ 13,602s ≈ 3.8h.",
    },
    {
        "id": "Q30", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": (
            "How many compute-bound jobs ran with more than 512 nodes, "
            "and what does the pjshowrsc command show for large job scheduling?"
        ),
        "intent_count": 2, "intent_types": ["sql", "doc"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "doc_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": ["knowledge_gap"],
        "coordination_chain": "sql_agent(COUNT CB nnumr>512) → doc_agent(KNOWLEDGE_GAP: pjshowrsc) → synthesizer",
        "checks": {"answer_min_length": 80, "must_contain": ["pjshowrsc"], "must_not_contain": []},
        "notes": "SQL large-job count + doc for the relevant monitoring command.",
    },
    {
        "id": "Q31", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": (
            "What is the failure rate for 128-node memory-bound jobs, and given that "
            "rate, should I be worried about submitting a similar job?"
        ),
        "intent_count": 2, "intent_types": ["sql", "predict"],
        "entry_agent": "pa_agent",
        "expected_agents": ["pa_agent", "sql_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": ["data_insufficiency"],
        "coordination_chain": (
            "pa_agent → sql_agent(DATA_INSUFFICIENCY: nnumr=128 MB fail rate) "
            "→ prediction enriched with historical context → synthesizer"
        ),
        "checks": {"answer_min_length": 80, "must_contain": [], "must_not_contain": []},
        "notes": "128-node MB: 5,938/55,260 = 10.74% fail rate. Well-sampled config.",
    },
    {
        "id": "Q32", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": (
            "How many jobs ran for more than 24 hours? "
            "What happens to a job that exceeds its walltime limit on Fugaku?"
        ),
        "intent_count": 2, "intent_types": ["sql", "doc"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "doc_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": ["knowledge_gap"],
        "coordination_chain": "sql_agent(COUNT duration>86400 → 335,252) → doc_agent(KNOWLEDGE_GAP: walltime behavior) → synthesizer",
        "checks": {"answer_min_length": 80, "must_contain": ["335,252"], "must_not_contain": []},
        "notes": "335,252 jobs exceeded 24h. Doc explains what happens at walltime limit.",
    },
    {
        "id": "Q33", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": (
            "Predict the failure risk for a 48-node memory-bound job with "
            "3-hour walltime. Also, what pjsub directives should I include "
            "for a memory-bound job?"
        ),
        "intent_count": 2, "intent_types": ["predict", "doc"],
        "entry_agent": "pa_agent",
        "expected_agents": ["pa_agent", "doc_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": ["knowledge_gap"],
        "coordination_chain": "pa_agent(predict 48-node MB) → doc_agent(KNOWLEDGE_GAP: pjsub MB directives) → synthesizer",
        "checks": {"answer_min_length": 80, "must_contain": ["pjsub", "fail"], "must_not_contain": []},
        "notes": "Prediction + job script setup. Natural new-user combined query.",
    },
    {
        "id": "Q34", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": (
            "How many jobs used the jobenv_req_2 environment type, "
            "and what does using that environment type imply for job execution?"
        ),
        "intent_count": 2, "intent_types": ["sql", "doc"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "doc_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": ["knowledge_gap"],
        "coordination_chain": "sql_agent(COUNT jobenv_req_2 → 5) → doc_agent(KNOWLEDGE_GAP: what jobenv_req_2 means) → synthesizer",
        "checks": {
            "answer_min_length": 60,
            "must_contain": ["5"],
            "must_not_contain": ["no data", "not found"],
        },
        "notes": "Sparse but real (5 jobs). Must return 5, not NOT_FOUND. Doc explains the env type.",
    },
    {
        "id": "Q35", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": (
            "What is the average duration and failure rate for memory-bound jobs? "
            "What operating system did most users submit from?"
        ),
        "intent_count": 2, "intent_types": ["sql", "sql_reject"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": ["partially_found"],
        "partial_available": True, "delegation_triggers": [],
        "coordination_chain": "sql_agent decomposes → MB stats ✓ + OS ✗ → PARTIALLY_FOUND → synthesizer",
        "checks": {
            "answer_min_length": 60,
            "must_contain": ["memory-bound"],
            "must_not_contain": ["windows", "linux had", "macos", "os is"],
        },
        "notes": "Type B breadth: partial answer. OS column absent — must not fabricate OS values.",
    },
    {
        "id": "Q36", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": (
            "Predict failure risk for a 64-node memory-bound job with 4-hour walltime "
            "for user usr_1898, and check whether their historical failure rate "
            "matches the model's prediction."
        ),
        "intent_count": 2, "intent_types": ["predict", "sql"],
        "entry_agent": "pa_agent",
        "expected_agents": ["pa_agent", "sql_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": ["data_insufficiency"],
        "coordination_chain": "pa_agent(predict usr_1898, known user → no CONFIDENCE_LOW) → sql_agent(DATA_INSUFFICIENCY: usr_1898 fail rate) → synthesizer",
        "checks": {"answer_min_length": 80, "must_contain": ["fail"], "must_not_contain": []},
        "notes": "Known power user (usr_1898). Both prediction and historical comparison. No uncertainty flags expected.",
    },

    # ── Triple-intent breadth (Q37–Q44) ──────────────────────────────────────

    {
        "id": "Q37", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": (
            "For a 128-node compute-bound job with 6-hour walltime: predict the failure risk, "
            "check what the historical data says about that config, "
            "and give me the pjsub directives I need."
        ),
        "intent_count": 3, "intent_types": ["predict", "sql", "doc"],
        "entry_agent": "pa_agent",
        "expected_agents": ["pa_agent", "sql_agent", "doc_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": ["data_insufficiency", "knowledge_gap"],
        "coordination_chain": "pa_agent(predict) → sql_agent(DATA_INSUFFICIENCY: historical) + doc_agent(KNOWLEDGE_GAP: pjsub) → synthesizer",
        "checks": {"answer_min_length": 60, "must_contain": [], "must_not_contain": []},
        "notes": "Full PA orchestration: predict + compare + directives. All three delegation paths.",
    },
    {
        "id": "Q38", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": (
            "What is the failure rate per job class? "
            "How many jobs exceeded 24 hours of runtime? "
            "What does pjdel do for jobs that run too long?"
        ),
        "intent_count": 3, "intent_types": ["sql", "sql", "doc"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "doc_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": ["knowledge_gap"],
        "coordination_chain": "sql_agent(2 SQL) → doc_agent(KNOWLEDGE_GAP: pjdel) → synthesizer",
        "checks": {"answer_min_length": 60, "must_contain": [], "must_not_contain": []},
        "notes": "Two SQL queries + doc delegation. The 24h count (335,252) and fail rate by pclass.",
    },
    {
        "id": "Q39", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": (
            "My team runs 64–256 node memory-bound jobs. What's the overall failure rate "
            "for that node range? What config should a new team member use for their "
            "first job? And where do I find the job submission guidelines?"
        ),
        "intent_count": 3, "intent_types": ["sql", "predict", "doc"],
        "entry_agent": "pa_agent",
        "expected_agents": ["pa_agent", "sql_agent", "doc_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": ["confidence_low"],
        "partial_available": False, "delegation_triggers": ["data_insufficiency", "knowledge_gap"],
        "coordination_chain": "pa_agent → sql_agent(stats for new user context) + doc_agent(guidelines) → CONFIDENCE_LOW (new user) → synthesizer",
        "checks": {"answer_min_length": 60, "must_contain": [], "must_not_contain": []},
        "notes": "New team member → CONFIDENCE_LOW. Three-way coordination: SQL gives context, PA predicts, doc gives guidelines.",
    },
    {
        "id": "Q40", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": (
            "I'm reviewing last year's performance: what was the average walltime "
            "for failed compute-bound jobs, how many ran over 12 hours, "
            "and what does the pjstat command show for monitoring?"
        ),
        "intent_count": 3, "intent_types": ["sql", "sql", "doc"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "doc_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": ["knowledge_gap"],
        "coordination_chain": "sql_agent(2 SQL queries) → doc_agent(KNOWLEDGE_GAP: pjstat) → synthesizer",
        "checks": {"answer_min_length": 60, "must_contain": [], "must_not_contain": []},
        "notes": "Annual review query. avg_elpl of failed CB ≈ 9.3h; over-12h count = 1,603,893.",
    },
    {
        "id": "Q41", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": (
            "What are the top 3 users by total energy consumption? "
            "What is the average energy for compute-bound vs memory-bound jobs? "
            "How can I estimate my own job's energy footprint from the docs?"
        ),
        "intent_count": 3, "intent_types": ["sql", "sql", "doc"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "doc_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": ["knowledge_gap"],
        "coordination_chain": "sql_agent(2 SQL: top users + avg econ by pclass) → doc_agent(KNOWLEDGE_GAP: energy estimation) → synthesizer",
        "checks": {"answer_min_length": 60, "must_contain": [], "must_not_contain": []},
        "notes": "Energy-focused three-intent query. Two SQL aggregations + doc guidance.",
    },
    {
        "id": "Q42", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": (
            "For a 96-node compute-bound job with 6-hour walltime: "
            "predict the risk, compare against historical failure rates, "
            "and tell me the cancellation procedure if I need to stop it early."
        ),
        "intent_count": 3, "intent_types": ["predict", "sql", "doc"],
        "entry_agent": "pa_agent",
        "expected_agents": ["pa_agent", "sql_agent", "doc_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": ["data_insufficiency", "knowledge_gap"],
        "coordination_chain": "pa_agent(predict) → sql_agent(DATA_INSUFFICIENCY: historical) + doc_agent(KNOWLEDGE_GAP: cancellation) → synthesizer",
        "checks": {"answer_min_length": 60, "must_contain": [], "must_not_contain": []},
        "notes": "Full three-way: predict + historical check + ops procedure. Natural pre-submission query.",
    },
    {
        "id": "Q43", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": (
            "What was the failure rate for compute-bound jobs in 2022 versus 2023 — "
            "has it changed? Based on that trend, predict the risk for a typical "
            "compute-bound job today, and explain what changed in the system."
        ),
        "intent_count": 3, "intent_types": ["sql", "predict", "doc"],
        "entry_agent": "pa_agent",
        "expected_agents": ["pa_agent", "sql_agent", "doc_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": ["data_insufficiency", "knowledge_gap"],
        "coordination_chain": "pa_agent → sql_agent(DATA_INSUFFICIENCY: 2022/2023 fail rate trend) → pa_agent(predict enriched with trend) → doc_agent(KNOWLEDGE_GAP: system changes) → synthesizer",
        "checks": {"answer_min_length": 60, "must_contain": [], "must_not_contain": []},
        "notes": "Trend analysis feeding prediction. SQL trend → PA enrichment → doc context.",
    },
    {
        "id": "Q44", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": (
            "I'm debugging repeated failures in my compute-bound jobs. "
            "Look up the failure rate for 64-node jobs, predict my risk with a fresh "
            "128-node submission, and tell me the diagnostic commands I should run."
        ),
        "intent_count": 3, "intent_types": ["sql", "predict", "doc"],
        "entry_agent": "pa_agent",
        "expected_agents": ["pa_agent", "sql_agent", "doc_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": ["data_insufficiency", "knowledge_gap"],
        "coordination_chain": "pa_agent → sql_agent(DATA_INSUFFICIENCY: 64-node CB fail rate) → pa_agent(predict 128-node) → doc_agent(KNOWLEDGE_GAP: diagnostic commands) → synthesizer",
        "checks": {"answer_min_length": 60, "must_contain": [], "must_not_contain": []},
        "notes": "Debugging scenario. Historical context feeds new prediction + doc for ops procedure.",
    },

    # ── Higher-intent / edge breadth (Q45–Q50) ───────────────────────────────

    {
        "id": "Q45", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": (
            "Full system overview: total job count, failure rate by pclass, "
            "average energy per class, top 3 users by jobs, and max node count ever used."
        ),
        "intent_count": 5, "intent_types": ["sql"] * 5,
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": [],
        "coordination_chain": "sql_agent(5 SQL decomposed) → synthesizer",
        "checks": {"answer_min_length": 50, "must_contain": [], "must_not_contain": []},
        "notes": "5-intent pure SQL breadth. Tests multi-SQL decomposition.",
    },
    {
        "id": "Q46", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": (
            "Predict failure risk for three jobs: "
            "a 64-node compute-bound job with 4-hour walltime, "
            "a 128-node memory-bound job with 6-hour walltime, and "
            "a 972-node compute-bound job with 8-hour walltime."
        ),
        "intent_count": 3, "intent_types": ["predict", "predict", "predict"],
        "entry_agent": "pa_agent",
        "expected_agents": ["pa_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": ["low_sample"],
        "partial_available": False, "delegation_triggers": [],
        "coordination_chain": "pa_agent extracts 3 specs → predicts all → 972-node triggers LOW_SAMPLE → synthesizer",
        "checks": {"answer_min_length": 50, "must_contain": [], "must_not_contain": []},
        "notes": "Multi-spec prediction. 64 and 128 node are reliable; 972-node triggers LOW_SAMPLE. Contrast in one answer.",
    },
    {
        "id": "Q47", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": "What is the risk of running a large job on Fugaku?",
        "intent_count": 1, "intent_types": ["doc"],
        "entry_agent": "doc_agent",
        "expected_agents": ["doc_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": [],
        "partial_available": False, "delegation_triggers": [],
        "coordination_chain": "doc_agent(RAG: large job risks/policies) → synthesizer",
        "checks": {"answer_min_length": 40, "must_contain": [], "must_not_contain": []},
        "notes": (
            "Ambiguous: 'risk' near predict/doc boundary. No specific job spec given "
            "→ gateway must classify as doc (guidance), NOT predict. "
            "Tests classification boundary robustness."
        ),
    },
    {
        "id": "Q48", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": (
            "What is the failure rate per pclass? How many jobs ran over 24 hours? "
            "What is the average GPU memory usage per job? What does pjsub do? "
            "Predict failure risk for a 64-node compute-bound job with 4-hour walltime."
        ),
        "intent_count": 5, "intent_types": ["sql", "sql", "sql_reject", "doc", "predict"],
        "entry_agent": "pa_agent",
        "expected_agents": ["pa_agent", "sql_agent", "doc_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": ["partially_found"],
        "partial_available": True, "delegation_triggers": ["data_insufficiency", "knowledge_gap"],
        "coordination_chain": "pa_agent → sql_agent(3 SQL: fail rate ✓, >24h ✓, GPU mem ✗) + doc_agent(pjsub) → PARTIALLY_FOUND → synthesizer",
        "checks": {"answer_min_length": 50, "must_contain": [], "must_not_contain": []},
        "notes": "5-intent with all types + GPU absent → PARTIALLY_FOUND. Tests flag propagation in complex chain.",
    },
    {
        "id": "Q49", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": (
            "How many total jobs ran? Average energy per pclass? Max nodes ever used? "
            "Average GPU usage per job? OS breakdown? Billing cost per department? "
            "Failure rate by network topology?"
        ),
        "intent_count": 7, "intent_types": ["sql", "sql", "sql", "sql_reject", "sql_reject", "sql_reject", "sql_reject"],
        "entry_agent": "sql_agent",
        "expected_agents": ["sql_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": ["partially_found"],
        "partial_available": True, "delegation_triggers": [],
        "coordination_chain": "sql_agent(7 SQL: 3 succeed + 4 absent) → PARTIALLY_FOUND → synthesizer",
        "checks": {"answer_min_length": 50, "must_contain": [], "must_not_contain": []},
        "notes": "7-intent stress test: 3 answerable, 4 absent. PARTIALLY_FOUND at 7-intent level.",
    },
    {
        "id": "Q50", "tier": "breadth", "query_type": None,
        "sigdial_claim": None,
        "query": (
            "Give me a complete picture of this system: the job failure statistics, "
            "energy trends, a prediction for a typical new-user job, "
            "and be explicit about what you cannot tell me."
        ),
        "intent_count": 3, "intent_types": ["sql", "sql", "predict"],
        "entry_agent": "pa_agent",
        "expected_agents": ["pa_agent", "sql_agent", "synthesizer"],
        "expected_reject": False, "expected_flags": ["confidence_low"],
        "partial_available": False, "delegation_triggers": ["data_insufficiency"],
        "coordination_chain": "pa_agent → sql_agent(DATA_INSUFFICIENCY: stats + energy) → pa_agent(predict, unknown user → CONFIDENCE_LOW) → synthesizer",
        "checks": {"answer_min_length": 80, "must_contain": [], "must_not_contain": []},
        "notes": (
            "Open-ended overview query with explicit 'tell me what you cannot answer'. "
            "Tests whether the system is self-aware about its limitations. "
            "CONFIDENCE_LOW (new user) must reach the final answer."
        ),
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_by_id(qid: str) -> dict | None:
    return next((q for q in QUERY_BANK if q["id"] == qid), None)


def get_by_tier(tier: str) -> list[dict]:
    return [q for q in QUERY_BANK if q["tier"] == tier]


def get_by_type(query_type: str) -> list[dict]:
    return [q for q in QUERY_BANK if q["query_type"] == query_type]


def get_by_claim(claim: int) -> list[dict]:
    return [q for q in QUERY_BANK if q.get("sigdial_claim") == claim]


TIERS = ["type_a", "type_b", "type_c", "breadth"]

TIER_LABELS = {
    "type_a":  "Type A  (Sanity)  ",
    "type_b":  "Type B  (Partial) ",
    "type_c":  "Type C  (Unansw.) ",
    "breadth": "Breadth (Mixed)   ",
}

STRICT_TIERS = {"type_a", "type_b", "type_c"}
