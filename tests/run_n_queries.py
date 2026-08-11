"""
Fast runner for N1-N5 SIGDIAL new queries across MAS, Blackboard, Unstructured.

Usage:
    cd /path/to/repo
    # MAS (no Qdrant needed)
    python3 tests/run_n_queries.py mas

    # Baselines (Qdrant needed)
    CUDA_VISIBLE_DEVICES="" QDRANT_PATH=/tmp/sai_qdrant_db python3 tests/run_n_queries.py blackboard
    CUDA_VISIBLE_DEVICES="" QDRANT_PATH=/tmp/sai_qdrant_db python3 tests/run_n_queries.py unstructured

Speed notes:
    - Agent is initialised ONCE before any queries run (models load once, not per query).
    - Qdrant lock is cleared once at startup only; the same client is reused across all queries.
    - No sentence-transformer pre-warming (avoids memory pressure that caused the 5-min hang).
"""

import asyncio, os, sys, time, textwrap

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PR   = os.path.join(_ROOT, "research")
_SA   = os.path.join(_PR,   "single_agent_baseline")
_SH   = os.path.join(_PR,   "shared")
_AN   = os.path.join(_ROOT, "analytics")
for _p in (_ROOT, _PR, _SA, _SH, _AN):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("QDRANT_PATH", "/tmp/sai_qdrant_db")


def _patch_rag_with_bm25():
    """
    Replace the Qdrant-backed rag_search in tools.py with the MAS BM25+numpy
    retriever (shared/doc_retriever.py).  Call this BEFORE importing any
    baseline agent so the patch is in place when the agent is constructed.

    Benefits:
    - No Qdrant process / lock file / sentence-transformer download
    - Same document collection as MAS doc_agent → fair comparison
    - Loads in < 1 s (embeddings cached in .npy)
    """
    _shared = os.path.join(_ROOT, "shared")
    if _shared not in sys.path:
        sys.path.insert(0, _shared)

    import tools                         # already on sys.path via _SA
    sys.path.insert(0, _ROOT)            # ensure 'shared' package visible
    from shared.doc_retriever import DocRetriever  # MAS retriever

    _retriever = DocRetriever()          # loads BM25 + .npy cache — fast

    def _bm25_rag_search(query: str) -> str:
        chunks = _retriever.search(query, top_k=5)
        if not chunks:
            return "No relevant documentation found."
        parts = []
        for i, c in enumerate(chunks, 1):
            breadcrumb = c.get("breadcrumb", c.get("source", "Fugaku docs"))
            parts.append(f"[{i}] {breadcrumb}\n{c['text']}")
        return "\n\n".join(parts)

    tools.rag_search = _bm25_rag_search  # hot-swap

# ── Queries ───────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# BATCH 2 — N6–N10  (natural HPC-user phrasing; doc + PA ground truth verified)
# ─────────────────────────────────────────────────────────────────────────────
# Ground truth quick-reference (all DB-verified 2026-05-26):
#   N6  MB >12h: n=2,884,742  fail=12.9%  avg_nnumr=33.88
#       nnumr=34 MB: n=112  fail=32.14%  → LOW_SAMPLE
#       doc: rscgrp=small (≤384 nodes, max 72h batch)
#   N7  usr_2111: n=824,021  fail=2.26%  avg_nnumr=27  MB-dominant
#       usr_1898: n=1,252,185  fail=45.31%  (~20× worse)
#       nnumr=27 MB global: n=3,398  fail=10.65%
#   N8  CB fail by year: 2021=6.11%  2022=7.15%  2023=17.99%  2024=8.33%
#       2023 CB by quarter: Q1=21.2%  Q2=19.78%  Q3=8.66%  Q4=18.71%
#       root cause of 2023 spike → NOT IN DATA  (REJECT expected from MAS)
#       doc: pjstat -v shows REASON=ELAPSE LIMIT EXCEEDED; ST=EXT on exit
#   N9  MB avg econ=12,030J (p50=41J, max=178M J — heavily skewed)
#       CB avg econ=4,247J (p50=231J)
#       billing cost per node-hour → NOT IN DATA  (hallucination trap)
#       doc: #PJM -S directive + pjstat -S --history for power/energy stats
#   N10 nnumr=192 MB global: n=153,388  fail=5.34%
#       usr_2111 nnumr=192 MB: n=83,513  fail=1.49%  avg_elpl=3600s
#       doc: 192 nodes → rscgrp=small (≤384)  max elapse=72h
#       doc: pjsub directives: node=192, elapse=08:00:00, rscgrp=small
# ─────────────────────────────────────────────────────────────────────────────
BATCH2_QUERIES = [
    {
        "id":    "N6",
        "claim": "4+2",
        "label": "Long-run MB: SQL→PA(LOW_SAMPLE) + Doc rscgrp grounding [Claim 4+2]",
        "query": (
            "I run genomics workflows that need more than 12 hours of walltime and are "
            "memory-bound. What is the historical failure rate for memory-bound jobs on "
            "Fugaku that ran longer than 12 hours, and what is the typical node count for "
            "such jobs? Based on those historical averages, if I submit a new 34-node "
            "memory-bound job with exactly 12-hour walltime, what failure risk should I "
            "expect? And which resource group (rscgrp) should I request according to the "
            "Fugaku manual for a job at this node count and duration?"
        ),
    },
    {
        "id":    "N7",
        "claim": "1",
        "label": "Two-user comparison + PA prediction for reliable user [Claim 1]",
        "query": (
            "Two of the most active users on Fugaku are usr_2111 and usr_1898. Compare "
            "their overall job failure rates and typical job sizes. Which user is more "
            "reliable, and by how much? If usr_2111 submits a new memory-bound job with "
            "27 nodes and a 30-minute walltime, what failure risk does the system predict "
            "for them?"
        ),
    },
    {
        "id":    "N8",
        "claim": "3+1",
        "label": "Year-over-year reliability trend + 2023 spike + pjstat doc [Claim 3+1]",
        "query": (
            "Has Fugaku's job reliability been improving or degrading over time? Break down "
            "the overall failure rate by year from 2021 through 2024, and do the same "
            "specifically for compute-bound jobs. Was there a particularly bad year, and "
            "what might explain the spike? Also, when a job is killed because it exceeded "
            "its walltime limit, what status code and reason message does pjstat display?"
        ),
    },
    {
        "id":    "N9",
        "claim": "2+3",
        "label": "Energy footprint + billing-cost hallucination trap + pjstat -S doc [Claim 2+3]",
        "query": (
            "My PI is asking for a sustainability report on our Fugaku usage. What is the "
            "average energy consumption per job for memory-bound versus compute-bound jobs? "
            "What does that work out to in billing cost per node-hour so we can estimate "
            "our research budget? And what pjstat option or pjsub directive should I use "
            "to monitor or record the actual power consumption of my jobs?"
        ),
    },
    {
        "id":    "N10",
        "claim": "1+2",
        "label": "Full 3-agent coordination: 192-node MB production run [Claim 1+2]",
        "query": (
            "I am usr_2111 and I am about to submit a large production run: 192 nodes, "
            "memory-bound, 8-hour walltime. Before I submit: what is the global failure "
            "rate for memory-bound jobs at exactly 192 nodes on Fugaku, and how does my "
            "personal track record at that node count compare to the system average? "
            "What failure risk does the system predict for this specific job? And what "
            "pjsub directives — including the correct resource group — should I include "
            "in my job script for a 192-node, 8-hour job?"
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# BATCH 3 — N11–N15  (mixed multi-intent queries; varied dependency patterns)
# ─────────────────────────────────────────────────────────────────────────────
# Ground truth (DB-verified 2026-05-26):
#   N11  MB 96-node: n=48,616  fail=9.42%  avg_elpl(all)=28,996s  avg_elpl(failed)=60,325s
#        rscgrp=small (≤384 nodes, max 72h)
#        Pattern: a→b AND a→c  (SQL forks into PA via DATA_INSUFFICIENCY + Doc)
#
#   N12  usr_2111 by class:  CB  n=21,349 fail=0.65% avg_nnumr=3.36
#                             MB  n=802,672 fail=2.30% avg_nnumr=27.25
#        usr_1898 by class:  CB  n=1,211,531 fail=45.00% avg_nnumr=1.42
#                             MB  n=40,654   fail=54.39% avg_nnumr=1.02
#        usr_2111 avg MB nnumr=27.25  →  PA target for usr_1898 27-node MB 4h
#        Pattern: (a AND b)→c→d  (two-user SQL join into PA, PA leads to Doc)
#
#   N13  2023 failed CB by walltime bucket:
#          <2h: n=186,165  avg_nnumr=8.86   avg_elpl=1,360s
#          2-8h: n=31,885  avg_nnumr=192.59 avg_elpl=12,225s  ← HIGH-NODE anomaly
#          >8h:  n=399,257 avg_nnumr=5.27   avg_elpl=50,799s  ← DOMINANT bucket
#        For (c) PA: 5-node CB ~14.1h (>8h dominant bucket params) new user → CONFIDENCE_LOW
#        Pattern: a→b→c→d  (4-hop linear chain, each step needs previous output)
#
#   N14  CB 64-node: n=23,198  fail=25.93%
#        MB 768-node: n=22,134  fail=4.43%
#        64-node rscgrp=small; 768-node rscgrp boundary (>384 nodes)
#        Pattern: (a→b) AND (c→d) AND e  (two independent parallel 2-hop chains + Doc)
#
#   N15  usr_2111 MB avg econ=1,298J (global MB avg=12,030J — usr_2111 much lower)
#        billing cost per node-hour → NOT IN DATA  (hallucination trap)
#        pjstat -S → correct command for post-job energy retrieval
#        Pattern: a→b AND a→d(TRAP) AND c(independent)
# ─────────────────────────────────────────────────────────────────────────────
BATCH3_QUERIES = [
    {
        "id":    "N11",
        "claim": "4+1",
        "label": "Fork a→(PA+Doc): SQL forks into PA chain and parallel Doc grounding",
        "query": (
            "I lead a computational biology group on Fugaku (usr_2111) and we are scaling "
            "from small test jobs to 96-node memory-bound production runs. Before we commit, "
            "I need three things: (a) What is the historical failure rate for memory-bound "
            "jobs at exactly 96 nodes, and what is the average walltime for those jobs? "
            "(b) Using the historical average walltime from (a) as the planned walltime for "
            "a new 96-node memory-bound job by usr_2111, what failure probability does the "
            "system predict? (c) According to the Fugaku submission guide, which resource "
            "group (rscgrp) applies to 96-node jobs and what is the maximum allowed walltime "
            "for that group?"
        ),
    },
    {
        "id":    "N12",
        "claim": "4+1+2",
        "label": "Join (a AND b)→c→d: two-user SQL join feeds PA, PA leads to Doc",
        "query": (
            "I am preparing a reliability comparison for two of Fugaku's most active users. "
            "(a) For usr_2111: what is their job failure rate broken down by class "
            "(compute-bound and memory-bound separately), and what is their average job size "
            "(nnumr) for memory-bound jobs? (b) For usr_1898: the same breakdown — failure "
            "rate by class and average memory-bound job size. (c) usr_1898 wants to run a "
            "memory-bound job at usr_2111's typical memory-bound node count (use the exact "
            "average from (a)): predict the failure risk for usr_1898 submitting a "
            "memory-bound job at that node count with a 4-hour walltime. (d) For the node "
            "count identified in (a), which rscgrp and what scheduling options does the "
            "Fugaku documentation specify?"
        ),
    },
    {
        "id":    "N13",
        "claim": "4+1+2",
        "label": "4-hop linear chain a→b→c→d: 2023 CB bucket→avg→PA→Doc",
        "query": (
            "I want to understand Fugaku's worst compute-bound failure pattern in 2023 "
            "and plan a new job accordingly. (a) For compute-bound jobs that failed in 2023, "
            "how many fell into each walltime bucket: under 2 hours, 2–8 hours, and over "
            "8 hours? Report job counts. (b) For the single largest failure bucket from (a), "
            "what were the average node count and average requested walltime of those failed "
            "jobs? (c) Using the exact averages from (b) as job parameters for a new "
            "compute-bound job submitted by a first-time Fugaku user (usr_newresearcher), "
            "predict the failure probability — and flag any data-confidence issues. "
            "(d) Based on the node count from (b) and the risk level from (c), which pjsub "
            "directives — including the correct rscgrp — and which pjstat command for "
            "monitoring job status does the Fugaku manual recommend?"
        ),
    },
    {
        "id":    "N14",
        "claim": "4+1+2",
        "label": "Parallel chains (a→b) AND (c→d) AND e: two jobs, two SQL→PA chains",
        "query": (
            "I have two very different jobs to submit this week on Fugaku and need risk "
            "assessments for both before queuing. Job Alpha: 64-node compute-bound, 8-hour "
            "walltime, account usr_2111. Job Beta: 768-node memory-bound, 24-hour walltime, "
            "brand-new project account usr_proj999 (no prior Fugaku history). Please: "
            "(a) What is the historical failure rate in the Fugaku database for compute-bound "
            "jobs at exactly 64 nodes? (b) Using that statistic, what failure risk does the "
            "system predict for Job Alpha under usr_2111? (c) What is the historical failure "
            "rate in the database for memory-bound jobs at exactly 768 nodes? "
            "(d) Using those statistics and noting that usr_proj999 has zero job history, "
            "predict the failure risk for Job Beta — explicitly flag any confidence "
            "limitations. (e) What is the correct rscgrp directive for each job?"
        ),
    },
    {
        "id":    "N15",
        "claim": "2+3+1",
        "label": "Fork+trap a→(PA,Doc,REJECT): SQL→PA chain + billing hallucination trap",
        "query": (
            "Our research group is auditing Fugaku energy usage for a grant report. "
            "(a) For jobs submitted by usr_2111, what is the average energy consumption "
            "(econ) for their memory-bound jobs, and how does that compare to the global "
            "average energy consumption for memory-bound jobs across all users? "
            "(b) Based on usr_2111's personal energy profile from (a), what does the "
            "predictor estimate for energy consumption and failure risk on their next "
            "memory-bound job: 48 nodes, 6-hour walltime? (c) According to Fugaku "
            "documentation, which pjstat option or pjsub directive lets us retrieve the "
            "actual energy used after a job finishes? (d) Given the energy figures from (a), "
            "what is the electricity billing cost in yen per node-hour for memory-bound jobs "
            "on Fugaku — we need this for our grant budget."
        ),
    },
]

# combined for convenience
ALL_QUERIES = []  # populated at bottom of file

NEW_QUERIES = [
    {
        "id":    "N1",
        "claim": "4+1",
        "label": "3-hop chain: SQL→PA(DATA_INSUFFICIENCY)→Doc(KNOWLEDGE_GAP)",
        "query": (
            "Among compute-bound jobs that failed in 2023, what was the average node count "
            "and average walltime? Use those exact historical averages — not defaults, not "
            "guesses — as the parameters for a new compute-bound job and predict its failure "
            "risk. Finally, based on that node count, what pjsub directives should I specify "
            "to configure the job correctly?"
        ),
    },
    {
        "id":    "N2",
        "claim": "2+3+1",
        "label": "PARTIALLY_FOUND + CPU-temp hallucination trap + dependent comparison",
        "query": (
            "For user usr_1898 on Fugaku: (a) What is their overall job failure rate across "
            "all their submitted jobs? (b) What was the average CPU temperature recorded "
            "during their failed jobs? (c) Compared to the system-wide average failure rate, "
            "is usr_1898 above or below average — and by how much?"
        ),
    },
    {
        "id":    "N3",
        "claim": "2",
        "label": "Dual-flag survival: CONFIDENCE_LOW + LOW_SAMPLE together",
        "query": (
            "Predict the failure risk for a new compute-bound job to be submitted by "
            "usr_99999 requesting 972 nodes and a 6-hour walltime. Additionally: how many "
            "compute-bound jobs with exactly 972 nodes have ever appeared in the Fugaku "
            "dataset? And what job queue type does the Fugaku manual recommend for a "
            "compute-bound job at this scale?"
        ),
    },
    {
        "id":    "N4",
        "claim": "1",
        "label": "Repair cycle: CHALLENGE→DataExplorer→revision on walltime anomaly",
        "query": (
            "For compute-bound jobs that failed on Fugaku, break down the failures by "
            "walltime into three groups: under 1 hour, 1 to 8 hours, and over 8 hours. "
            "For each group, report both the count of failures and the average node count. "
            "The 1-to-8-hour group shows a strikingly different node count pattern — what "
            "does that imply operationally for HPC job management? And for the dominant "
            "failure group (over 8 hours), what is the correct pjsub directive to set a "
            "time limit?"
        ),
    },
    {
        "id":    "N5",
        "claim": "3",
        "label": "Claim-3 killer: high fact coverage, LOW_SAMPLE flag split",
        "query": (
            "usr_1898 has previously submitted jobs at 432 nodes. (a) What was their "
            "historical failure rate specifically for 432-node compute-bound jobs? "
            "(b) Based on that historical pattern, what failure risk does the system predict "
            "for their next 432-node compute-bound job with a 2-hour walltime? "
            "(c) Across all users on Fugaku, what is the overall failure rate for "
            "compute-bound jobs at exactly 432 nodes?"
        ),
    },
]

ALL_QUERIES = NEW_QUERIES + BATCH2_QUERIES + BATCH3_QUERIES   # N1–N15 combined

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN QUERIES — D1–D5  (domain-framed: quantum physics, nuclear chemistry,
#   climate, DFT, molecular dynamics — same DB/PA/doc agents, domain language)
# ─────────────────────────────────────────────────────────────────────────────
# Ground truth (DB-verified 2026-05-26):
#   D1  DFT / quantum chemistry  → pclass=compute-bound, 48-384 nodes, >4h
#       n=88,502  fail=9.13%  avg_nnumr=113.2  avg_elpl=24.3h
#       rscgrp=small (≤384 nodes)
#
#   D2  Quantum Monte Carlo / nuclear neutron transport
#       CB 288-node: n=522  fail=13.22%  LOW_SAMPLE likely
#       CB >384 nodes overall: n=119,614  fail=20.89%  avg_nnumr=922.7  avg_elpl=4.31h
#       rscgrp boundary: >384 nodes (large or equivalent)
#
#   D3  Climate / large-scale molecular dynamics  → pclass=memory-bound, >96 nodes, >4h
#       n=229,130  fail=15.21%  avg_nnumr=900.9  avg_elpl=20.1h
#       billing cost NOT IN DATA  (hallucination trap)
#       rscgrp: depends on node count
#
#   D4  Coupled simulation (nuclear + fluid): two-pclass comparison
#       CB: fail by pclass + usr_2111 profile; MB: same
#       usr_2111 >2h CB: n=13  fail=0%  avg_nnumr=1.0  avg_elpl=3h
#       usr_2111 >2h MB: n=10  fail=0%  avg_nnumr=9.8  avg_elpl=5.4h
#       (very small sample → LOW_SAMPLE for usr_2111 domain jobs)
#
#   D5  Quantum simulation energy audit + pjstat power monitoring
#       Global MB avg econ=12,030J vs CB avg econ=4,247J
#       Billing NOT IN DATA  (hallucination trap same as N9/N15)
#       pjstat -S / #PJM -S for energy measurement
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN_QUERIES = [
    {
        "id":    "D1",
        "claim": "4+1+2",
        "label": "DFT campaign: SQL→PA chain + Doc rscgrp + CONFIDENCE_LOW new group",
        "query": (
            "Our computational chemistry group is starting a density functional theory "
            "(DFT) campaign on Fugaku. DFT jobs are compute-bound and we plan to use "
            "between 48 and 384 nodes with walltimes over 4 hours. "
            "(a) Historically, what is the failure rate for compute-bound jobs in that "
            "node range with walltime over 4 hours, and what is the average node count "
            "for such jobs? "
            "(b) Using those historical averages, what failure risk does the system predict "
            "for a new DFT job at the typical node count and walltime, submitted by our "
            "group account (usr_dft_group — no prior Fugaku history)? "
            "(c) Which rscgrp should we request and what are the node allocation rules "
            "for a multi-node DFT job in the Fugaku manual?"
        ),
    },
    {
        "id":    "D2",
        "claim": "4+1+2",
        "label": "Nuclear neutron transport: LOW_SAMPLE at 288-node + large-scale chain",
        "query": (
            "I am running nuclear reactor neutron transport simulations (compute-bound) "
            "on Fugaku. My production runs require exactly 288 nodes for 12 hours. "
            "(a) How many compute-bound jobs at exactly 288 nodes appear in the Fugaku "
            "historical dataset, and what is their failure rate? "
            "(b) Given the historical data and the fact that I am a new Fugaku user "
            "(usr_nuclear_sim), what failure risk does the predictor assign my 288-node, "
            "12-hour compute-bound job — and are there confidence caveats I should know? "
            "(c) For nuclear simulation jobs that scale beyond 384 nodes in the future: "
            "what is the overall historical failure rate for compute-bound jobs above 384 "
            "nodes on Fugaku, and what pjsub directives apply at that scale?"
        ),
    },
    {
        "id":    "D3",
        "claim": "2+3+1",
        "label": "Climate CFD: large MB→PA chain + carbon-cost hallucination trap",
        "query": (
            "I lead a global climate modeling team running large computational fluid "
            "dynamics (CFD) simulations on Fugaku. Our jobs are memory-bound with more "
            "than 96 nodes and run for over 4 hours. "
            "(a) What is the historical failure rate for memory-bound jobs with more than "
            "96 nodes and walltime over 4 hours on Fugaku? What are the average node "
            "count and average walltime for this class of job? "
            "(b) Using those averages, predict the failure risk for our next climate run — "
            "submitted by usr_climate_team, a new project account. "
            "(c) For our sustainability report: what is the estimated carbon emissions "
            "cost in kg CO2 per node-hour for memory-bound jobs on Fugaku? "
            "(d) What pjsub directive should we use to set the memory-bound resource "
            "group, and what is the correct node allocation syntax for our scale?"
        ),
    },
    {
        "id":    "D4",
        "claim": "4+1",
        "label": "Coupled nuclear+fluid sim: two-pclass SQL join feeds PA prediction",
        "query": (
            "Our nuclear engineering lab runs two types of jobs on Fugaku: "
            "compute-bound neutron transport (CB) and memory-bound thermal-hydraulic "
            "simulations (MB). We submit under usr_2111. "
            "(a) For usr_2111, compare the failure rates and typical job sizes for "
            "compute-bound versus memory-bound jobs separately. Which class is more "
            "reliable for this user? "
            "(b) We want to couple the two simulations: the CB job outputs boundary "
            "conditions fed into the MB job. If the CB job uses usr_2111's typical "
            "compute-bound node count and the MB job uses usr_2111's typical memory-bound "
            "node count, what failure probability does the predictor assign each — and "
            "what is the combined probability that at least one job in the coupled pair "
            "fails? "
            "(c) What pjsub job dependency directive allows us to chain these two jobs "
            "so the MB job only starts after the CB job succeeds?"
        ),
    },
    {
        "id":    "D5",
        "claim": "2+3+1",
        "label": "Quantum sim energy audit: SQL→PA energy + billing REJECT + pjstat -S",
        "query": (
            "My quantum physics research group is preparing an energy efficiency report "
            "comparing our memory-bound quantum field theory simulations to compute-bound "
            "quantum chemistry jobs on Fugaku. "
            "(a) What is the average energy consumption (econ) per job for memory-bound "
            "jobs globally on Fugaku, and how does it compare to the average for "
            "compute-bound jobs? "
            "(b) Our next quantum field theory run: memory-bound, 192 nodes, 8-hour "
            "walltime, user usr_2111. What failure risk and expected energy consumption "
            "does the predictor give? "
            "(c) To include actual measured energy in our journal paper, which pjstat "
            "option or pjsub directive records post-job energy data on Fugaku? "
            "(d) Our funding agency requires reporting in yen per kWh. What is the "
            "electricity billing rate per node-hour for Fugaku jobs?"
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# BATCH 4 — N16–N25  (OS-trap hallucination + domain-framed multi-intent)
# ─────────────────────────────────────────────────────────────────────────────
# Ground truth (DB-verified 2026-05-26 against real Fugaku parquet):
#
#   N16  DFT campaign + macOS 'watch pjstat' OS trap
#        CB 48-192 nodes, elpl>4h: n=75,116  fail=9.84%  avg_nnumr=80.1  avg_elpl=24.0h
#        rscgrp=small (≤384 nodes)
#        OS TRAP: user says "watch pjstat on my MacBook" — Fugaku is Linux only; no macOS
#        'watch' available; correct answer is 'pjstat' loop on Linux login node
#        Pattern: a→b (SQL→PA via DATA_INSUFFICIENCY) + c (Doc) + d (OS TRAP→REJECT)
#
#   N17  MCNP nuclear neutron transport + Windows Task Manager OS trap
#        CB >384 nodes: n=119,614  fail=20.89%  avg_nnumr=923  avg_elpl=4.31h
#        rscgrp=large (>384 nodes boundary)
#        OS TRAP: user asks for Windows Task Manager equivalent → pjstat -v (Linux)
#        Pattern: a→b (SQL→PA) + c (Doc) + d (OS TRAP)
#
#   N18  Lattice QCD long-run: CB vs MB >24h comparison then PA
#        CB >24h: n=655,650  fail=6.79%  avg_nnumr=7.9   avg_elpl=61.74h
#        MB >24h: n=985,703  fail=16.02% avg_nnumr=22.1  avg_elpl=64.25h
#        MB is riskier at long runtimes (16% vs 7%)
#        Pattern: (a AND b)→c (comparison funnels to PA for riskier class) + d (Doc)
#
#   N19  Nuclear molecular dynamics + carbon footprint REJECT trap
#        MB 48-192 nodes: n=862,368  fail=9.91%  avg_nnumr=107.2  avg_econ=20,421J
#        carbon footprint NOT IN DATA  (REJECT expected)
#        Pattern: a→b (SQL→PA) + c (Doc pjstat -S) + d (REJECT carbon data)
#
#   N20  Quantum chemistry parallel chains: CB 48-node vs MB 384-node
#        CB nnumr=48: n=38,112  fail=7.86%  avg_elpl=12.85h
#        MB nnumr=384: n=27,973  fail=10.52% avg_elpl=9.88h
#        MB 384-node is riskier than CB 48-node (10.52% vs 7.86%)
#        Both use new account usr_qchem_team → CONFIDENCE_LOW on both chains
#        Pattern: (a→b) AND (c→d) AND e — two independent SQL→PA chains + Doc
#
#   N21  Biophysics membrane protein + Windows 'dir' disk-space OS trap
#        MB nnumr≤16: n=13,973,554  fail=9.39%  avg_elpl=8.51h  (massive sample)
#        OS TRAP: user asks for Windows 'dir' equivalent → df -h / ls -la on Linux
#        Pattern: a→b (SQL→PA) + c (Doc) + d (OS TRAP)
#
#   N22  Nuclear Monte Carlo: CB 192-node year trend → worst-year PA → Doc  (3-hop)
#        CB 192-node by year: 2021=3.82%  2022=4.83%  2023=29.35% (worst)  2024=5.13%
#        2023 avg failed elpl=29.94h  → PA input for next submission
#        Pattern: a→b→c (3-hop linear: SQL→PA→Doc)
#
#   N23  Cross-domain scale risk: CB vs MB >192 nodes comparison → PA + Doc
#        CB >192 nodes: n=144,836  fail=18.09%  avg_elpl=6.06h
#        MB >192 nodes: n=649,563  fail=8.71%   avg_elpl=5.80h
#        CB is riskier at large scale (18% vs 9%)
#        Pattern: (a AND b)→c→d — comparison feeds PA then Doc
#
#   N24  VQE quantum computing: CB 576-node + macOS 'watch -n 5 pjstat' OS trap
#        CB nnumr=576: n=614  fail=15.31%  avg_elpl=5.64h  avg_failed_elpl=9.39h
#        OS TRAP: 'watch' not available on Fugaku nodes; use pjstat loop or --wait
#        Pattern: a→b (SQL→PA, LOW_SAMPLE at 576) + c (OS TRAP)
#
#   N25  Genome+QC parallel diamond: MB 1024-node AND CB 48-192-node → PA → Doc
#        MB nnumr=1024: n=9,899  fail=19.42%  avg_elpl=6.77h
#        CB 48-192 nodes: n=75,116  fail=9.84%  (reuse N16 stats)
#        MB 1024-node is riskier (19.42% vs 9.84%)
#        Pattern: (a AND b)→c (diamond join: two SQL arms feed one PA comparison) + d (Doc)
# ─────────────────────────────────────────────────────────────────────────────
BATCH4_QUERIES = [
    {
        "id":    "N16",
        "claim": "1+2+3",
        "label": "DFT materials science: SQL→PA chain + Doc rscgrp + macOS OS trap [Claim 1+2+3]",
        "query": (
            "Our computational materials science group is starting a density functional "
            "theory (DFT) campaign on Fugaku. DFT jobs are compute-bound and we plan to "
            "use between 48 and 192 nodes with walltimes over 4 hours. "
            "(a) Historically, what is the failure rate for compute-bound jobs in that "
            "node range with walltime over 4 hours on Fugaku, and what are the average "
            "node count and walltime for such jobs? "
            "(b) Using those exact historical averages — not defaults — as job parameters "
            "for our new DFT campaign submitted by our group account usr_dft_lab (no "
            "prior Fugaku history), what failure probability does the system predict? "
            "(c) Which rscgrp should we specify in our pjsub scripts for 80-node "
            "compute-bound DFT jobs? "
            "(d) I usually track job progress on my MacBook by running 'watch pjstat' "
            "in a terminal — is that the right way to monitor jobs on Fugaku, or is "
            "there a different recommended approach?"
        ),
    },
    {
        "id":    "N17",
        "claim": "1+2+3",
        "label": "MCNP nuclear sim: large-scale SQL→PA + Doc large rscgrp + Windows OS trap [Claim 1+2+3]",
        "query": (
            "We are running MCNP6 neutron transport simulations for nuclear reactor "
            "design on Fugaku. These are compute-bound jobs requiring more than 384 nodes "
            "for our full-scale reactor models. "
            "(a) What is the historical failure rate for compute-bound jobs with more "
            "than 384 nodes on Fugaku? What are the typical average node count and "
            "walltime for such large-scale jobs? "
            "(b) Using those historical parameters, what failure probability does the "
            "system predict for our next MCNP run at that scale, submitted under our "
            "project account usr_mcnp_lab? "
            "(c) Which resource group (rscgrp) and pjsub directives does the Fugaku "
            "manual require for jobs that exceed 384 nodes? "
            "(d) We come from a Windows HPC cluster environment where we use Windows "
            "Task Manager to check CPU utilization across compute nodes. What is the "
            "Fugaku equivalent command for monitoring how our parallel MPI processes "
            "are consuming CPU resources?"
        ),
    },
    {
        "id":    "N18",
        "claim": "4+1+2",
        "label": "Lattice QCD long-run: (CB AND MB)>24h comparison → PA for riskier class + Doc [Claim 4+1+2]",
        "query": (
            "Our lattice QCD (quantum chromodynamics) simulations for particle physics "
            "typically require more than 24 hours of continuous computation. We need to "
            "decide between compute-bound and memory-bound job configurations before "
            "committing to a multi-month campaign. "
            "(a) What is the historical failure rate for compute-bound jobs that ran "
            "longer than 24 hours on Fugaku? What are the average node count and "
            "walltime for that class? "
            "(b) What is the historical failure rate for memory-bound jobs that ran "
            "longer than 24 hours on Fugaku? What are the average node count and "
            "walltime for that class? "
            "(c) The higher-risk configuration from (a) and (b) will be our primary "
            "strategy. Using its historical averages as job parameters, what failure "
            "probability does the system predict for our first lattice QCD run submitted "
            "by our group account usr_lqcd_team? "
            "(d) Does the Fugaku manual permit 24-hour jobs in the standard queue, and "
            "what is the maximum walltime allowed per resource group?"
        ),
    },
    {
        "id":    "N19",
        "claim": "2+3+1",
        "label": "Nuclear MD: MB 48-192-node SQL→PA + Doc energy + carbon REJECT trap [Claim 2+3+1]",
        "query": (
            "Our group runs molecular dynamics simulations for nuclear materials research "
            "— specifically radiation damage cascade simulations that are memory-bound "
            "and use between 48 and 192 nodes on Fugaku. "
            "(a) Historically, what is the failure rate and average energy consumption "
            "per job for memory-bound jobs in the 48–192 node range on Fugaku? What is "
            "the average node count and walltime for that class? "
            "(b) Using the historical profile from (a), what failure risk does the "
            "predictor assign our next 120-node memory-bound radiation cascade job "
            "submitted by usr_radcasc_group? "
            "(c) According to the Fugaku documentation, what pjstat command or pjsub "
            "directive should we use to record the actual energy consumption of each "
            "job for our energy efficiency report? "
            "(d) For our institution's sustainability report, we need the CO₂ carbon "
            "footprint and carbon emissions equivalent per node-hour for memory-bound "
            "jobs on Fugaku. What is that figure?"
        ),
    },
    {
        "id":    "N20",
        "claim": "4+1+2",
        "label": "Quantum chem parallel chains: (CB 48-node→PA) AND (MB 384-node→PA) AND Doc [Claim 4+1+2]",
        "query": (
            "We are evaluating two job strategies for a quantum chemistry campaign on "
            "Fugaku under our new group account usr_qchem_team. Strategy A uses "
            "compute-bound jobs at exactly 48 nodes. Strategy B uses memory-bound jobs "
            "at exactly 384 nodes. "
            "(a) What is the historical failure rate for compute-bound jobs at exactly "
            "48 nodes on Fugaku, and what is their average walltime? "
            "(b) Using Strategy A's historical statistics, predict the failure risk for "
            "a new 48-node compute-bound job submitted by usr_qchem_team — and flag any "
            "confidence limitations. "
            "(c) What is the historical failure rate for memory-bound jobs at exactly "
            "384 nodes on Fugaku, and what is their average walltime? "
            "(d) Using Strategy B's historical statistics, predict the failure risk for "
            "a new 384-node memory-bound job submitted by usr_qchem_team — again noting "
            "any confidence caveats. "
            "(e) For both job sizes, which rscgrp should we specify in the pjsub script "
            "according to the Fugaku documentation?"
        ),
    },
    {
        "id":    "N21",
        "claim": "1+2+3",
        "label": "Biophysics MD: MB small-node SQL→PA + Doc + Windows 'dir' OS trap [Claim 1+2+3]",
        "query": (
            "I am a biophysics researcher running large-scale membrane protein molecular "
            "dynamics simulations on Fugaku. These are memory-bound jobs at small node "
            "counts — we use 16 or fewer nodes. "
            "(a) What is the typical failure rate for memory-bound jobs with 16 or fewer "
            "nodes on Fugaku, and what is the average walltime for such jobs? "
            "(b) Given the historical statistics from (a) and the fact that I am a new "
            "Fugaku user (usr_biomd_pi, no prior job history), what failure probability "
            "does the predictor assign my next 12-node memory-bound 8-hour job? "
            "(c) According to the Fugaku manual, which rscgrp applies to jobs under "
            "16 nodes and what are the relevant submission directives? "
            "(d) I am migrating from a Windows HPC cluster. When I want to check how "
            "much disk space I have used on Fugaku, I normally type 'dir' in PowerShell "
            "to list files and sizes. What is the correct Linux command to use on Fugaku "
            "instead to check my disk quota and storage usage?"
        ),
    },
    {
        "id":    "N22",
        "claim": "4+1",
        "label": "Nuclear Monte Carlo: 3-hop CB 192-node year trend → worst-year PA → Doc [Claim 4+1]",
        "query": (
            "We are planning a nuclear reactor Monte Carlo simulation campaign for 2025 "
            "using 192-node compute-bound jobs on Fugaku. Before committing resources, "
            "we need to understand the historical reliability trend. "
            "(a) Break down the failure rate for compute-bound jobs at exactly 192 nodes "
            "by year from 2021 through 2024. Which year had the highest failure rate, "
            "and for that worst year, what was the average walltime of failed jobs? "
            "(b) Using the worst year's average failed walltime from (a) as the planned "
            "walltime for our next 192-node compute-bound Monte Carlo job (submitted by "
            "usr_montecarlo_team), what failure probability does the system predict — "
            "and should we be concerned? "
            "(c) What pjstat option should we use to continuously monitor the status of "
            "our long-running Monte Carlo jobs on Fugaku, and how do we retrieve job "
            "history after completion?"
        ),
    },
    {
        "id":    "N23",
        "claim": "4+1+2",
        "label": "Cross-domain scale risk: (CB AND MB) >192 nodes → PA for riskier + Doc [Claim 4+1+2]",
        "query": (
            "My computational physics group runs both compute-bound particle simulations "
            "and memory-bound fluid dynamics codes on Fugaku at large scale — all jobs "
            "use more than 192 nodes. We need to know which job class carries higher "
            "failure risk at that scale to prioritize our reliability engineering. "
            "(a) What is the historical failure rate for compute-bound jobs with more "
            "than 192 nodes on Fugaku, and what are the average node count and walltime? "
            "(b) What is the historical failure rate for memory-bound jobs with more "
            "than 192 nodes on Fugaku, and the same statistics? "
            "(c) For the riskier class identified in (a) and (b), what failure "
            "probability does the predictor assign our next 200-node job in that class "
            "with a 6-hour walltime, submitted by our group account usr_phys_group? "
            "(d) What rscgrp boundary applies at 200 nodes and what monitoring commands "
            "does the Fugaku documentation recommend for large-scale jobs?"
        ),
    },
    {
        "id":    "N24",
        "claim": "1+2+3",
        "label": "VQE quantum computing: CB 576-node SQL→PA (LOW_SAMPLE) + macOS 'watch' OS trap [Claim 1+2+3]",
        "query": (
            "Our quantum computing research group is benchmarking variational quantum "
            "eigensolver (VQE) algorithms on Fugaku using compute-bound jobs at exactly "
            "576 nodes — each run is a full classical optimizer iteration. "
            "(a) How many compute-bound jobs at exactly 576 nodes appear in the Fugaku "
            "historical dataset, and what is their failure rate and average walltime? "
            "Are there enough historical jobs at this exact scale for a high-confidence "
            "prediction? "
            "(b) Using the 576-node statistics from (a), what failure probability does "
            "the system predict for our next VQE run submitted under our group account "
            "usr_vqe_lab (new to Fugaku)? Flag any data-confidence issues explicitly. "
            "(c) I usually monitor long-running jobs on my Mac by running "
            "'watch -n 5 pjstat' in a terminal window — does this work on Fugaku's "
            "login nodes, or do I need to use a different method to periodically "
            "refresh job status?"
        ),
    },
    {
        "id":    "N25",
        "claim": "4+1+2",
        "label": "Genome+QC diamond join: (MB 1024-node AND CB 48-192-node)→PA comparison→Doc [Claim 4+1+2]",
        "query": (
            "Our lab runs two very different simulation types on Fugaku: (i) genome "
            "assembly pipelines that are memory-bound and run at 1024 nodes, and (ii) "
            "quantum chemistry DFT jobs that are compute-bound and run in the 48–192 "
            "node range. We need a joint risk assessment before planning our next "
            "allocation request. "
            "(a) What is the historical failure rate for memory-bound jobs at exactly "
            "1024 nodes on Fugaku, and what is the average walltime for such jobs? "
            "(b) What is the historical failure rate for compute-bound jobs in the "
            "48–192 node range with walltimes over 4 hours — the DFT profile? "
            "(c) Comparing (a) and (b): which workflow carries higher failure risk? "
            "For a new group account usr_compare_lab (no prior Fugaku history), what "
            "failure probability does the predictor assign each workflow? "
            "(d) For the 1024-node memory-bound genome jobs specifically, which rscgrp "
            "does Fugaku require and are there additional scheduling considerations or "
            "priority rules the documentation mentions for very large memory-bound jobs?"
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# BATCH 5 — N26–N35  (unexplored patterns: DAG reconvergence, CHALLENGE-primary,
#   3-user PA, temporal 2-hop, cascading dual uncertainty, OS+delegation,
#   cross-claim conflict, temporal large-scale, 3-way parallel PA, econ skew)
# ─────────────────────────────────────────────────────────────────────────────
# Ground truth (DB-verified 2026-05-26 against real Fugaku parquet):
#
#   N26  DAG convergent+divergent: (CB 96-node→PA) AND (MB 96-node→PA) → compare → Doc
#        CB nnumr=96: n=11,596  fail=5.63%  avg_elpl=16.91h  avg_failed_elpl=19.0h
#        MB nnumr=96: n=48,616  fail=9.42%  avg_elpl=8.05h   avg_failed_elpl=16.76h
#        MB is riskier at 96 nodes (9.42% vs 5.63%)
#        Pattern: (a→c) AND (b→c) → d  (two SQL arms converge into PA then PA→Doc)
#
#   N27  CHALLENGE→revision primary: CB 192-node energy distribution skew
#        CB nnumr=192 econ: n=12,269  avg=63,044J  p50=8,431J  p90=265,870J  max=1,494,207J
#        Mean/median=7.5x skew → reflector CHALLENGE on "avg energy" claim
#        Pattern: a→(CHALLENGE→revision→SYNTHESIZE+CAVEAT) + b(Doc)
#
#   N28  3-user PA comparison (SQL for all three feeds PA for worst user)
#        usr_1122 MB: n=705,657  fail=0.49%  avg_nnumr=11.25
#        usr_2111 MB: n=802,672  fail=2.30%  avg_nnumr=27.25
#        usr_1898 MB: n=40,654   fail=54.39% avg_nnumr=1.02
#        Pattern: (a AND b AND c)→d(compare)→PA(worst user)→Doc
#
#   N29  Temporal 2-hop: CB 192-node year breakdown → worst year → quarter breakdown → PA
#        CB nnumr=192 by year: 2023 is worst year (see N22 ground truth)
#        CB nnumr=192 2023 by quarter: Q1=5.86%(n=768) Q2=21.82%(n=1370) Q3=79.14%(n=163) Q4=81.65%(n=387)
#        Q4 2023 worst: avg_failed_elpl=40.89h
#        Pattern: a→b→c→d  (year SQL → quarter SQL → PA → Doc; true 4-hop temporal)
#
#   N30  Cascading dual uncertainty: CB 432-node extreme anomaly (84.37%!) + new user
#        CB nnumr=432: n=24,269  fail=84.37%  avg_elpl=4.39h  avg_failed_elpl=3.55h
#        Compare: CB >384 baseline fail=20.89% — 432-node is 4× worse than baseline
#        Pattern: a(SQL anomaly) → (HIGH_FAIL_CAVEAT + CONFIDENCE_LOW) → PA + Doc
#
#   N31  OS-trap + DATA_INSUFFICIENCY delegation: usr_2111 CB 288-node
#        CB nnumr=288 global: n=522  fail=13.22%  avg_elpl=11.88h
#        usr_2111 CB nnumr=288: n=0  → DATA_INSUFFICIENCY → PA fallback to global stats
#        OS TRAP: macOS Activity Monitor GUI → no equivalent on Fugaku; use pjstat
#        Pattern: a(SQL global) + b(DATA_INSUFFICIENCY delegation) + c(OS TRAP)
#
#   N32  Cross-claim conflict: usr_1912 (most reliable user) real stats + network trap
#        usr_1912: n=791,277  fail=0.04%  avg_nnumr=1.0  avg_elpl=4.3h  avg_econ=294J
#        TRAP: inter-node network latency per job NOT IN DATA (no network metric columns)
#        Pattern: a(real stats) + b(REJECT network trap) + c(comparison) + d(PA)
#
#   N33  Temporal dependency: CB >384 year trend → worst year → PA prediction
#        CB >384: 2021=14.83%  2022=27.48%(worst)  2023=27.23%  2024=1.36%(near-zero)
#        2022 CB >384: avg_failed_elpl=4.28h  avg_nnumr=1,133
#        Pattern: a→b→c→d  (year trend SQL → worst year drill → PA → Doc)
#
#   N34  3-way parallel PA: usr_1122 vs usr_2111 vs usr_1898, each gets own PA prediction
#        usr_1122 MB 10-node 1h: low risk (0.49% historical)
#        usr_2111 MB 27-node 3h: moderate (2.30% historical)
#        usr_1898 MB 1-node 1h: high risk (54.39% historical — surprising for small job)
#        Pattern: (a→PA_1) AND (b→PA_2) AND (c→PA_3) → 3-arm parallel, all converge
#
#   N35  Cross-claim econ skew CHALLENGE trigger: CB >192 nodes mean vs median
#        CB >192: n=144,836  avg_econ=180,960J  p50=29,126J  mean/median=6.2x
#        CHALLENGE expected when synthesizer reports avg without flagging extreme skew
#        Pattern: a→(CHALLENGE→revision→SYNTHESIZE+CAVEAT) + b(Doc)
# ─────────────────────────────────────────────────────────────────────────────
BATCH5_QUERIES = [
    {
        "id":    "N26",
        "claim": "4+1+2",
        "label": "DAG reconvergence: (CB 96-node AND MB 96-node)→PA comparison→Doc [Claim 4+1+2]",
        "query": (
            "I am scaling CFD simulations on Fugaku to 96 nodes but have not decided "
            "between compute-bound and memory-bound job configurations. Before committing "
            "to either approach, I need a joint risk assessment for both. "
            "(a) What is the historical failure rate for compute-bound jobs at exactly "
            "96 nodes on Fugaku? Report the average walltime and — separately — the "
            "average walltime of failed jobs only at that node count. "
            "(b) What is the historical failure rate for memory-bound jobs at exactly "
            "96 nodes on Fugaku? Report the same statistics: average walltime for all "
            "jobs and the average walltime of failed jobs. "
            "(c) Comparing (a) and (b): which configuration carries higher failure risk "
            "at 96 nodes? Using the riskier class's historical statistics, predict the "
            "failure probability for our next 96-node job submitted by our new group "
            "account usr_cfd_lab (no prior Fugaku history). "
            "(d) For 96-node jobs in the selected class, which rscgrp should we specify "
            "in our pjsub script, and what are the relevant submission directives?"
        ),
    },
    {
        "id":    "N27",
        "claim": "1+3",
        "label": "CHALLENGE→revision primary: CB 192-node econ mean/median skew (7.5x) [Claim 1+3]",
        "query": (
            "I am preparing a detailed energy analysis of 192-node compute-bound jobs "
            "on Fugaku for a performance paper and need the full distribution shape — "
            "not just a summary average. "
            "(a) For compute-bound jobs at exactly 192 nodes on Fugaku, what is the "
            "average (mean) energy consumption per job in joules? Report the raw number. "
            "(b) For the same job set, what is the median (p50) energy consumption? "
            "How does it compare to the mean from (a) — and is the mean a reliable "
            "representative value for energy budget planning? "
            "(c) What is the 90th percentile (p90) energy value and the single highest "
            "energy recorded for a 192-node compute-bound job? Given the spread between "
            "mean, median, p90, and max, what does this distribution imply for jobs "
            "planning their energy allowance? "
            "(d) According to the Fugaku documentation, which pjstat command or pjsub "
            "directive records per-job energy data so we can build our own distribution "
            "from empirical measurements?"
        ),
    },
    {
        "id":    "N28",
        "claim": "4+1+2",
        "label": "3-user MB comparison: SQL for all three → PA for worst → Doc [Claim 4+1+2]",
        "query": (
            "Three research groups use Fugaku for memory-bound simulations and we are "
            "preparing a comparative reliability audit for an allocation proposal: "
            "usr_1122, usr_2111, and usr_1898. "
            "(a) For each of the three users — usr_1122, usr_2111, and usr_1898 — report "
            "their total number of memory-bound jobs, their overall memory-bound failure "
            "rate, and their average node count for memory-bound jobs. "
            "(b) Rank the three users from most to least reliable based on their "
            "memory-bound failure rates. Which user is the most problematic, and by "
            "how much does their failure rate exceed the next-worst user? "
            "(c) For the least reliable user identified in (b), predict the failure "
            "risk for their next memory-bound job at their historical average node count "
            "with a 1-hour walltime. What data-confidence caveats apply? "
            "(d) The most unreliable user wants to improve — which Fugaku documentation "
            "resource explains how to diagnose job failures and what the common exit "
            "status codes mean for memory-bound jobs?"
        ),
    },
    {
        "id":    "N29",
        "claim": "4+1",
        "label": "Temporal 2-hop: CB 192-node year→worst-year quarter→PA→Doc [Claim 4+1]",
        "query": (
            "I need to drill down into the worst historical period for 192-node "
            "compute-bound jobs on Fugaku to calibrate risk expectations for our "
            "upcoming Monte Carlo campaign. "
            "(a) Break down the failure rate for compute-bound jobs at exactly 192 "
            "nodes by submission year from 2021 through 2024. Which year had the "
            "highest failure rate? "
            "(b) For the worst year identified in (a), break down the 192-node "
            "compute-bound failure rate further by quarter: Q1 (Jan–Mar), Q2 (Apr–Jun), "
            "Q3 (Jul–Sep), and Q4 (Oct–Dec). Which single quarter was worst, and what "
            "was its failure rate and job count? "
            "(c) For the worst quarter from (b), what was the average walltime of failed "
            "jobs at 192 nodes in that period? Using that average failed walltime as the "
            "planned walltime parameter for a new 192-node compute-bound Monte Carlo "
            "job submitted by usr_montecarlo_team, predict the failure probability. "
            "(d) What pjstat option should I use to continuously monitor long-running "
            "Monte Carlo jobs, and how do I retrieve the job history after completion?"
        ),
    },
    {
        "id":    "N30",
        "claim": "2+4",
        "label": "Cascading dual uncertainty: CB 432-node anomaly (84% fail!) + new user [Claim 2+4]",
        "query": (
            "I am a new Fugaku user (usr_newresearcher) planning compute-bound jobs at "
            "exactly 432 nodes — a node count fixed by our domain decomposition scheme "
            "and not negotiable. "
            "(a) How many compute-bound jobs at exactly 432 nodes exist in the Fugaku "
            "historical dataset? What is their failure rate and average requested walltime? "
            "(b) How does the 432-node failure rate compare to the overall baseline "
            "failure rate for large compute-bound jobs (more than 384 nodes) on Fugaku? "
            "Is the 432-node rate anomalously high, and what operational warning should "
            "this trigger before I proceed? "
            "(c) Given both the historical anomaly identified in (a) and the fact that "
            "I am a first-time Fugaku user with no prior job history, predict the failure "
            "risk for my 432-node, 4-hour compute-bound job. List every uncertainty flag "
            "that applies — do not suppress any caveats. "
            "(d) Given the extreme risk identified, should I reconsider this node count? "
            "What does the Fugaku documentation say about recommended node counts and "
            "allocation units for large compute-bound jobs?"
        ),
    },
    {
        "id":    "N31",
        "claim": "4+1+3",
        "label": "OS-trap + DATA_INSUFFICIENCY: usr_2111 CB 288-node personal gap + macOS trap [Claim 4+1+3]",
        "query": (
            "I am usr_2111 and I am planning my first compute-bound production run at "
            "exactly 288 nodes for our next project milestone. "
            "(a) For compute-bound jobs at exactly 288 nodes globally on Fugaku, what "
            "is the historical failure rate and average walltime? "
            "(b) Specifically for my account, usr_2111, how many compute-bound jobs at "
            "exactly 288 nodes do I have in my personal history on Fugaku? If my own "
            "data at this node count is insufficient, what is the system's fallback "
            "strategy for generating a prediction? "
            "(c) Using the best available data — my personal history if sufficient, "
            "otherwise the global statistics from (a) — predict the failure risk for "
            "my 288-node, 12-hour compute-bound job, and explicitly state which data "
            "source the prediction is based on. "
            "(d) I do all my work on a MacBook and I am used to opening Activity "
            "Monitor to watch CPU and memory usage for my processes. What is the "
            "equivalent way to check my job's resource utilization on Fugaku?"
        ),
    },
    {
        "id":    "N32",
        "claim": "2+1+3",
        "label": "Cross-claim conflict: usr_1912 (0.04% fail) real stats + network latency REJECT [Claim 2+1+3]",
        "query": (
            "We are publishing a benchmark paper on Fugaku user reliability and need "
            "detailed statistics for usr_1912, who appears in our preliminary analysis "
            "as an exceptionally reliable account. "
            "(a) For usr_1912, report their total job count, overall failure rate, "
            "average node count, average walltime, and average energy consumption per job. "
            "(b) How does usr_1912's failure rate compare to the system-wide average "
            "failure rate across all users and all jobs in the Fugaku dataset? Quantify "
            "the reliability advantage. "
            "(c) For our MPI parallel communication analysis section: what is the average "
            "inter-node network latency experienced by usr_1912's jobs during execution? "
            "We need this figure to estimate parallel efficiency for the paper. "
            "(d) Using usr_1912's historical profile, predict the failure probability "
            "for their next 1-node, 4-hour compute-bound job submission."
        ),
    },
    {
        "id":    "N33",
        "claim": "4+1+2",
        "label": "Temporal large-scale: CB >384 year trend → worst year PA → Doc [Claim 4+1+2]",
        "query": (
            "Our team runs very large compute-bound simulations — always more than "
            "384 nodes — on Fugaku and we are reviewing the historical reliability "
            "trend before planning our next allocation cycle. "
            "(a) Break down the failure rate for compute-bound jobs with more than "
            "384 nodes by submission year from 2021 through 2024. Which year had "
            "the highest failure rate and which year had the lowest? "
            "(b) For the worst year identified in (a), what was the average walltime "
            "of failed compute-bound jobs above 384 nodes, and what was the average "
            "node count of those failed jobs? "
            "(c) Using the worst year's average failed-job parameters from (b), what "
            "failure probability does the predictor assign our next large-scale run "
            "submitted by usr_largescale_team (no prior Fugaku history)? How should "
            "we interpret the near-zero 2024 failure rate when making our risk decision? "
            "(d) What scheduling directives — including rscgrp and any large-job "
            "specific options — does the Fugaku manual specify for compute-bound "
            "jobs requiring more than 384 nodes?"
        ),
    },
    {
        "id":    "N34",
        "claim": "4+1",
        "label": "3-way parallel PA: usr_1122, usr_2111, usr_1898 each get independent prediction [Claim 4+1]",
        "query": (
            "Three research groups each have a memory-bound job ready to submit to "
            "Fugaku this week. I need individual risk assessments for all three "
            "simultaneously before any of us queue our jobs. "
            "Job A: usr_1122 — memory-bound, 10 nodes, 1-hour walltime. "
            "Job B: usr_2111 — memory-bound, 27 nodes, 3-hour walltime. "
            "Job C: usr_1898 — memory-bound, 1 node, 1-hour walltime. "
            "(a) For each user — usr_1122, usr_2111, and usr_1898 — retrieve their "
            "personal memory-bound failure rate and total memory-bound job count "
            "from the historical database. "
            "(b) For each of the three jobs (A, B, and C), predict the individual "
            "failure probability using that user's personal memory-bound history. "
            "Present all three predictions side by side. "
            "(c) Which user has the highest predicted failure risk? Is there anything "
            "surprising about that result given the job sizes involved? Does the "
            "ordering match what you would expect from job size alone? "
            "(d) For the highest-risk job, what Fugaku documentation or diagnostics "
            "should that user consult before submitting, and what exit status codes "
            "should they watch for?"
        ),
    },
    {
        "id":    "N35",
        "claim": "1+3",
        "label": "Cross-claim econ skew CHALLENGE: CB >192 mean=180K J vs median=29K J (6.2x) [Claim 1+3]",
        "query": (
            "We are writing a sustainability paper on energy consumption of large "
            "compute-bound jobs on Fugaku. We need statistically rigorous energy "
            "figures for compute-bound jobs using more than 192 nodes — specifically "
            "we want the full distribution shape, not just a headline average. "
            "(a) How many compute-bound jobs with more than 192 nodes exist in the "
            "Fugaku dataset? What is the average (mean) energy consumption per job "
            "in joules for this class? "
            "(b) For the same job class, what is the median (p50) energy consumption? "
            "Report the mean-to-median ratio explicitly — and is the mean an appropriate "
            "representative value for our grant proposal energy budget? "
            "(c) Given the distribution shape implied by (a) and (b): are there extreme "
            "energy outlier jobs skewing the mean? What does this imply for interpreting "
            "energy statistics in sustainability reports for this class of job? "
            "(d) Which pjstat command or pjsub directive should we use to retrieve "
            "per-job energy measurements after job completion for our empirical dataset?"
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# BATCH 6 — N41–N55  (Interdependent multi-intent chains: SQL→PA, Doc→PA, SQL→Doc→PA)
# ─────────────────────────────────────────────────────────────────────────────
# SIGDIAL claim: typed A2A DialogueAct messaging is essential for uncertainty
# propagation across agents.  When agent B cannot answer (SQL REJECT), only
# the MAS architecture formally propagates this to agent A (PA), which then
# emits PARTIALLY_FOUND.  Blackboard and Unstructured proceed with defaults.
#
# Ground truth (verified against fugaku.duckdb / 2021-03-01 → 2024-05-08):
#   CB 192n 2022 fail: 4.83%     CB 192n 2023 fail: 29.35%
#   CB 512n fail:      11.56%    CB 4–12h walltime fail: 14.93%
#   CB 64n fail:       25.4%     CB 512n fail:           14.04%
#   MB 192n fail:       5.34%    CB overall fail:        10.74%
#   ec=4 count:       100,999    ec=4 avg walltime:       7.49 h
#   usr_1898 fail:    45.31%     usr_1898 most common nnumr: 1
#   usr_2111 fail:     2.26%     usr_2111 most common nnumr: 1
#   usr_3025 fail:   100.0%      usr_3025 most common nnumr: 432 (2.5 h avg)
#   distinct users 2024: 971
#
# NOT in schema (always REJECT): GPU util/pressure/model, inter-node bandwidth,
#   inter-node latency, thermal/temperature, OS/kernel version, billing/cost,
#   L2 cache miss rate, CPU pipeline stalls, real-time power monitoring,
#   thermal throttling events, institutional affiliation.
# ─────────────────────────────────────────────────────────────────────────────
BATCH6_QUERIES = [
    # ── N41: SQL→Doc→PA — CB 192n 2023 spike + large-scale pjsub + prediction ──────
    {
        "id":    "N41",
        "title": "Chain SQL→Doc→PA — CB 192n 2023 spike + pjsub directives + prediction [FC]",
        "query": (
            "Compute-bound jobs at exactly 192 nodes showed an unusual failure pattern in 2023. "
            "(a) What was the exact failure rate for those jobs in 2023? "
            "(b) What pjsub directives should a user specify when submitting a large-scale "
            "compute-bound job at 192 nodes to maximize their success rate? "
            "(c) Given this historical failure rate, predict the failure risk for a new "
            "compute-bound job at 192 nodes requesting a 3-hour walltime."
        ),
        "notes": (
            "FC test — coordination advantage: SQL INFORM(fail=29.35%, nnumr=192) → "
            "PA uses 192 not default 64; Doc chain provides large-scale submission context. "
            "GT: CB 192n 2023 fail=29.35%."
        ),
    },
    # ── N42: SQL→PA — CB 512n fail rate + GPU workload trap ──────────────────────
    {
        "id":    "N42",
        "title": "Chain SQL→PA — CB 512n fail rate + GPU workload trap [FC+UAA]",
        "query": (
            "For compute-bound jobs running at exactly 512 nodes: "
            "(a) What was the historical failure rate across the full dataset? "
            "(b) Based on the average GPU workload characteristics — including GPU utilization "
            "and GPU memory pressure — measured during those 512-node jobs, predict failure risk "
            "for a new compute-bound job at 512 nodes with a 6-hour walltime. "
            "(c) Which GPU model is installed in Fugaku's compute nodes?"
        ),
        "notes": (
            "UAA test: GPU data absent → SQL REJECT → MAS PA emits PARTIALLY_FOUND. "
            "BB/UN: give prediction with defaults, no GPU uncertainty on output. "
            "GT: CB 512n fail=11.56% (FC). GPU util/pressure/model → REJECT (UAA)."
        ),
    },
    # ── N43: SQL→Doc→PA three-way — usr_3025 100% fail + guidance + prediction ──
    {
        "id":    "N43",
        "title": "Chain SQL→Doc→PA — usr_3025 100% fail + config guidance + prediction [FC]",
        "query": (
            "User usr_3025 has a 100% failure rate across all their submitted jobs. "
            "(a) What node counts and walltimes does this user typically use? "
            "(b) What Fugaku documentation on job configuration, resource limits, or submission "
            "best practices could help usr_3025 improve their job success rate? "
            "(c) Given their typical job scale, predict failure risk for usr_3025's next "
            "submission at 432 nodes with a 2.5-hour walltime."
        ),
        "notes": (
            "FC test — three-way chain: SQL INFORM(nnumr=432, fail=100%) → PA uses 432 not 64; "
            "Doc provides failure-reduction context formally chained. "
            "GT: usr_3025 fail=100%, typical nnumr=432."
        ),
    },
    # ── N44: SQL→PA — WLE failures + inter-node bandwidth trap ──────────────────
    {
        "id":    "N44",
        "title": "Chain SQL→PA — WLE walltime-budget overruns + inter-node bandwidth trap [FC+UAA]",
        "query": (
            "Walltime-budget overruns (WLE events) are a common source of HPC job loss on Fugaku. "
            "(a) How many WLE events occurred across the full dataset, and what was the average "
            "requested walltime for those over-budget jobs? "
            "(b) For those over-budget jobs, what was the typical inter-node communication "
            "bandwidth utilization in bytes per second between compute nodes? "
            "(c) Using this inter-node bandwidth profile as a predictor, assess whether a new "
            "256-node memory-bound job requesting 6 hours is at elevated risk of a WLE event."
        ),
        "notes": (
            "UAA test: inter-node bandwidth absent → SQL REJECT → MAS PA PARTIALLY_FOUND. "
            "BB/UN: give count+avg, give prediction without network uncertainty. "
            "GT: WLE count=100,999, avg_wt=7.49h (FC). Bandwidth → REJECT (UAA)."
        ),
    },
    # ── N45: SQL+Doc→PA — CB 4–12h fail rate + walltime pjsub + prediction ──────
    {
        "id":    "N45",
        "title": "Chain SQL+Doc→PA — CB 4–12h fail rate + walltime pjsub directives + prediction [FC]",
        "query": (
            "Compute-bound jobs with walltimes in the 4–12 hour range: "
            "(a) What is the overall failure rate for compute-bound jobs in this walltime range? "
            "(b) What pjsub directives and documentation guidance exist for setting appropriate "
            "walltime limits to avoid job termination on Fugaku? "
            "(c) Based on both the historical failure statistics for this walltime range and the "
            "documented best practices, predict failure risk for a new 128-node compute-bound "
            "job requesting a 6-hour walltime."
        ),
        "notes": (
            "FC test — SQL+Doc→PA: SQL INFORM(14.93%) + Doc INFORM(walltime directives) → "
            "PA formally uses both for grounded prediction. "
            "GT: CB 4–12h fail=14.93%."
        ),
    },
    # ── N46: SQL→PA — usr_1898 45% fail + typical node scale + affiliation trap ──
    {
        "id":    "N46",
        "title": "Chain SQL→PA — usr_1898 45% fail + most common scale + affiliation trap [FC+UAA]",
        "query": (
            "User usr_1898 is the highest-volume submitter on the system. "
            "(a) What is this user's overall failure rate, and what node count do they use most "
            "frequently for compute-bound jobs? "
            "(b) Which research institution or university is usr_1898 affiliated with, and what "
            "compute allocation project funds their jobs? "
            "(c) Given their typical job configuration, predict failure risk if usr_1898 submits "
            "a compute-bound job at their most common node count with a 2-hour walltime."
        ),
        "notes": (
            "FC+UAA: SQL INFORM(nnumr=1, fail=45.31%) → PA uses nnumr=1 not default 64. "
            "Affiliation NOT in schema → SQL REJECT (UAA). "
            "GT: usr_1898 fail=45.31%, most common nnumr=1. Affiliation → REJECT."
        ),
    },
    # ── N47: SQL→PA — CB 192n 2022 vs 2023 rates + thermal/OS kernel trap ──────
    {
        "id":    "N47",
        "title": "Chain SQL→PA — CB 192n 2022 vs 2023 spike + thermal/OS kernel trap [FC+UAA]",
        "query": (
            "Compute-bound jobs at 192 nodes showed a dramatic change between 2022 and 2023. "
            "(a) What were the exact failure rates for compute-bound 192-node jobs in 2022 and in 2023? "
            "(b) Did elevated node operating temperatures or changes in the OS kernel version "
            "contribute to the failure spike? What thermal or system-level data supports this? "
            "(c) Predict failure risk for a new compute-bound job at 192 nodes submitted today, "
            "accounting for any thermal or kernel-level risk factors identified."
        ),
        "notes": (
            "FC+UAA: CB 192n 2022=4.83%, 2023=29.35% (FC). "
            "Thermal data and OS/kernel version absent → SQL REJECT → MAS PA PARTIALLY_FOUND. "
            "BB/UN: give rates, give confident prediction without thermal/OS caveat."
        ),
    },
    # ── N48: SQL+SQL→PA — MB 192n vs CB overall + riskier pclass prediction ──────
    {
        "id":    "N48",
        "title": "Chain SQL+SQL→PA — MB 192n vs CB overall fail rates + riskier pclass prediction [FC]",
        "query": (
            "Compare memory-bound and compute-bound workloads at 192-node scale: "
            "(a) What is the failure rate for memory-bound jobs at exactly 192 nodes? "
            "(b) What is the overall failure rate for compute-bound jobs across the entire dataset? "
            "(c) Which workload class is historically riskier? Predict failure risk for a new "
            "192-node job in the riskier workload class with a 4-hour walltime."
        ),
        "notes": (
            "FC test — two SQL INFORMs → PA correctly identifies CB (10.74%) as riskier vs MB (5.34%). "
            "PA prediction uses pclass='compute-bound' from SQL reasoning. "
            "GT: MB 192n fail=5.34%, CB overall fail=10.74%."
        ),
    },
    # ── N49: Doc+SQL→PA — pjstat monitoring + 2024 user count + new user prediction ──
    {
        "id":    "N49",
        "title": "Chain Doc+SQL→PA — pjstat monitoring + 2024 user count + new user prediction [FC]",
        "query": (
            "A new Fugaku user wants to understand how to monitor and optimize their jobs. "
            "(a) What pjstat commands and monitoring tools does Fugaku provide for tracking "
            "running jobs and diagnosing issues? "
            "(b) How many distinct users submitted jobs on Fugaku in 2024? "
            "(c) For this new user submitting their first 64-node compute-bound job at 1-hour "
            "walltime, predict failure risk — and explicitly flag any uncertainty arising from "
            "the absence of personal submission history."
        ),
        "notes": (
            "FC test — Doc+SQL→PA: Doc INFORM(pjstat commands) + SQL INFORM(971 users) + "
            "PA predict with CONFIDENCE_LOW formally propagated. "
            "GT: distinct_users_2024=971. CONFIDENCE_LOW must appear in answer."
        ),
    },
    # ── N50: SQL→PA — MB 192n job loss rate + inter-node latency trap ───────────
    {
        "id":    "N50",
        "title": "Chain SQL→PA — MB 192n job loss rate + inter-node MPI latency trap [FC+UAA]",
        "query": (
            "For memory-bound workloads running at exactly 192 nodes on Fugaku: "
            "(a) What percentage of these large-scale jobs did not complete successfully? "
            "(b) What was the measured average MPI message-passing round-trip latency in "
            "microseconds between compute nodes for these workloads, and how does that "
            "round-trip time correlate with job completion outcomes? "
            "(c) Using this MPI latency profile as an input, estimate the risk of "
            "unsuccessful completion for a new 192-node memory-bound job with a 4-hour walltime."
        ),
        "notes": (
            "UAA test: inter-node latency absent → SQL REJECT → MAS PA PARTIALLY_FOUND. "
            "BB/UN: give job loss rate, give confident prediction without latency caveat. "
            "GT: MB 192n fail=5.34% (FC). Inter-node latency → REJECT (UAA)."
        ),
    },
    # ── N51: SQL→Doc→PA full chain — WLE overruns + walltime guidance + prediction ──
    {
        "id":    "N51",
        "title": "Chain SQL→Doc→PA — WLE overruns + walltime pjsub guidance + prediction [FC]",
        "query": (
            "Walltime budget overruns (WLE events) are a common efficiency problem on Fugaku. "
            "(a) How many total Fugaku jobs were lost to walltime budget overruns, and what was "
            "the average requested walltime for those over-budget jobs? "
            "(b) What guidance does Fugaku documentation provide for setting safe walltime limits "
            "and structuring walltime requests to avoid over-budget events? "
            "(c) Using the average walltime of over-budget jobs as the planned walltime, predict "
            "resource loss risk for a new compute-bound job at 256 nodes."
        ),
        "notes": (
            "FC test — full SQL→Doc→PA chain: SQL INFORM(100999, 7.49h) → Doc INFORM → "
            "PA uses 7.49h not arbitrary default walltime. "
            "GT: WLE count=100,999, avg_wt=7.49h."
        ),
    },
    # ── N52: SQL→PA — usr_2111 fail rate + billing/allocation trap + prediction ──
    {
        "id":    "N52",
        "title": "Chain SQL→PA — usr_2111 2.26% fail + billing/allocation trap + prediction [FC+UAA]",
        "query": (
            "User usr_2111 is one of the highest-volume submitters on the system. "
            "(a) What is this user's overall failure rate and their most common job node count? "
            "(b) What compute allocation budget — in core-hours or billing units — has been "
            "charged to usr_2111 for their failed jobs across the entire dataset? "
            "(c) Given this user's history, predict failure risk for a new job from usr_2111 "
            "at 192 nodes with a 2-hour compute-bound workload."
        ),
        "notes": (
            "FC+UAA: SQL INFORM(fail=2.26%, nnumr=1) + SQL REJECT(billing) → PARTIALLY_FOUND. "
            "BB/UN: give fail rate, give prediction, no billing uncertainty flagged. "
            "GT: usr_2111 fail=2.26%, most common nnumr=1. Billing → REJECT."
        ),
    },
    # ── N53: SQL→PA — 64n vs 512n CB comparison + prediction at riskier scale ──
    {
        "id":    "N53",
        "title": "Chain SQL→PA — 64n vs 512n CB fail rates + prediction at riskier scale [FC]",
        "query": (
            "Compare compute-bound job failure rates at two scales: "
            "(a) What is the historical failure rate for compute-bound jobs at exactly 64 nodes, "
            "and what is it for compute-bound jobs at exactly 512 nodes? "
            "(b) Which node count is historically riskier, and what factors might explain this? "
            "(c) Using the exact node count of the riskier configuration, predict failure risk "
            "for a new compute-bound job at that scale with a 4-hour walltime."
        ),
        "notes": (
            "FC test — counterintuitive finding: 64n fail=25.4% > 512n fail=14.04%. "
            "MAS: SQL INFORM(64n=25.4%, 512n=14.04%) → PA picks nnumr=64 as riskier. "
            "BB/UN: PA may assume 512 (expected 'large=riskier') or use arbitrary default. "
            "GT: 64n fail=25.4%, 512n fail=14.04%."
        ),
    },
    # ── N54: SQL→PA — CB 192n 2022/2023 + L2 cache miss / pipeline stall trap ──
    {
        "id":    "N54",
        "title": "Chain SQL→PA — CB 192n 2022/2023 rates + L2 cache miss / pipeline stall trap [FC+UAA]",
        "query": (
            "The 2023 failure spike for 192-node compute-bound jobs warrants investigation. "
            "(a) What were the failure rates for compute-bound jobs at 192 nodes in 2022 and in 2023? "
            "(b) What was the average L2 cache miss rate and CPU pipeline stall frequency for those "
            "2023 failed jobs — do these micro-architectural metrics explain the spike? "
            "(c) Accounting for these cache-level performance characteristics, predict failure risk "
            "for a new 192-node compute-bound job in 2024."
        ),
        "notes": (
            "FC+UAA: CB 192n 2022=4.83%, 2023=29.35% (FC). "
            "L2 cache miss rate and CPU pipeline stalls absent → SQL REJECT → MAS PA PARTIALLY_FOUND. "
            "BB/UN: give rates, give confident prediction without cache/pipeline caveat."
        ),
    },
    # ── N55: Doc→SQL→PA three-way + power monitoring / thermal throttling trap ──
    {
        "id":    "N55",
        "title": "Chain Doc→SQL→PA — MB 192n pjsub + fail rate + real-time power monitoring trap [FC+UAA]",
        "query": (
            "A user is planning large-scale memory-bound jobs on Fugaku. "
            "(a) What pjsub resource directives and submission best practices apply specifically "
            "to large-scale memory-bound jobs on Fugaku? "
            "(b) What is the historical failure rate for memory-bound jobs at exactly 192 nodes? "
            "(c) For those large-scale memory-bound jobs, what was the real-time CPU power draw "
            "monitoring data, and did thermal throttling events contribute to the failures? "
            "(d) Incorporating the documented best practices and the historical failure rate, "
            "predict failure risk for a new 256-node memory-bound job at a 3-hour walltime."
        ),
        "notes": (
            "FC+UAA — three-way chain: Doc INFORM(pjsub) + SQL INFORM(MB 192n fail=5.34%) + "
            "SQL REJECT(real-time power/thermal throttling) → PA PARTIALLY_FOUND. "
            "BB/UN: Doc retrieved separately; no formal chain to PA; power caveat absent. "
            "GT: MB 192n fail=5.34%. Real-time power monitoring + thermal throttling → REJECT."
        ),
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────
SEP   = "=" * 72

def _header(system: str, suite: str):
    label = {
        "batch1": "N1–N5", "batch2": "N6–N10", "batch3": "N11–N15",
        "batch4": "N16–N25 (OS-trap + Domain)", "batch5": "N26–N35 (DAG + CHALLENGE + temporal)",
        "batch6": "N41–N55 (Interdependent chain queries: SQL→PA, Doc→PA, SQL→Doc→PA)",
        "domain": "D1–D5 (Domain)", "all": "N1–N55 + D1–D5 (55 queries)",
    }.get(suite, suite)
    print(f"\n{SEP}")
    print(f"  SIGDIAL QUERIES {label}  —  system: {system.upper()}")
    print(SEP)

def _print_q(q: dict):
    print(f"\n{SEP}")
    # BATCH6 uses 'title'; earlier batches use 'claim'+'label'
    if "title" in q:
        print(f"  [{q['id']}] {q['title']}")
    else:
        print(f"  [{q['id']}] Claim {q['claim']} — {q['label']}")
    print(f"  {textwrap.fill(q['query'], 70, subsequent_indent='  ')}")
    print(SEP)

def _print_ans(answer: str, dur: float):
    print(f"\n── ANSWER ({dur:.1f}s) ──")
    print(textwrap.fill(answer, 72))

# ── Runners ───────────────────────────────────────────────────────────────────
def run_mas(queries):
    from orchestrator import HpcOrchestrator
    agent = HpcOrchestrator(verbose=True)

    async def _go():
        for q in queries:
            _print_q(q)
            t0 = time.perf_counter()
            try:
                ans = await agent.gateway.run(q["query"], f"n_{q['id']}")
            except Exception as e:
                ans = f"[SYSTEM ERROR — query skipped: {type(e).__name__}: {e}]"
            _print_ans(ans, time.perf_counter() - t0)

    asyncio.run(_go())


def run_blackboard(queries):
    _patch_rag_with_bm25()               # swap Qdrant → BM25 before agent init
    from blackboard_baseline.agent import BlackboardMAS
    agent = BlackboardMAS()
    for q in queries:
        _print_q(q)
        t0 = time.perf_counter()
        try:
            ans = agent.run(q["query"])
        except Exception as e:
            ans = f"[SYSTEM ERROR — query skipped: {type(e).__name__}: {e}]"
        _print_ans(ans, time.perf_counter() - t0)


def run_unstructured(queries):
    _patch_rag_with_bm25()               # swap Qdrant → BM25 before agent init
    from unstructured_baseline.agent import UnstructuredMAS
    agent = UnstructuredMAS()
    for q in queries:
        _print_q(q)
        t0 = time.perf_counter()
        try:
            ans = agent.run(q["query"])
        except Exception as e:
            ans = f"[SYSTEM ERROR — query skipped: {type(e).__name__}: {e}]"
        _print_ans(ans, time.perf_counter() - t0)


def run_unstructured_a2a(queries):
    _patch_rag_with_bm25()               # swap Qdrant → BM25 before agent init
    from unstructured_a2a_baseline.agent import NaturalA2AMAS
    agent = NaturalA2AMAS()
    for q in queries:
        _print_q(q)
        t0 = time.perf_counter()
        try:
            ans = agent.run(q["query"])
        except Exception as e:
            ans = f"[SYSTEM ERROR — query skipped: {type(e).__name__}: {e}]"
        _print_ans(ans, time.perf_counter() - t0)


def run_single_agent(queries):
    _patch_rag_with_bm25()               # swap Qdrant → BM25 before agent init
    from single_agent_baseline.agent import SingleAgent
    agent = SingleAgent(verbose=False)
    for q in queries:
        _print_q(q)
        t0 = time.perf_counter()
        try:
            ans = agent.run(q["query"])
        except Exception as e:
            ans = f"[SYSTEM ERROR — query skipped: {type(e).__name__}: {e}]"
        _print_ans(ans, time.perf_counter() - t0)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    system = sys.argv[1].lower() if len(sys.argv) > 1 else "mas"
    suite  = sys.argv[2].lower() if len(sys.argv) > 2 else "batch3"

    queries = {
        "batch1":  NEW_QUERIES,
        "batch2":  BATCH2_QUERIES,
        "batch3":  BATCH3_QUERIES,
        "batch4":  BATCH4_QUERIES,
        "batch5":  BATCH5_QUERIES,
        "batch6":  BATCH6_QUERIES,
        "domain":  DOMAIN_QUERIES,
        "all":     ALL_QUERIES + BATCH4_QUERIES + BATCH5_QUERIES + DOMAIN_QUERIES + BATCH6_QUERIES,
    }.get(suite)
    if queries is None:
        print(f"Unknown suite '{suite}'. Choose: batch1 | batch2 | batch3 | batch4 | batch5 | batch6 | domain | all")
        sys.exit(1)

    _header(system, suite)

    if system == "mas":
        run_mas(queries)
    elif system == "blackboard":
        run_blackboard(queries)
    elif system == "unstructured":
        run_unstructured(queries)
    elif system in ("unstructured_a2a", "un_a2a", "a2a"):
        run_unstructured_a2a(queries)
    elif system in ("single_agent", "sa", "single"):
        run_single_agent(queries)
    else:
        print(f"Unknown system '{system}'. Choose: mas | blackboard | unstructured | unstructured_a2a | single_agent")
        sys.exit(1)
