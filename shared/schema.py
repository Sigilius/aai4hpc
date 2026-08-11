# shared/schema.py
# Schema definitions for Fugaku job telemetry database.
# This is the authoritative domain knowledge injected into every SQL/doc prompt.

FUGAKU_SCHEMA_NOTES = """
WHAT THIS DATASET DOES NOT CONTAIN — return CANNOT_GENERATE for these:
- NO temperature data: no CPU, node, ambient, or thermal sensor columns
- NO financial data: no cost, billing, price, budget, rate, or tariff columns
- NO GPU data: Fugaku has no GPUs; no GPU utilization, GPU count, or GPU memory columns
- NO network metrics: no interconnect bandwidth, latency, or topology columns
- NO user personal data: users are anonymized as usr_XXXX; no names, emails, or affiliations
- NO real-time or live data: this is a historical archive (Mar 2021 – Apr 2024 only)
- NO operating system data: no OS type, machine type, Windows/Linux/macOS columns; Fugaku runs Linux only and OS is not logged per-job
- NO hardware configuration per job: no CPU model, rack, node identifier, or machine class columns beyond nnumr/nnuma

COLUMN MEANINGS THAT ARE COMMONLY MISREAD:
- uctmut (DOUBLE): accumulated user-mode CPU time in milliseconds across all cores.
  Values are in the millions-to-billions range. NOT temperature. NOT utilization %.
- cnumut (INTEGER): count of CPU cores that executed work. NOT a utilization percentage.
- perf1–perf6: raw hardware performance counter values in trillions. NOT human-readable.
  Do not interpret, label, or compute averages on perf columns without normalization.

KEY FACTS ABOUT THIS DATA:
- 713,617 jobs per monthly file, Mar 2021 – Apr 2024 (38 files)
- Zero null values across all columns
- 1 Fugaku node = 48 CPU cores (A64FX ARM). No GPUs.
- cnumr = cores requested. nnumr = nodes requested. Relationship: cnumr ≈ nnumr * 48

POWER COLUMNS (avgpcon, minpcon, maxpcon, econ):
- econ: total energy consumed by the job in JOULES (not a power limit)
- avgpcon = TOTAL job power in watts across ALL allocated nodes (not per-node)
- There is NO per-job power limit or cap column in the schema.
  Power limits are system-side controls and are not logged in telemetry.
- For per-node power: use avgpcon / nnuma
- 1 corrupt row exists with extreme avgpcon: ONLY add WHERE avgpcon BETWEEN 50 AND 10000000
  when your SELECT or WHERE clause references avgpcon, minpcon, or maxpcon directly.
  Do NOT add this filter for queries about econ, duration, node counts, or user stats.
- Realistic single-node range: 100W–200W

JOB CLASSIFICATION:
- pclass: only 2 values — 'compute-bound' and 'memory-bound'
- compute-bound = high operational intensity (opint > ~10)
- memory-bound = low operational intensity, bandwidth-limited
- No 'leadership class' definition exists — use nnumr thresholds if needed

JOB SIZE CLASSIFICATION (use these thresholds when user says 'large', 'small', etc.):
- Small jobs:   nnumr < 10
- Medium jobs:  nnumr 10–384
- Large jobs:   nnumr > 384
- Massive jobs: nnumr > 1000 (rare outliers — only use if user explicitly says 'massive' or '>1000')

Column reference:
- pclass (VARCHAR): The job class/category column. Contains 'compute-bound' or
  'memory-bound'. Always use GROUP BY pclass when the query asks about job classes,
  job types, or job categories — never GROUP BY nnumr for class-level questions.
- nnumr (INTEGER): Number of nodes actually used. Use for filtering (e.g. large jobs),
  not for grouping by job class.
- elpl (DOUBLE): The REQUESTED walltime limit in seconds (the time the user asked for).
  Use elpl when asked about "walltime", "time limit", "max walltime", "requested walltime",
  "planned walltime", or "average walltime". Do NOT use duration as a proxy for walltime.
  CRITICAL: For failed jobs, duration << elpl because jobs terminate before hitting the limit.
  Always use elpl in failure-analysis walltime queries unless "actual runtime" is specified.
- duration (DOUBLE, pre-computed): ACTUAL elapsed time in seconds the job ran
  (= EPOCH(edt) - EPOCH(sdt)). Use ONLY when asked "how long did the job actually run",
  "actual execution time", or "actual runtime". Never use duration as a walltime stand-in
  for failed-job analysis.
- exit state (VARCHAR): Job exit status. Known values:
  - 'completed' = job finished successfully
  - 'failed'    = job failed
  Failure condition: "exit state" = 'failed'
  Success condition: "exit state" = 'completed'
  Do NOT use IS NOT NULL or != 'success' — use the exact values above.

SENTINEL VALUES (do not analyze these as real data):
- msza = 18446744073709551615 means no memory limit set (uint64 max)
- deldt = 1970-01-01 means no deadline set (unix epoch)
- pri = 127 for ALL jobs — constant, useless for filtering

DERIVED METRICS USEFUL FOR ANALYSIS:
- Wait time in seconds = EPOCH(CAST(sdt AS TIMESTAMP)) - EPOCH(CAST(qdt AS TIMESTAMP))
- Job duration in seconds = EPOCH(CAST(edt AS TIMESTAMP)) - EPOCH(CAST(sdt AS TIMESTAMP))
  (the pre-computed `duration` column is also available and equivalent)
- Per-node power = avgpcon / nnuma
- Core utilization = cnumut / cnumat

TEMPORAL FILTER RULES:
- "jobs in YEAR" / "submitted in YEAR" → filter by qdt (queue/submission time)
  e.g. qdt >= '2023-01-01' AND qdt < '2024-01-01'
- "jobs that ran in YEAR" / "jobs that started in YEAR" → filter by sdt (start time)
- For average wait time "in YEAR": filter by qdt (jobs submitted that year)

QUIRKS TO WARN AGENT ABOUT:
- nnuma > nnumr for 1% of jobs (scheduler over-allocation)
- Short jobs (duration < 60s) are test runs, not production
- Massive jobs (nnumr > 1000) are rare but skew all power/perf means — treat as outliers
- perf1-perf6 are raw hardware counter values in trillions — not human-readable without normalization
- freq_req/freq_alloc: only 2 unique values (2000 MHz = normal, other = boosted)
- jobenv_req: 3 types — ask doc agent to clarify what each means
- jobenv_req codes are anonymized (jobenv_req_0 through jobenv_req_8, not sequential)
  — ask doc agent what each code represents if user asks about job environment types

DUCKDB SYNTAX RULES (strictly follow these):
- Date arithmetic: CURRENT_DATE - INTERVAL '7' DAY   (NOT DATE_SUB)
- Last N days: sdt >= CURRENT_DATE - INTERVAL '7' DAY
- Epoch to timestamp: to_timestamp(epoch_col)
- String contains: col LIKE '%value%'  or  contains(col, 'value')
- No LIMIT needed unless user asks for top-N
- sdt, qdt, edt, deldt are stored as VARCHAR in ISO-8601 format (e.g. "2023-01-15 10:30:00+09:00")

- DATE RANGE FILTERING — always use TIMESTAMPTZ (plain comparisons do NOT trigger pytz):
    CAST(sdt AS TIMESTAMPTZ) >= '2022-01-01'::TIMESTAMPTZ
    AND CAST(sdt AS TIMESTAMPTZ) <  '2022-02-01'::TIMESTAMPTZ
  NEVER use LEFT(sdt, 7) for WHERE filtering — it gives wrong counts (~6% error).

- YEAR/MONTH GROUPING in SELECT — use LEFT() on the raw VARCHAR to avoid pytz:
    LEFT(sdt, 7)  → "2023-01"  (YYYY-MM, use for GROUP BY month)
    LEFT(sdt, 4)  → "2023"     (YYYY, use for GROUP BY year)
  Combined pattern for monthly breakdown queries:
    SELECT LEFT(sdt, 7) AS month, COUNT(*) AS jobs
    FROM jobs
    WHERE CAST(sdt AS TIMESTAMPTZ) >= '2023-01-01'::TIMESTAMPTZ
      AND CAST(sdt AS TIMESTAMPTZ) <  '2024-01-01'::TIMESTAMPTZ
    GROUP BY month ORDER BY month

- EPOCH/time-difference — cast to TIMESTAMP (NOT TIMESTAMPTZ) to avoid pytz:
    EPOCH(CAST(sdt AS TIMESTAMP)) - EPOCH(CAST(qdt AS TIMESTAMP))  → seconds as DOUBLE
  NEVER subtract two TIMESTAMPTZ columns directly — produces INTERVAL, cannot AVG/SUM.
- jobenv_req: VARCHAR, categorical, 5 encoded values only:
    'jobenv_req_0', 'jobenv_req_1', 'jobenv_req_2', 'jobenv_req_6', 'jobenv_req_8'
  Never cast to numeric. Use for grouping only:
    SELECT jobenv_req, COUNT(*) FROM jobs GROUP BY jobenv_req ORDER BY COUNT(*) DESC
  Meaning of each code is unknown — do not interpret or label them

CASE WHEN BUCKET QUERIES — DUCKDB PATTERN (CRITICAL):
- When grouping by a computed bucket (CASE WHEN expression), you MUST GROUP BY the alias, NOT the raw column.
- CORRECT pattern:
    SELECT
      CASE WHEN nnumr = 1     THEN '1 node'
           WHEN nnumr <= 16   THEN '2-16 nodes'
           WHEN nnumr <= 128  THEN '17-128 nodes'
           WHEN nnumr <= 384  THEN '129-384 nodes'
           ELSE                    '>384 nodes'
      END AS bucket,
      COUNT(*) AS job_count
    FROM jobs
    GROUP BY bucket          -- GROUP BY the alias, not nnumr
    ORDER BY MIN(nnumr)      -- ORDER BY an aggregate for correct sort
- NEVER write: GROUP BY nnumr when nnumr is only inside CASE WHEN (not in SELECT directly).
  DuckDB will throw: "column must appear in GROUP BY or be part of an aggregate".
- This rule applies to ANY computed expression alias used in SELECT — always GROUP BY the alias.
"""
