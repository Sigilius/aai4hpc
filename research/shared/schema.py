# schema.py

FUGAKU_SCHEMA_NOTES = """
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
- 1 corrupt row exists: filter with WHERE avgpcon BETWEEN 50 AND 10000000
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
- Wait time = sdt - qdt (queue wait)
- Per-node power = avgpcon / nnuma
- Core utilization = cnumut / cnumat

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
- sdt, qdt, edt, deldt are stored as VARCHAR with timezone — always cast as:
  CAST(sdt AS TIMESTAMPTZ) >= '2024-03-01'::TIMESTAMPTZ
- jobenv_req: VARCHAR, categorical, 5 encoded values only:
    'jobenv_req_0', 'jobenv_req_1', 'jobenv_req_2', 'jobenv_req_6', 'jobenv_req_8'
  Never cast to numeric. Use for grouping only:
    SELECT jobenv_req, COUNT(*) FROM jobs GROUP BY jobenv_req ORDER BY COUNT(*) DESC
  Meaning of each code is unknown — do not interpret or label them
"""