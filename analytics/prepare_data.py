"""
Step 1b — Data preparation for Fugaku failure classifier + regression models.
Outputs: train.parquet, test.parquet (80/20 chronological split)
"""
import duckdb
import numpy as np

import os
PARQUET_GLOB = os.path.join(os.getenv("FUGAKU_DATA_PATH", "data/fugaku"), "*.parquet")
OUT_DIR      = os.getenv("PREPARED_PATH", "data/prepared/")

db = duckdb.connect()
db.execute(f"CREATE VIEW jobs AS SELECT * FROM read_parquet('{PARQUET_GLOB}')")

# ── 1. Find the 80th percentile cutoff timestamp ─────────────
cutoff = db.execute("""
    SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (
        ORDER BY CAST(sdt AS TIMESTAMP WITH TIME ZONE)
    )
    FROM jobs
    WHERE sdt IS NOT NULL
""").fetchone()[0]
print(f"Train/test cutoff (80th pct of sdt): {cutoff}")

# ── 2. Build prepared dataset via DuckDB SQL ──────────────────
# Submission-time features + training labels only.
# perf1-6, flops, mbwidth, opint, nnumu etc. excluded (post-run).
# avgpcon clipped to [10, 200000] to remove sensor faults.
db.execute(f"CREATE OR REPLACE DIRECTORY '{OUT_DIR}'") if False else None

query = """
CREATE OR REPLACE TABLE prepared AS
SELECT
    -- ── Chronological split flag ──────────────────────────────
    CASE
        WHEN CAST(sdt AS TIMESTAMPTZ) <= CAST('{cutoff}' AS TIMESTAMPTZ)
        THEN 'train' ELSE 'test'
    END AS split,

    -- ── SUBMISSION-TIME FEATURES ──────────────────────────────
    -- Node / core counts (log-scaled — heavily right-skewed)
    LN(1 + nnumr)                    AS log_nnumr,
    LN(1 + nnuma)                    AS log_nnuma,
    LN(1 + cnumr)                    AS log_cnumr,

    -- Requested walltime (log-scaled)
    LN(1 + elpl)                     AS log_elpl,

    -- Memory (log-scaled)
    LN(1 + CASE WHEN mszl < 0 OR mszl > 1e18 THEN NULL ELSE mszl END) AS log_mszl,
    LN(1 + CASE WHEN CAST(msza AS DOUBLE) > 1e18 THEN NULL ELSE CAST(msza AS DOUBLE) END) AS log_msza,

    -- Job class (binary: 1=compute-bound, 0=memory-bound)
    CASE WHEN pclass = 'compute-bound' THEN 1 ELSE 0 END AS is_compute_bound,

    -- Frequency & priority (keep raw — already bounded)
    COALESCE(freq_req, 0)             AS freq_req,
    COALESCE(pri, 0)                  AS pri,

    -- jobenv_req (one-hot top-2, rest → 0)
    CASE WHEN jobenv_req = 'jobenv_req_0' THEN 1 ELSE 0 END AS jobenv_0,
    CASE WHEN jobenv_req = 'jobenv_req_1' THEN 1 ELSE 0 END AS jobenv_1,

    -- ── TRAINING LABELS ───────────────────────────────────────
    -- Failure (binary)
    CASE WHEN "exit state" = 'failed' THEN 1 ELSE 0 END AS failed,

    -- Failure type (for two-stage model)
    CASE
        WHEN "exit state" = 'completed'        THEN 0
        WHEN "exit state" = 'failed'
             AND duration < 300                  THEN 1   -- quick fail
        WHEN "exit state" = 'failed'
             AND duration BETWEEN 300 AND 7200   THEN 2   -- medium fail
        WHEN "exit state" = 'failed'
             AND duration > 7200                 THEN 3   -- slow fail (costly)
    END AS fail_type,

    -- Runtime (log-scaled)
    LN(1 + duration)                  AS log_duration,

    -- Energy — direct label, no approximation needed
    LN(1 + econ)                      AS log_econ,

    -- Power (clipped — remove sensor faults)
    CASE
        WHEN avgpcon BETWEEN 10 AND 200000 THEN avgpcon
        ELSE NULL
    END AS avgpcon_clean,

    -- Wasted node-hours (only meaningful for failed jobs)
    CASE
        WHEN "exit state" = 'failed'
        THEN nnumr * duration / 3600.0
        ELSE 0
    END AS wasted_node_hours

FROM jobs
WHERE sdt IS NOT NULL
  AND duration > 0
  AND elpl    > 0
""".format(cutoff=cutoff)

print("Building prepared table...")
db.execute(query)

# ── 3. Verify split sizes ─────────────────────────────────────
counts = db.execute("""
    SELECT split, COUNT(*) AS n,
           ROUND(AVG(failed)*100, 2) AS failure_pct
    FROM prepared
    GROUP BY split ORDER BY split DESC
""").fetchall()
print("\nSplit sizes:")
for r in counts:
    print(f"  {r[0]:<6}  rows={r[1]:>10,}  failure_rate={r[2]}%")

# ── 4. Verify no post-run leakage ────────────────────────────
cols = db.execute("DESCRIBE prepared").fetchall()
print(f"\nFinal columns ({len(cols)}):")
for c in cols:
    print(f"  {c[0]:<25} {c[1]}")

# ── 5. Export to parquet ──────────────────────────────────────
import os
os.makedirs(OUT_DIR, exist_ok=True)

print("\nExporting train.parquet...")
db.execute(f"""
    COPY (SELECT * FROM prepared WHERE split = 'train')
    TO '{OUT_DIR}train.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("Exporting test.parquet...")
db.execute(f"""
    COPY (SELECT * FROM prepared WHERE split = 'test')
    TO '{OUT_DIR}test.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("\nDone. Files written to:", OUT_DIR)
