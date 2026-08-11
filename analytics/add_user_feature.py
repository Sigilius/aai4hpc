"""
add_user_feature.py
Rebuilds train.parquet / test.parquet from original Fugaku data,
adding user_fail_rate (Laplace-smoothed historical failure rate per user).
One-time run — output replaces the existing prepared files.
~5-7 min on 25M rows.
"""
import duckdb, os, time

PARQUET_GLOB = os.path.join(os.getenv("FUGAKU_DATA_PATH", "data/fugaku"), "*.parquet")
OUT_DIR      = os.getenv("PREPARED_PATH", "data/prepared/")
CUTOFF       = "2023-08-28 00:37:43+00:00"   # 80th pct sdt from prepare_data.py

db = duckdb.connect()
db.execute("PRAGMA threads=8")
db.execute("PRAGMA memory_limit='14GB'")

# ── Step 1: user failure rates from TRAINING period only ─────────
# Laplace smoothing: (n_failed + 1) / (n_jobs + 10)
# → for users with <10 jobs this shrinks toward 0.10 (global rate)
print("Computing user failure rates from training period...")
t0 = time.time()
db.execute(f"""
CREATE OR REPLACE TABLE user_stats AS
SELECT
    usr,
    COUNT(*)                                                AS n_jobs,
    SUM(CASE WHEN "exit state" = 'failed' THEN 1 ELSE 0 END) AS n_failed,
    ROUND(
        (SUM(CASE WHEN "exit state" = 'failed' THEN 1 ELSE 0 END) + 1.0)
        / (COUNT(*) + 10.0), 6
    ) AS user_fail_rate
FROM read_parquet('{PARQUET_GLOB}')
WHERE sdt IS NOT NULL
  AND CAST(sdt AS TIMESTAMPTZ) <= CAST('{CUTOFF}' AS TIMESTAMPTZ)
GROUP BY usr
""")
n_users = db.execute("SELECT COUNT(*) FROM user_stats").fetchone()[0]
print(f"  {n_users:,} unique users indexed  [{time.time()-t0:.1f}s]")

# Sanity check: rate distribution
rows = db.execute("""
    SELECT PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY user_fail_rate) AS median,
           PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY user_fail_rate) AS p95,
           MAX(user_fail_rate) AS max,
           COUNT(*) FILTER (WHERE n_jobs >= 100) AS power_users
    FROM user_stats
""").fetchone()
print(f"  Median rate: {rows[0]:.4f}  P95: {rows[1]:.4f}  "
      f"Max: {rows[2]:.4f}  Power users (≥100 jobs): {rows[3]:,}")

# ── Step 2: full feature engineering SQL with user join ──────────
# Identical to prepare_data.py but adds user_fail_rate column.
# GLOBAL_FAIL_RATE used as fallback for unseen users in test set.
GLOBAL_RATE = 0.0989

FEAT_QUERY = f"""
CREATE OR REPLACE TABLE prepared_v2 AS
SELECT
    CASE
        WHEN CAST(sdt AS TIMESTAMPTZ) <= CAST('{CUTOFF}' AS TIMESTAMPTZ)
        THEN 'train' ELSE 'test'
    END AS split,

    -- submission-time features
    LN(1 + nnumr)                                    AS log_nnumr,
    LN(1 + nnuma)                                    AS log_nnuma,
    LN(1 + cnumr)                                    AS log_cnumr,
    LN(1 + elpl)                                     AS log_elpl,
    LN(1 + CASE WHEN mszl < 0 OR mszl > 1e18
               THEN NULL ELSE mszl END)              AS log_mszl,
    LN(1 + CASE WHEN CAST(msza AS DOUBLE) > 1e18
               THEN NULL ELSE CAST(msza AS DOUBLE)
               END)                                  AS log_msza,
    CASE WHEN pclass = 'compute-bound' THEN 1 ELSE 0 END AS is_compute_bound,
    COALESCE(freq_req, 0)                            AS freq_req,
    COALESCE(pri, 0)                                 AS pri,
    CASE WHEN jobenv_req = 'jobenv_req_0' THEN 1 ELSE 0 END AS jobenv_0,
    CASE WHEN jobenv_req = 'jobenv_req_1' THEN 1 ELSE 0 END AS jobenv_1,

    -- ★ user historical failure rate (Laplace smoothed) ★
    COALESCE(u.user_fail_rate, {GLOBAL_RATE}) AS user_fail_rate,

    -- labels
    CASE WHEN "exit state" = 'failed' THEN 1 ELSE 0 END AS failed,
    CASE
        WHEN "exit state" = 'completed'                      THEN 0
        WHEN "exit state" = 'failed' AND duration < 300      THEN 1
        WHEN "exit state" = 'failed' AND duration <= 7200    THEN 2
        WHEN "exit state" = 'failed' AND duration > 7200     THEN 3
    END AS fail_type,
    LN(1 + duration)                                 AS log_duration,
    LN(1 + econ)                                     AS log_econ,
    CASE WHEN avgpcon BETWEEN 10 AND 200000
         THEN avgpcon ELSE NULL END                  AS avgpcon_clean,
    nnumr * duration / 3600.0                        AS wasted_node_hours

FROM read_parquet('{PARQUET_GLOB}') j
LEFT JOIN user_stats u ON j.usr = u.usr
WHERE j.sdt IS NOT NULL
  AND j.duration > 0
  AND j.elpl    > 0
"""

print("\nBuilding feature table with user stats joined...")
t0 = time.time()
db.execute(FEAT_QUERY)
print(f"  Done [{time.time()-t0:.1f}s]")

# ── Step 3: verify ───────────────────────────────────────────────
counts = db.execute("""
    SELECT split, COUNT(*) AS n,
           ROUND(AVG(failed)*100,2)           AS fail_pct,
           ROUND(AVG(user_fail_rate),4)       AS avg_user_rate
    FROM prepared_v2
    GROUP BY split ORDER BY split DESC
""").fetchall()
print("\nSplit summary:")
for r in counts:
    print(f"  {r[0]:<6}  rows={r[1]:>10,}  "
          f"fail%={r[2]}  avg_user_fail_rate={r[3]}")

# ── Step 4: export ───────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
for split in ["train", "test"]:
    out = f"{OUT_DIR}{split}_v2.parquet"
    print(f"\nExporting {split}_v2.parquet...")
    t0 = time.time()
    db.execute(f"""
        COPY (SELECT * FROM prepared_v2 WHERE split = '{split}')
        TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"  Written: {out}  [{time.time()-t0:.1f}s]")

print("\nDone.")
