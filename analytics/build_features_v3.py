"""
build_features_v3.py
Set TEST_MODE = True to validate on 3 files before the full 38-file run.
"""
import duckdb, os, glob, time, pickle
import numpy as np
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────
TEST_MODE      = False   # ← flip to False for full run
PARQUET_GLOB   = os.path.join(os.getenv("FUGAKU_DATA_PATH", "data/fugaku"), "*.parquet")
OUT_DIR        = os.getenv("PREPARED_PATH", "data/prepared/")
CUTOFF         = "2023-08-28 00:37:43+00:00"
GLOBAL_RATE    = 0.0989
EMB_COMPONENTS = 64

# ── SETUP ─────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
db = duckdb.connect()
db.execute("PRAGMA threads=30")
db.execute("PRAGMA memory_limit='16GB'")

all_files = sorted(glob.glob(PARQUET_GLOB))
pq_files  = all_files[:3] if TEST_MODE else all_files
# DuckDB list-of-files syntax — works for both 3 and 38 files
files_arg = "['" + "', '".join(pq_files) + "']"
print(f"{'[TEST MODE]' if TEST_MODE else '[FULL RUN]'} "
      f"{len(pq_files)}/{len(all_files)} files")

# ── STEP 1: embedding dimension ────────────────────────────────────
print("\nStep 1: Confirming embedding dimension...")
t0 = time.time()
emb_dim = db.execute(
    "SELECT len(ANY_VALUE(embedding)) FROM read_parquet(" + files_arg + ")"
).fetchone()[0]
n_components = min(EMB_COMPONENTS, emb_dim)
print(f"  dim={emb_dim}  →  {n_components} PCA components  [{time.time()-t0:.1f}s]")

# STEP 2: PCA — load if already fitted, skip the 45-min refit
pca_path = OUT_DIR + ("embedding_pca_test.pkl" if TEST_MODE else "embedding_pca.pkl")

if os.path.exists(pca_path):
    print(f"\nStep 2: Loading existing PCA from {pca_path} ...")
    pca = pickle.load(open(pca_path, "rb"))
    var_exp = pca.explained_variance_ratio_.sum()
    print(f"  n_components={pca.n_components_}  variance_explained={var_exp:.3f}  [skipped refit]")
else:
    print("\nStep 2: Fitting PCA (file not found, running full fit)...")
    print("\nStep 2: Fitting PCA on ~3M random training rows...")
    t0 = time.time()
    import threading

    result_holder = [None]
    error_holder  = [None]

    def run_query():
        try:
            result_holder[0] = db.execute("""
                SELECT embedding
                FROM read_parquet(""" + files_arg + """)
                WHERE embedding IS NOT NULL
                AND CAST(sdt AS TIMESTAMPTZ) <= CAST('""" + CUTOFF + """' AS TIMESTAMPTZ)
                USING SAMPLE 3000000 ROWS (reservoir, 42)
            """).fetchall()
        except Exception as e:
            error_holder[0] = e

    t = threading.Thread(target=run_query)
    t.start()

    # spinner in main thread while DuckDB works
    with tqdm(desc="  sampling embeddings", unit="s", bar_format=
            "{desc}  [{elapsed}]  {postfix}", ncols=60, colour="cyan") as pbar:
        while t.is_alive():
            t.join(timeout=1.0)
            pbar.update(1)
            pbar.set_postfix_str("DuckDB reservoir sampling...")
        pbar.set_postfix_str("done")

    if error_holder[0]:
        raise error_holder[0]

    rows    = result_holder[0]
    X_sample = np.array([r[0] for r in rows], dtype=np.float32)
    print(f"  Sample shape: {X_sample.shape}  [{time.time()-t0:.1f}s]")

    from sklearn.decomposition import PCA
    t1 = time.time()

    # PCA fit also gets a spinner
    pca = [None]
    def run_pca():
        pca[0] = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
        pca[0].fit(X_sample)

    tp = threading.Thread(target=run_pca)
    tp.start()
    with tqdm(desc="  fitting PCA", unit="s", bar_format=
            "{desc}  [{elapsed}]  {postfix}", ncols=60, colour="green") as pbar:
        while tp.is_alive():
            tp.join(timeout=1.0)
            pbar.update(1)
            pbar.set_postfix_str("randomized SVD...")
        pbar.set_postfix_str("done")

    pca = pca[0]
    del X_sample
    var_exp = pca.explained_variance_ratio_.sum()
    print(f"  PCA fitted  [{time.time()-t1:.1f}s]")
    print(f"  Variance explained: {var_exp:.3f}")
    pca_path = OUT_DIR + ("embedding_pca_test.pkl" if TEST_MODE else "embedding_pca.pkl")
    pickle.dump(pca, open(pca_path, "wb"))
    print(f"  Saved: {pca_path}")

# ── STEP 3: rich user statistics ──────────────────────────────────
print("\nStep 3: User statistics...")
t0 = time.time()
db.execute("""
CREATE OR REPLACE TABLE user_stats AS
WITH train_jobs AS (
    SELECT usr, pclass,
           CASE WHEN "exit state"='failed' THEN 1 ELSE 0 END AS failed,
           LN(1 + nnumr) AS log_nnumr,
           LN(1 + elpl)  AS log_elpl,
           ROW_NUMBER() OVER (PARTITION BY usr ORDER BY CAST(sdt AS TIMESTAMPTZ)) AS rn,
           COUNT(*) OVER (PARTITION BY usr) AS user_total
    FROM read_parquet(""" + files_arg + """)
    WHERE CAST(sdt AS TIMESTAMPTZ) <= CAST('""" + CUTOFF + """' AS TIMESTAMPTZ)
      AND sdt IS NOT NULL AND duration > 0 AND elpl > 0
),
recent AS (
    SELECT usr, AVG(failed) AS recent_fail_rate
    FROM train_jobs WHERE rn > user_total - 20
    GROUP BY usr
),
by_class AS (
    SELECT usr,
        AVG(CASE WHEN pclass='compute-bound' THEN CAST(failed AS DOUBLE) END) AS compute_fail_rate,
        AVG(CASE WHEN pclass='memory-bound'  THEN CAST(failed AS DOUBLE) END) AS memory_fail_rate,
        AVG(log_nnumr) AS avg_log_nnumr,
        AVG(log_elpl)  AS avg_log_elpl,
        COUNT(*)       AS n_jobs,
        SUM(failed)    AS n_failed
    FROM train_jobs GROUP BY usr
)
SELECT
    b.usr,
    b.n_jobs,
    ROUND((b.n_failed + 1.0) / (b.n_jobs + 10.0), 6)              AS user_fail_rate,
    ROUND(COALESCE(b.compute_fail_rate, """ + str(GLOBAL_RATE) + """), 6) AS user_compute_fail_rate,
    ROUND(COALESCE(b.memory_fail_rate,  """ + str(GLOBAL_RATE) + """), 6) AS user_memory_fail_rate,
    ROUND(COALESCE(r.recent_fail_rate,  """ + str(GLOBAL_RATE) + """), 6) AS user_recent_fail_rate,
    ROUND(b.avg_log_nnumr, 6) AS user_avg_log_nnumr,
    ROUND(b.avg_log_elpl,  6) AS user_avg_log_elpl
FROM by_class b
LEFT JOIN recent r ON b.usr = r.usr
""")
n_users = db.execute("SELECT COUNT(*) FROM user_stats").fetchone()[0]
print(f"  {n_users:,} users  [{time.time()-t0:.1f}s]")

# ── STEP 4: feature table ──────────────────────────────────────────
print("\nStep 4: Building feature table...")
t0 = time.time()
GR = str(GLOBAL_RATE)
db.execute("""
CREATE OR REPLACE TABLE prepared_v3 AS
SELECT
    CASE WHEN CAST(j.sdt AS TIMESTAMPTZ) <= CAST('""" + CUTOFF + """' AS TIMESTAMPTZ)
         THEN 'train' ELSE 'test' END                    AS split,
    LN(1 + j.nnumr)                                      AS log_nnumr,
    LN(1 + j.nnuma)                                      AS log_nnuma,
    LN(1 + j.cnumr)                                      AS log_cnumr,
    LN(1 + j.elpl)                                       AS log_elpl,
    LN(1 + CASE WHEN j.mszl < 0 OR j.mszl > 1e18
                THEN NULL ELSE j.mszl END)               AS log_mszl,
    LN(1 + CASE WHEN CAST(j.msza AS DOUBLE) > 1e18
                THEN NULL ELSE CAST(j.msza AS DOUBLE) END) AS log_msza,
    CASE WHEN j.pclass='compute-bound' THEN 1.0 ELSE 0.0 END AS is_compute_bound,
    COALESCE(j.freq_req, 0.0)                            AS freq_req,
    COALESCE(j.pri, 0.0)                                 AS pri,
    CASE WHEN j.jobenv_req='jobenv_req_0' THEN 1.0 ELSE 0.0 END AS jobenv_0,
    CASE WHEN j.jobenv_req='jobenv_req_1' THEN 1.0 ELSE 0.0 END AS jobenv_1,
    CAST(EXTRACT(HOUR FROM CAST(j.qdt AS TIMESTAMPTZ)) AS FLOAT) AS hour_of_day,
    CAST(EXTRACT(DOW  FROM CAST(j.qdt AS TIMESTAMPTZ)) AS FLOAT) AS day_of_week,
    CASE WHEN EXTRACT(DOW FROM CAST(j.qdt AS TIMESTAMPTZ)) IN (0,6)
         THEN 1.0 ELSE 0.0 END                           AS is_weekend,
    (LN(1+j.elpl) - LN(1+j.nnumr))                      AS elpl_per_node,
    CASE WHEN j.pclass='compute-bound'
         THEN LN(1+j.nnumr) ELSE 0.0 END                AS compute_x_nodes,
    CAST(CASE WHEN LN(1+j.nnumr) < 0.693 THEN 0
              WHEN LN(1+j.nnumr) < 3.497 THEN 1
              WHEN LN(1+j.nnumr) < 6.238 THEN 2
              ELSE 3 END AS FLOAT)                       AS node_bucket,
    COALESCE(u.user_fail_rate,         """ + GR + """)   AS user_fail_rate,
    COALESCE(u.user_compute_fail_rate, """ + GR + """)   AS user_compute_fail_rate,
    COALESCE(u.user_memory_fail_rate,  """ + GR + """)   AS user_memory_fail_rate,
    COALESCE(u.user_recent_fail_rate,  """ + GR + """)   AS user_recent_fail_rate,
    COALESCE(u.user_avg_log_nnumr,     LN(2.0))          AS user_avg_log_nnumr,
    COALESCE(u.user_avg_log_elpl,      LN(7201.0))       AS user_avg_log_elpl,
    COALESCE(CAST(u.n_jobs AS FLOAT),  0.0)              AS user_n_jobs,
    (LN(1+j.nnumr) - COALESCE(u.user_avg_log_nnumr, LN(2.0))) AS nnumr_anomaly,
    CASE WHEN j."exit state"='failed' THEN 1 ELSE 0 END  AS failed,
    CASE WHEN j."exit state"='completed'                  THEN 0
         WHEN j."exit state"='failed' AND j.duration <  300  THEN 1
         WHEN j."exit state"='failed' AND j.duration <= 7200 THEN 2
         WHEN j."exit state"='failed' AND j.duration >  7200 THEN 3 END AS fail_type,
    LN(1 + j.duration)                                   AS log_duration,
    LN(1 + j.econ)                                       AS log_econ,
    CASE WHEN j.avgpcon BETWEEN 10 AND 200000
         THEN j.avgpcon ELSE NULL END                    AS avgpcon_clean,
    j.nnumr * j.duration / 3600.0                        AS wasted_node_hours,
    j.jid
FROM read_parquet(""" + files_arg + """) j
LEFT JOIN user_stats u ON j.usr = u.usr
WHERE j.sdt IS NOT NULL AND j.duration > 0 AND j.elpl > 0
""")
print(f"  Feature table built [{time.time()-t0:.1f}s]")

counts = db.execute("""
    SELECT split, COUNT(*) AS n, ROUND(AVG(failed)*100,2) AS fail_pct
    FROM prepared_v3 GROUP BY split ORDER BY split DESC
""").fetchall()
for r in counts:
    print(f"  {r[0]:<6}  rows={r[1]:>8,}  fail%={r[2]}")

# ── STEP 5: export ─────────────────────────────────────────────────
suffix = "_test" if TEST_MODE else ""
for split in ["train", "test"]:
    out = f"{OUT_DIR}{split}_v3{suffix}.parquet"
    print(f"\nExporting {out} ...")
    t0 = time.time()
    db.execute(f"""
        COPY (SELECT * FROM prepared_v3 WHERE split='{split}')
        TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    sz = os.path.getsize(out) / 1e6
    print(f"  {sz:.1f} MB  [{time.time()-t0:.1f}s]")

print("\nDone. Set TEST_MODE=False for full run.")