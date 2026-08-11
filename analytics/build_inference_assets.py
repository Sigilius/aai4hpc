"""
build_inference_assets.py
Saves two lookup files needed by the predictor at inference time:
  1. user_stats.parquet  — per-user feature values (from original parquets)
  2. jnam_emb.pkl        — jnam → mean PCA embedding (64-dim)
Run once after embed_and_merge.py. ~7-10 min total.
"""
import os, glob, pickle, time
import numpy as np
import duckdb
from multiprocessing import Pool
from tqdm import tqdm

ORIG_GLOB = os.path.join(os.getenv("FUGAKU_DATA_PATH", "data/fugaku"), "*.parquet")
MODEL_DIR = os.getenv("MODELS_PATH", "models/")
PCA_PATH  = os.path.join(os.getenv("PREPARED_PATH", "data/prepared/"), "embedding_pca.pkl")
CUTOFF    = "2023-08-28 00:37:43+00:00"
GR        = "0.0989"
N_WORKERS = 10

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Step 1: user stats from original parquets (~2 min) ────────────
print("Step 1: Building user_stats.parquet ...")
t0 = time.time()
db = duckdb.connect()
db.execute("PRAGMA threads=30")

db.execute("""
CREATE OR REPLACE TABLE user_stats AS
WITH train_jobs AS (
    SELECT usr, pclass,
           CASE WHEN "exit state" = 'failed' THEN 1 ELSE 0 END AS failed,
           LN(1 + nnumr) AS log_nnumr,
           LN(1 + elpl)  AS log_elpl,
           ROW_NUMBER() OVER (PARTITION BY usr ORDER BY CAST(sdt AS TIMESTAMPTZ)) AS rn,
           COUNT(*)       OVER (PARTITION BY usr) AS user_total
    FROM read_parquet('""" + ORIG_GLOB + """')
    WHERE CAST(sdt AS TIMESTAMPTZ) <= CAST('""" + CUTOFF + """' AS TIMESTAMPTZ)
      AND sdt IS NOT NULL AND duration > 0 AND elpl > 0
),
recent AS (
    SELECT usr, AVG(failed) AS recent_fail_rate
    FROM train_jobs
    WHERE rn > user_total - 20
    GROUP BY usr
),
by_class AS (
    SELECT usr,
        AVG(CASE WHEN pclass = 'compute-bound' THEN CAST(failed AS DOUBLE) END) AS compute_fail_rate,
        AVG(CASE WHEN pclass = 'memory-bound'  THEN CAST(failed AS DOUBLE) END) AS memory_fail_rate,
        AVG(log_nnumr) AS avg_log_nnumr,
        AVG(log_elpl)  AS avg_log_elpl,
        COUNT(*)       AS n_jobs,
        SUM(failed)    AS n_failed
    FROM train_jobs
    GROUP BY usr
)
SELECT
    b.usr,
    ROUND((b.n_failed + 1.0) / (b.n_jobs + 10.0), 6)         AS user_fail_rate,
    ROUND(COALESCE(b.compute_fail_rate, """ + GR + """), 6)   AS user_compute_fail_rate,
    ROUND(COALESCE(b.memory_fail_rate,  """ + GR + """), 6)   AS user_memory_fail_rate,
    ROUND(COALESCE(r.recent_fail_rate,  """ + GR + """), 6)   AS user_recent_fail_rate,
    ROUND(b.avg_log_nnumr, 6)                                  AS user_avg_log_nnumr,
    ROUND(b.avg_log_elpl,  6)                                  AS user_avg_log_elpl,
    CAST(b.n_jobs AS FLOAT)                                    AS user_n_jobs
FROM by_class b
LEFT JOIN recent r ON b.usr = r.usr
""")

db.execute(
    "COPY user_stats TO '" + MODEL_DIR + "user_stats.parquet' "
    "(FORMAT PARQUET, COMPRESSION ZSTD)"
)
n = db.execute("SELECT COUNT(*) FROM user_stats").fetchone()[0]
print(f"  {n:,} users saved  [{time.time()-t0:.1f}s]")
db.close()

# ── Step 2: jnam → mean PCA embedding (~5-8 min, parallel) ────────
print("\nStep 2: Building jnam → PCA embedding lookup ...")
pca = pickle.load(open(PCA_PATH, "rb"))

def read_jnam_embs(fpath):
    """Aggregate inside worker — returns small {jnam: 64-float vec} dict."""
    db = duckdb.connect()
    rows = db.execute(
        "SELECT jnam, embedding FROM read_parquet('" + fpath + "') "
        "WHERE embedding IS NOT NULL AND jnam IS NOT NULL "
        "AND CAST(sdt AS TIMESTAMPTZ) <= CAST('" + CUTOFF + "' AS TIMESTAMPTZ)"
    ).fetchall()
    db.close()
    if not rows:
        return {}

    local = {}
    for jnam, emb in rows:
        if jnam not in local:
            local[jnam] = []
        local[jnam].append(emb)

    names       = list(local.keys())
    vecs        = np.array([np.mean(local[n], axis=0) for n in names], dtype=np.float32)
    transformed = pca.transform(vecs).astype(np.float32)
    return {name: vec for name, vec in zip(names, transformed)}

parquet_files = sorted(glob.glob(ORIG_GLOB))
print(f"  {len(parquet_files)} files  |  {N_WORKERS} parallel workers")
t0 = time.time()

jnam_pca = {}
with Pool(processes=N_WORKERS) as pool:
    for local_dict in tqdm(pool.imap(read_jnam_embs, parquet_files),
                           total=len(parquet_files), unit="file", ncols=80):
        for jnam, vec in local_dict.items():
            if jnam not in jnam_pca:
                jnam_pca[jnam] = vec   # first training-period occurrence wins

out_path = MODEL_DIR + "jnam_emb.pkl"
pickle.dump(jnam_pca, open(out_path, "wb"))
sz = os.path.getsize(out_path) / 1e6
print(f"  {len(jnam_pca):,} unique job names  |  {sz:.1f} MB  [{time.time()-t0:.1f}s]")

print("\nDone. Assets saved to:", MODEL_DIR)
