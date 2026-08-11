'''
import duckdb, os, glob, time, pickle
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

PARQUET_GLOB = os.path.join(os.getenv("FUGAKU_DATA_PATH", "data/fugaku"), "*.parquet")
OUT_DIR      = os.getenv("PREPARED_PATH", "data/prepared/")
PCA_PATH     = OUT_DIR + "embedding_pca.pkl"
N_WORKERS    = 10   # concurrent storage reads — tune down if storage throttles

pca = pickle.load(open(PCA_PATH, "rb"))
N   = pca.n_components_
print(f"PCA: {pca.n_features_in_} → {N} components")

def process_file(fpath):
    db = duckdb.connect()   # each worker gets its own connection
    rows = db.execute(
        "SELECT jid, embedding FROM read_parquet('" + fpath + "')"
    ).fetchall()
    db.close()
    if not rows:
        return [], np.empty((0, pca.n_features_in_), dtype=np.float32)
    jids = [r[0] for r in rows]
    X    = np.array([r[1] for r in rows], dtype=np.float32)
    Z    = pca.transform(X).astype(np.float32)
    return jids, Z

parquet_files = sorted(glob.glob(PARQUET_GLOB))
print(f"\nStep 1: Processing {len(parquet_files)} files with {N_WORKERS} parallel workers...")
t0 = time.time()

all_jids  = []
all_embs  = []

with Pool(processes=N_WORKERS) as pool:
    for jids, Z in tqdm(
        pool.imap(process_file, parquet_files),
        total=len(parquet_files), unit="file", ncols=88, colour="cyan"
    ):
        all_jids.extend(jids)
        if Z.shape[0] > 0:
            all_embs.append(Z)

emb_array = np.vstack(all_embs)
del all_embs
print(f"  {len(all_jids)/1e6:.2f}M jobs  [{time.time()-t0:.1f}s]")

# merge into parquets
db = duckdb.connect()
db.execute("PRAGMA threads=30")
db.execute("PRAGMA memory_limit='24GB'")

print("\nStep 2: Building lookup table...")
emb_dict = {"jid": all_jids}
for i in range(N):
    emb_dict[f"emb_{i}"] = emb_array[:, i]
emb_df = pd.DataFrame(emb_dict)
db.register("emb_lookup_df", emb_df)
db.execute("CREATE OR REPLACE TABLE emb_lookup AS SELECT * FROM emb_lookup_df")
del emb_df, emb_array, all_jids

emb_col_sql = ", ".join([f"e.emb_{i}" for i in range(N)])
for split in ["train", "test"]:
    pq = f"{OUT_DIR}{split}_v3.parquet"
    print(f"\nStep 3: Merging into {split}_v3.parquet ...")
    t0 = time.time()
    db.execute(f"""
        CREATE OR REPLACE TABLE merged AS
        SELECT f.*, {emb_col_sql}
        FROM read_parquet('{pq}') f
        LEFT JOIN emb_lookup e ON f.jid = e.jid
    """)
    db.execute(f"COPY merged TO '{pq}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    print(f"  {os.path.getsize(pq)/1e6:.1f} MB  [{time.time()-t0:.1f}s]")

print("\nDone.")
'''

import duckdb, os, glob, time, pickle
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq_lib
from multiprocessing import Pool
from tqdm import tqdm

PARQUET_GLOB = os.path.join(os.getenv("FUGAKU_DATA_PATH", "data/fugaku"), "*.parquet")
OUT_DIR      = os.getenv("PREPARED_PATH", "data/prepared/")
PCA_PATH     = OUT_DIR + "embedding_pca.pkl"
CKPT_EMB     = OUT_DIR + "emb_array_ckpt.npy"
CKPT_JIDS    = OUT_DIR + "jids_ckpt.pkl"
N_WORKERS    = 10

pca = pickle.load(open(PCA_PATH, "rb"))
N   = pca.n_components_
print(f"PCA: {pca.n_features_in_} → {N} components")

# ── Step 1: parallel read + PCA transform (skip if checkpoint exists) ──
if os.path.exists(CKPT_EMB) and os.path.exists(CKPT_JIDS):
    print("\nStep 1: Loading checkpoint (skipping re-read)...")
    emb_array = np.load(CKPT_EMB)
    all_jids  = pickle.load(open(CKPT_JIDS, "rb"))
    print(f"  Loaded {len(all_jids)/1e6:.2f}M jobs  shape={emb_array.shape}")
else:
    def process_file(fpath):
        db   = duckdb.connect()
        rows = db.execute(
            "SELECT jid, embedding FROM read_parquet('" + fpath + "')"
        ).fetchall()
        db.close()
        if not rows:
            return [], np.empty((0, pca.n_features_in_), dtype=np.float32)
        jids = [r[0] for r in rows]
        X    = np.array([r[1] for r in rows], dtype=np.float32)
        return jids, pca.transform(X).astype(np.float32)

    parquet_files = sorted(glob.glob(PARQUET_GLOB))
    print(f"\nStep 1: Processing {len(parquet_files)} files ({N_WORKERS} workers)...")
    t0 = time.time()
    all_jids, all_embs = [], []
    with Pool(processes=N_WORKERS) as pool:
        for jids, Z in tqdm(pool.imap(process_file, parquet_files),
                            total=len(parquet_files), unit="file", ncols=88, colour="cyan"):
            all_jids.extend(jids)
            if Z.shape[0] > 0:
                all_embs.append(Z)

    emb_array = np.vstack(all_embs)
    del all_embs
    print(f"  {len(all_jids)/1e6:.2f}M jobs  [{time.time()-t0:.1f}s]")

    # ── save checkpoint so a Step 2 crash never costs another 5-min read ──
    print("  Saving checkpoint...")
    np.save(CKPT_EMB, emb_array)
    pickle.dump(all_jids, open(CKPT_JIDS, "wb"))
    print(f"  Saved: {CKPT_EMB} ({os.path.getsize(CKPT_EMB)/1e9:.1f} GB)")

# ── Step 2: jid → row index map ───────────────────────────────────
print("\nStep 2: Building jid index...")
t0 = time.time()
jid_to_idx = {jid: i for i, jid in enumerate(all_jids)}
print(f"  {len(jid_to_idx)/1e6:.2f}M entries  [{time.time()-t0:.1f}s]")

# ── Step 3: merge into parquets using pyarrow (zero temp disk) ────
for split in ["train", "test"]:
    pq = f"{OUT_DIR}{split}_v3.parquet"
    print(f"\nStep 3: Merging into {split}_v3.parquet ...")
    t0 = time.time()

    table  = pq_lib.read_table(pq)
    jids_s = table.column("jid").to_pylist()

    # fancy-index into emb_array — no copy, no temp disk
    idx = np.array([jid_to_idx.get(j, 0) for j in jids_s], dtype=np.int64)
    Z   = emb_array[idx].astype(np.float32)

    for i in range(N):
        table = table.append_column(f"emb_{i}", pa.array(Z[:, i]))
    del Z

    pq_lib.write_table(table, pq, compression="zstd")
    del table
    print(f"  {os.path.getsize(pq)/1e6:.0f} MB  [{time.time()-t0:.1f}s]")

# ── cleanup checkpoints ───────────────────────────────────────────
for f in [CKPT_EMB, CKPT_JIDS]:
    os.remove(f)
    print(f"  Removed checkpoint: {f}")

print("\nDone.")