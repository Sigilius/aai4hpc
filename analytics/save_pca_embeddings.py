"""
One-time utility: load 384-dim embeddings, fit PCA(32), save reduced embeddings
as a parquet file (jid, emb_0..emb_31). Future training runs load 3GB not 38GB.
Run once: python analytics/save_pca_embeddings.py
"""
import duckdb, numpy as np, pickle, os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

RAW_GLOB     = os.path.join(os.getenv("FUGAKU_DATA_PATH", "data/fugaku"), "*.parquet")
PREPARED_DIR = os.getenv("PREPARED_PATH", "data/prepared/")
MODEL_DIR    = os.getenv("MODELS_PATH", "models/")
CUTOFF       = "2023-08-28 00:37:43+00:00"
N_PCA        = 32
os.makedirs(PREPARED_DIR, exist_ok=True)

db = duckdb.connect(config={
    "temp_directory": os.getenv("DUCKDB_TMP", "/tmp/duckdb_tmp"),
    "max_temp_directory_size": "500GiB",
    "memory_limit": "160GB",
    "threads": 8,
})
print("query 1 executing")
db.execute(f"CREATE VIEW raw AS SELECT * FROM read_parquet('{RAW_GLOB}')")
print("query 1 executed")

print("Loading ALL jids and embeddings...")
result = db.execute("""
    SELECT jid, embedding FROM raw
    WHERE sdt IS NOT NULL AND duration > 0 AND elpl > 0
""").fetchnumpy()

jids = result["jid"]
print(f"  Loaded {len(jids):,} jids")

print("Stacking into 2D matrix (this takes a few minutes)...")
emb = np.stack(result["embedding"]).astype(np.float32)
print(f"  Shape: {emb.shape}  ({emb.nbytes/1e9:.1f}GB)")

# Fit PCA on training rows only to avoid leakage
print("Fitting PCA on training rows only...")
train_mask = db.execute(f"""
    SELECT jid FROM raw
    WHERE sdt IS NOT NULL AND duration > 0 AND elpl > 0
      AND CAST(sdt AS TIMESTAMPTZ) <= CAST('{CUTOFF}' AS TIMESTAMPTZ)
""").fetchnumpy()["jid"]
train_set  = set(train_mask.tolist())
train_idx  = np.array([i for i, j in enumerate(jids) if j in train_set])

scaler = StandardScaler()
pca    = PCA(n_components=N_PCA, random_state=42)
emb_train_scaled = scaler.fit_transform(emb[train_idx])
pca.fit(emb_train_scaled)
print(f"  Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# Save PCA bundle
with open(MODEL_DIR + "embedding_pca.pkl", "wb") as f:
    pickle.dump({"scaler": scaler, "pca": pca, "n_components": N_PCA}, f)
print("  Saved embedding_pca.pkl")

# Transform ALL rows and save as parquet
print("Transforming all rows and saving reduced embeddings...")
emb_reduced = pca.transform(scaler.transform(emb))  # (N, 32)
pca_cols    = [f"emb_{i}" for i in range(N_PCA)]

import pandas as pd
df_pca = pd.DataFrame(emb_reduced, columns=pca_cols)
df_pca.insert(0, "jid", jids)
df_pca.to_parquet(PREPARED_DIR + "embeddings_pca32.parquet", index=False, compression="zstd")
print(f"  Saved embeddings_pca32.parquet  ({os.path.getsize(PREPARED_DIR+'embeddings_pca32.parquet')/1e9:.2f}GB)")
print("Done. Future training runs load 3GB instead of 38GB.")
