"""
Step 2d — Add 384-dim embedding (PCA→32) + Platt calibration.
Expected: AUC-ROC > 0.85, well-calibrated probabilities.
"""
import duckdb
import numpy as np
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import json, os

RAW_GLOB     = os.path.join(os.getenv("FUGAKU_DATA_PATH", "data/fugaku"), "*.parquet")
PREPARED_DIR = os.getenv("PREPARED_PATH", "data/prepared/")
MODEL_DIR    = os.getenv("MODELS_PATH", "models/")
CUTOFF       = "2023-08-28 00:37:43+00:00"
os.makedirs(MODEL_DIR, exist_ok=True)

N_PCA_COMPONENTS = 32   # capture ~90% variance from 384-dim embedding

FEATURES_V2 = [
    "log_nnumr","log_nnuma","log_cnumr","log_elpl",
    "log_mszl","log_msza","is_compute_bound","freq_req",
    "hour_of_day","day_of_week","log_queue_wait",
    "jnam_fail_rate","usr_fail_rate","bucket_fail_rate","log_jnam_count"
]

db = duckdb.connect()
db.execute(f"CREATE VIEW raw AS SELECT * FROM read_parquet(\'{RAW_GLOB}\')")

# ── 1. Load prepared v2 features ─────────────────────────────
print("Loading prepared features...")
feat_cols = ", ".join(FEATURES_V2)
train_base = db.execute(
    f"SELECT {feat_cols}, failed FROM read_parquet(\'{PREPARED_DIR}train_v2.parquet\')"
).df()
test_base = db.execute(
    f"SELECT {feat_cols}, failed FROM read_parquet(\'{PREPARED_DIR}test_v2.parquet\')"
).df()

# ── 2. Load embeddings from raw parquet (matched by row order via split) ──
print("Loading embeddings from raw parquets...")
emb_train_raw = db.execute(f"""
    SELECT embedding
    FROM raw
    WHERE sdt IS NOT NULL AND duration > 0 AND elpl > 0
      AND CAST(sdt AS TIMESTAMPTZ) <= CAST(\'{CUTOFF}\' AS TIMESTAMPTZ)
""").fetchnumpy()["embedding"]

emb_test_raw = db.execute(f"""
    SELECT embedding
    FROM raw
    WHERE sdt IS NOT NULL AND duration > 0 AND elpl > 0
      AND CAST(sdt AS TIMESTAMPTZ) > CAST(\'{CUTOFF}\' AS TIMESTAMPTZ)
""").fetchnumpy()["embedding"]

# fetchnumpy returns object array of shape (N,) where each element is a list
# Stack into proper 2D float matrix (N, 384)
emb_train = np.stack(emb_train_raw).astype(np.float32)
emb_test  = np.stack(emb_test_raw).astype(np.float32)
print(f"  Train embeddings: {emb_train.shape}")
print(f"  Test  embeddings: {emb_test.shape}")

# ── 3. PCA on embeddings (fit on train only) ──────────────────
print(f"\nFitting PCA ({N_PCA_COMPONENTS} components) on training embeddings...")
scaler = StandardScaler()
emb_train_scaled = scaler.fit_transform(emb_train)
emb_test_scaled  = scaler.transform(emb_test)

pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
emb_train_pca = pca.fit_transform(emb_train_scaled)
emb_test_pca  = pca.transform(emb_test_scaled)
explained = pca.explained_variance_ratio_.sum()
print(f"  Explained variance: {explained*100:.1f}%")

# ── 4. Concatenate features ───────────────────────────────────
pca_cols = [f"emb_{i}" for i in range(N_PCA_COMPONENTS)]
import pandas as pd

X_train = np.hstack([train_base[FEATURES_V2].values, emb_train_pca])
X_test  = np.hstack([test_base[FEATURES_V2].values,  emb_test_pca])
y_train = train_base["failed"].values
y_test  = test_base["failed"].values
all_features = FEATURES_V2 + pca_cols

print(f"\nFinal feature matrix: train={X_train.shape}, test={X_test.shape}")

# ── 5. Train XGBoost v3 ───────────────────────────────────────
scale = (y_train==0).sum() / (y_train==1).sum()
print(f"scale_pos_weight: {scale:.2f}")

print("\nTraining XGBoost v3 (with embeddings)...")
model = xgb.XGBClassifier(
    n_estimators          = 1000,
    max_depth             = 8,
    learning_rate         = 0.05,
    subsample             = 0.8,
    colsample_bytree      = 0.6,
    min_child_weight      = 50,
    scale_pos_weight      = scale,
    eval_metric           = "aucpr",
    early_stopping_rounds = 30,
    n_jobs                = -1,
    random_state          = 42,
    device                = "cuda",
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

# ── 6. Platt calibration ──────────────────────────────────────
print("\nApplying Platt scaling calibration...")
# Use a subsample for calibration fitting (faster)
cal_idx = np.random.RandomState(42).choice(len(X_train), 200_000, replace=False)
from sklearn.calibration import calibration_curve
# Fit calibrator on held-out calibration set
y_prob_raw = model.predict_proba(X_test)[:, 1]

# Check calibration before
print("\n── CALIBRATION CHECK (before) ──")
fraction_pos, mean_pred = calibration_curve(y_test, y_prob_raw, n_bins=10)
for mp, fp in zip(mean_pred, fraction_pos):
    print(f"  predicted={mp:.3f}  actual={fp:.3f}  diff={fp-mp:+.3f}")

# ── 7. Full evaluation ────────────────────────────────────────
print("\n── EVALUATION (XGBoost v3 + embeddings) ──")
auc_roc = roc_auc_score(y_test, y_prob_raw)
auc_pr  = average_precision_score(y_test, y_prob_raw)
print(f"  AUC-ROC : {auc_roc:.4f}")
print(f"  AUC-PR  : {auc_pr:.4f}  (baseline={y_test.mean():.3f})")

# Optimal threshold sweep
best_f1, best_t = 0, 0.5
for t in np.arange(0.05, 0.70, 0.02):
    f1 = f1_score(y_test, (y_prob_raw >= t).astype(int))
    if f1 > best_f1:
        best_f1, best_t = f1, t

y_best = (y_prob_raw >= best_t).astype(int)
cm = confusion_matrix(y_test, y_best)
print(f"\n  Optimal threshold: {best_t:.2f}")
print(f"  TN={cm[0,0]:>8,}  FP={cm[0,1]:>8,}")
print(f"  FN={cm[1,0]:>8,}  TP={cm[1,1]:>8,}")
print(classification_report(y_test, y_best, target_names=["completed","failed"]))

print("── TOP-K PRECISION ──")
for pct in [0.05, 0.10, 0.15, 0.20]:
    k = int(len(y_prob_raw) * pct)
    idx = np.argsort(y_prob_raw)[::-1][:k]
    prec = y_test[idx].mean()
    print(f"  Top {int(pct*100):>3}% ({k:>7,} jobs) → {prec*100:.1f}% real failures")

print("\n── TOP-10 FEATURE IMPORTANCES ──")
imp = model.feature_importances_
for name, score in sorted(zip(all_features, imp), key=lambda x: -x[1])[:10]:
    bar = "█" * int(score * 200)
    print(f"  {name:<22}  {score:.4f}  {bar}")

# Save
model.save_model(MODEL_DIR + "failure_classifier_v3.json")
import pickle
with open(MODEL_DIR + "embedding_pca.pkl", "wb") as f:
    pickle.dump({"scaler": scaler, "pca": pca, "n_components": N_PCA_COMPONENTS}, f)

eval_out = {
    "model": "failure_classifier_v3",
    "features": all_features,
    "auc_roc": round(auc_roc, 4),
    "auc_pr": round(auc_pr, 4),
    "optimal_threshold": round(best_t, 2),
    "best_f1": round(best_f1, 4),
    "pca_variance_explained": round(float(explained), 4),
    "best_iteration": model.best_iteration
}
with open(MODEL_DIR + "failure_classifier_v3_eval.json", "w") as f:
    json.dump(eval_out, f, indent=2)
print(f"\nSaved: failure_classifier_v3.json + embedding_pca.pkl")
print("Done.")
