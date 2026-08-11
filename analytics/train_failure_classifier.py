"""
Step 2 — Train XGBoost failure classifier on Fugaku prepared data.
Input:  <PREPARED_PATH>/train.parquet
        <PREPARED_PATH>/test.parquet
Output: <MODELS_PATH>/failure_classifier.json
        <MODELS_PATH>/failure_classifier_eval.txt
"""
import duckdb
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
import json, os

PREPARED_DIR = os.getenv("PREPARED_PATH", "data/prepared/")
MODEL_DIR    = os.getenv("MODELS_PATH", "models/")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "log_nnumr", "log_nnuma", "log_cnumr",
    "log_elpl",
    "log_mszl", "log_msza",
    "is_compute_bound",
    "freq_req", "pri",
    "jobenv_0", "jobenv_1"
]
TARGET = "failed"

# ── 1. Load data ──────────────────────────────────────────────
print("Loading train/test parquets...")
db = duckdb.connect()

train = db.execute(f"""
    SELECT {", ".join(FEATURES)}, {TARGET}
    FROM read_parquet('{PREPARED_DIR}train.parquet')
""").df()

test = db.execute(f"""
    SELECT {", ".join(FEATURES)}, {TARGET}
    FROM read_parquet('{PREPARED_DIR}test.parquet')
""").df()

X_train = train[FEATURES].values
y_train = train[TARGET].values
X_test  = test[FEATURES].values
y_test  = test[TARGET].values

print(f"  Train: {X_train.shape}  pos={y_train.sum():,} ({100*y_train.mean():.1f}%)")
print(f"  Test:  {X_test.shape}   pos={y_test.sum():,} ({100*y_test.mean():.1f}%)")

# ── 2. Class imbalance weight ─────────────────────────────────
# Test has 13.4% failure vs train 9% — use scale_pos_weight from train
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale = neg / pos
print(f"\nscale_pos_weight: {scale:.2f}  (neg={neg:,} / pos={pos:,})")

# ── 3. Train XGBoost ──────────────────────────────────────────
print("\nTraining XGBoost classifier...")
model = xgb.XGBClassifier(
    n_estimators      = 500,
    max_depth         = 6,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = scale,
    eval_metric       = "aucpr",
    early_stopping_rounds = 20,
    n_jobs            = -1,
    random_state      = 42,
    device            = "cuda",        # uses your GPU
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# ── 4. Evaluate ───────────────────────────────────────────────
print("\n── EVALUATION ──")
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

auc_roc = roc_auc_score(y_test, y_prob)
auc_pr  = average_precision_score(y_test, y_prob)
print(f"  AUC-ROC:  {auc_roc:.4f}")
print(f"  AUC-PR:   {auc_pr:.4f}   (random baseline = {y_test.mean():.3f})")

print("\nClassification report (threshold=0.5):")
print(classification_report(y_test, y_pred, target_names=["completed", "failed"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(f"  TN={cm[0,0]:>8,}  FP={cm[0,1]:>8,}")
print(f"  FN={cm[1,0]:>8,}  TP={cm[1,1]:>8,}")

# ── 5. Feature importance ─────────────────────────────────────
print("\n── FEATURE IMPORTANCE ──")
imp = model.feature_importances_
for name, score in sorted(zip(FEATURES, imp), key=lambda x: -x[1]):
    bar = "█" * int(score * 300)
    print(f"  {name:<20}  {score:.4f}  {bar}")

# ── 6. Save model ─────────────────────────────────────────────
model_path = MODEL_DIR + "failure_classifier.json"
model.save_model(model_path)
print(f"\nModel saved: {model_path}")

# Save eval summary as JSON for the PA agent to reference
eval_summary = {
    "model": "failure_classifier",
    "features": FEATURES,
    "auc_roc": round(auc_roc, 4),
    "auc_pr":  round(auc_pr, 4),
    "train_failure_rate": round(float(y_train.mean()), 4),
    "test_failure_rate":  round(float(y_test.mean()), 4),
    "scale_pos_weight":   round(float(scale), 2),
    "n_estimators_used":  model.best_iteration
}
with open(MODEL_DIR + "failure_classifier_eval.json", "w") as f:
    json.dump(eval_summary, f, indent=2)
print("Eval summary saved.")
print("\nDone.")
