import pickle, json, numpy as np, duckdb
from sklearn.metrics import *
from sklearn.isotonic import IsotonicRegression

import os
MODEL_DIR = os.getenv("MODELS_PATH", "models/")
_PREPARED = os.getenv("PREPARED_PATH", "data/prepared/")
TRAIN_PQ  = os.path.join(_PREPARED, "train_v3.parquet")
TEST_PQ   = os.path.join(_PREPARED, "test_v3.parquet")

db = duckdb.connect()
db.execute("PRAGMA threads=30")

m1  = pickle.load(open(MODEL_DIR + "s1_failure.pkl",    "rb"))
iso = pickle.load(open(MODEL_DIR + "s1_calibrator.pkl", "rb"))

# discover features same way as training script
schema = db.execute("SELECT * FROM read_parquet('" + TEST_PQ + "') LIMIT 0").description
LABEL_COLS = {"failed","fail_type","log_duration","log_econ",
              "avgpcon_clean","wasted_node_hours","split","jid"}
FEATURES = [col[0] for col in schema if col[0] not in LABEL_COLS]

raw  = db.execute("SELECT " + ",".join(FEATURES) +
                  ", failed, fail_type FROM read_parquet('" + TEST_PQ + "')").fetchnumpy()
X_te = np.column_stack([raw[c].astype(np.float32) for c in FEATURES])
y_te = {k: raw[k].astype(np.float32) for k in ["failed", "fail_type"]}

p1_te  = m1.predict(X_te)
p1_cal = iso.transform(p1_te)

# Threshold sweep — maximise slow-fail recall subject to precision >= 0.35
thresholds   = np.arange(0.02, 0.80, 0.005)
slow_recalls = [recall_score((y_te["fail_type"]==3).astype(int),
                (p1_cal>t).astype(np.int8), zero_division=0) for t in thresholds]
precs        = [precision_score(y_te["failed"], (p1_cal>t).astype(np.int8),
                zero_division=0) for t in thresholds]
f1s          = [f1_score(y_te["failed"], (p1_cal>t).astype(np.int8),
                zero_division=0) for t in thresholds]

# Primary: highest slow-fail recall where precision >= 0.35
valid  = [(sr, i) for i, (sr, p) in enumerate(zip(slow_recalls, precs)) if p >= 0.35]
best_t = float(thresholds[max(valid)[1]] if valid else thresholds[np.argmax(slow_recalls)])
pred1  = (p1_cal > best_t).astype(np.int8)

auc1   = roc_auc_score(y_te["failed"], p1_te)
brier1 = brier_score_loss(y_te["failed"], p1_cal)
prec1  = precision_score(y_te["failed"], pred1, zero_division=0)
rec1   = recall_score(y_te["failed"],    pred1, zero_division=0)
f1_1   = f1_score(y_te["failed"],        pred1, zero_division=0)
slow_r = recall_score((y_te["fail_type"]==3).astype(int), pred1, zero_division=0)

# Print full tradeoff curve so you can see the operating point
print("\n  Precision / Recall(slow) tradeoff:")
print(f"  {'Threshold':>10} {'Prec':>7} {'Rec(all)':>9} {'Rec(slow)':>10} {'F1':>7}")
print("  " + "-"*48)
for t, sr, p, f in zip(thresholds, slow_recalls, precs, f1s):
    if t % 0.05 < 0.005:   # print every 0.05
        marker = " ◄" if abs(t - best_t) < 0.003 else ""
        print(f"  {t:>10.3f} {p:>7.3f} "
              f"{recall_score(y_te['failed'],(p1_cal>t).astype(np.int8),zero_division=0):>9.3f} "
              f"{sr:>10.3f} {f:>7.3f}{marker}")

print(f"\n  AUC-ROC          : {auc1:.4f}")
print(f"  Brier score      : {brier1:.4f}")
print(f"  Threshold        : {best_t:.3f}  (max slow-recall @ precision>=0.35)")
print(f"  Precision(fail)  : {prec1:.4f}")
print(f"  Recall(fail)     : {rec1:.4f}")
print(f"  F1(fail)         : {f1_1:.4f}")
print(f"  Recall(slow-fail): {slow_r:.4f}")