"""
train_models_v6.py
All features (tabular + emb_0..emb_63) read directly from parquet.
No special embedding loading — embed_and_merge.py handled that already.
"""
import os, time, json, pickle
import numpy as np
import lightgbm as lgb
import duckdb
from sklearn.metrics import (roc_auc_score, f1_score, recall_score,
                             precision_score, classification_report,
                             brier_score_loss)
from sklearn.isotonic import IsotonicRegression

PREPARED  = os.getenv("PREPARED_PATH", "data/prepared/")
MODEL_DIR = os.getenv("MODELS_PATH", "models/")
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_PQ = PREPARED + "train_v3.parquet"
TEST_PQ  = PREPARED + "test_v3.parquet"

db = duckdb.connect()
db.execute("PRAGMA threads=30")
db.execute("PRAGMA memory_limit='16GB'")

# ── Discover features from parquet schema ────────────────────────
schema_cols = db.execute(
    "SELECT * FROM read_parquet('" + TRAIN_PQ + "') LIMIT 0"
).description
schema_cols = [col[0] for col in schema_cols]
LABEL_COLS  = {"failed","fail_type","log_duration","log_econ",
               "avgpcon_clean","wasted_node_hours","split","jid"}
FEATURES    = [c for c in schema_cols if c not in LABEL_COLS]
N_PCA       = sum(1 for c in FEATURES if c.startswith("emb_"))
print(f"Features: {len(FEATURES)} total  ({len(FEATURES)-N_PCA} tabular + {N_PCA} PCA embedding)")

# ── Loader ───────────────────────────────────────────────────────
def load(path, where_extra=""):
    feat_sql = ", ".join(FEATURES)
    label_sql = "failed, fail_type, log_duration, log_econ, avgpcon_clean, wasted_node_hours"
    where = "WHERE " + where_extra if where_extra else ""
    t0 = time.time()
    raw  = db.execute(
        f"SELECT {feat_sql}, {label_sql} FROM read_parquet('{path}') {where}"
    ).fetchnumpy()
    X = np.column_stack([raw[c].astype(np.float32) for c in FEATURES])
    y = {k: raw[k].astype(np.float32) for k in
         ["failed","fail_type","log_duration","log_econ","avgpcon_clean","wasted_node_hours"]}
    print(f"  {X.shape[0]:>10,} rows × {X.shape[1]} features  [{time.time()-t0:.1f}s]")
    return X, y

BASE = dict(
    device="cpu",      # ← was "gpu" — 30 cores beats GPU for 89-feature 20M-row data anyway
    n_jobs=30,         # ← saturate all cores
    num_leaves=255,
    max_depth=10,
    learning_rate=0.02,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    verbosity=-1,
)
CB = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
thresholds  = np.arange(0.02, 0.80, 0.005)
metrics_out = {}

# ════════════════════════════════════════════════════════════════
# STAGE 1 — Failure Classifier
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STAGE 1 — Failure Classifier")
print("="*60)
print("train:"); X_tr, y_tr = load(TRAIN_PQ)
print("test: "); X_te, y_te = load(TEST_PQ)

# cost-sensitive weights: slow-fail 3x, medium-fail 1.5x
sw = np.ones(len(y_tr["failed"]), dtype=np.float32)
sw[(y_tr["failed"]==1) & (y_tr["fail_type"]==3)] *= 3.0
sw[(y_tr["failed"]==1) & (y_tr["fail_type"]==2)] *= 1.5

m1 = lgb.train(
    {**BASE, "objective":"binary", "metric":"auc", "scale_pos_weight":9.1},
    lgb.Dataset(X_tr, label=y_tr["failed"], weight=sw,
                feature_name=FEATURES, free_raw_data=True),
    num_boost_round=1000,
    valid_sets=[lgb.Dataset(X_te, label=y_te["failed"], feature_name=FEATURES)],
    callbacks=CB,
)

p1_tr = m1.predict(X_tr)
p1_te = m1.predict(X_te)

# isotonic calibration
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(p1_tr, y_tr["failed"])
p1_cal = iso.transform(p1_te)

f1s   = [f1_score(y_te["failed"], (p1_cal>t).astype(np.int8), zero_division=0) for t in thresholds]
best_t = float(thresholds[np.argmax(f1s)])
pred1  = (p1_cal > best_t).astype(np.int8)

auc1   = roc_auc_score(y_te["failed"], p1_te)
brier1 = brier_score_loss(y_te["failed"], p1_cal)
prec1  = precision_score(y_te["failed"], pred1, zero_division=0)
rec1   = recall_score(y_te["failed"],    pred1, zero_division=0)
f1_1   = f1_score(y_te["failed"],        pred1, zero_division=0)
slow_r = recall_score((y_te["fail_type"]==3).astype(int), pred1, zero_division=0)

print(f"\n  AUC-ROC          : {auc1:.4f}")
print(f"  Brier score      : {brier1:.4f}  (lower=better)")
print(f"  Threshold        : {best_t:.3f}")
print(f"  Precision(fail)  : {prec1:.4f}")
print(f"  Recall(fail)     : {rec1:.4f}")
print(f"  F1(fail)         : {f1_1:.4f}")
print(f"  Recall(slow-fail): {slow_r:.4f}")
print(classification_report(y_te["failed"], pred1,
      target_names=["completed","failed"], digits=3))

# calibration reliability diagram
print("  Calibration reliability (is P(fail)=0.6 actually 60% fail?):")
bins = np.linspace(0, 1, 11)
for i in range(len(bins)-1):
    mask = (p1_cal >= bins[i]) & (p1_cal < bins[i+1])
    if mask.sum() < 100: continue
    actual = y_te["failed"][mask].mean()
    bar = "█" * int(20*actual)
    print(f"    pred[{bins[i]:.1f}-{bins[i+1]:.1f}] n={mask.sum():>7,}  "
          f"actual={100*actual:>5.1f}%  {bar}")

# feature importance
print("\n  Feature importance (top 20):")
imp    = m1.feature_importance(importance_type="gain")
total  = imp.sum()+1e-9
ranked = sorted(zip(FEATURES,imp), key=lambda x:-x[1])
for name, score in ranked[:20]:
    bar = "█" * int(30*score/(ranked[0][1]+1e-9))
    print(f"    {name:<28} {bar}  {100*score/total:.1f}%")

pickle.dump(m1,  open(MODEL_DIR+"s1_failure.pkl","wb"))
pickle.dump(iso, open(MODEL_DIR+"s1_calibrator.pkl","wb"))
json.dump({"threshold":best_t}, open(MODEL_DIR+"s1_threshold.json","w"))
metrics_out["s1"] = dict(auc=round(auc1,4), brier=round(brier1,4),
    threshold=round(best_t,3), precision=round(prec1,4),
    recall=round(rec1,4), f1=round(f1_1,4), slow_recall=round(slow_r,4))

# ════════════════════════════════════════════════════════════════
# STAGE 2 — Fail-Cost Classifier
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STAGE 2 — Fail-Cost Classifier  (stacked with P_fail)")
print("="*60)
print("train:"); X_tr_f, y_tr_f = load(TRAIN_PQ, "failed=1")
print("test: "); X_te_f, y_te_f = load(TEST_PQ,  "failed=1")

p_f_tr = iso.transform(m1.predict(X_tr_f)).reshape(-1,1).astype(np.float32)
p_f_te = iso.transform(m1.predict(X_te_f)).reshape(-1,1).astype(np.float32)
X2_tr  = np.hstack([X_tr_f, p_f_tr])
X2_te  = np.hstack([X_te_f, p_f_te])
F2     = FEATURES + ["p_fail"]

y2_tr = (y_tr_f["fail_type"] >= 2).astype(np.float32)
y2_te = (y_te_f["fail_type"] >= 2).astype(np.float32)
print(f"  Expensive fail rate — train: {y2_tr.mean():.3f}  test: {y2_te.mean():.3f}")

m2 = lgb.train(
    {**BASE, "objective":"binary", "metric":"auc"},
    lgb.Dataset(X2_tr, label=y2_tr, feature_name=F2, free_raw_data=True),
    num_boost_round=1000,
    valid_sets=[lgb.Dataset(X2_te, label=y2_te, feature_name=F2)],
    callbacks=CB,
)
iso2 = IsotonicRegression(out_of_bounds="clip")
iso2.fit(m2.predict(X2_tr), y2_tr)
p2_cal  = iso2.transform(m2.predict(X2_te))
auc2    = roc_auc_score(y2_te, m2.predict(X2_te))
brier2  = brier_score_loss(y2_te, p2_cal)
f1s2    = [f1_score(y2_te, (p2_cal>t).astype(np.int8), zero_division=0) for t in thresholds]
best_t2 = float(thresholds[np.argmax(f1s2)])
pred2   = (p2_cal > best_t2).astype(np.int8)
f1_2    = f1_score(y2_te, pred2, zero_division=0)
print(f"\n  AUC-ROC       : {auc2:.4f}")
print(f"  Brier score   : {brier2:.4f}")
print(f"  Threshold     : {best_t2:.3f}   F1(expensive): {f1_2:.4f}")
print(classification_report(y2_te, pred2,
      target_names=["cheap-fail","expensive-fail"], digits=3))

print("  Feature importance (top 10):")
imp2   = m2.feature_importance(importance_type="gain")
total2 = imp2.sum()+1e-9
ranked2 = sorted(zip(F2,imp2), key=lambda x:-x[1])
for name, score in ranked2[:10]:
    bar = "█" * int(25*score/(ranked2[0][1]+1e-9))
    print(f"    {name:<28} {bar}  {100*score/total2:.1f}%")

pickle.dump(m2,   open(MODEL_DIR+"s2_failcost.pkl","wb"))
pickle.dump(iso2, open(MODEL_DIR+"s2_calibrator.pkl","wb"))
json.dump({"threshold":best_t2}, open(MODEL_DIR+"s2_threshold.json","w"))
metrics_out["s2"] = dict(auc=round(auc2,4), brier=round(brier2,4),
    threshold=round(best_t2,3), f1_exp=round(f1_2,4))
del X_tr_f, X_te_f, X2_tr, X2_te, y_tr_f, y_te_f

# ════════════════════════════════════════════════════════════════
# STAGE 3a — Runtime Regressor  (completed)
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STAGE 3a — Runtime Regressor  (completed jobs)")
print("="*60)
print("train:"); Xc_tr, yc_tr = load(TRAIN_PQ, "failed=0")
print("test: "); Xc_te, yc_te = load(TEST_PQ,  "failed=0")

mc = lgb.train(
    {**BASE, "objective":"regression_l1", "metric":"mae"},
    lgb.Dataset(Xc_tr, label=yc_tr["log_duration"], feature_name=FEATURES, free_raw_data=True),
    num_boost_round=1000,
    valid_sets=[lgb.Dataset(Xc_te, label=yc_te["log_duration"], feature_name=FEATURES)],
    callbacks=CB,
)
pc     = mc.predict(Xc_te)
mae_c  = float(np.mean(np.abs(pc - yc_te["log_duration"])))
mape_c = float(np.mean(np.abs(pc-yc_te["log_duration"])/(np.abs(yc_te["log_duration"])+1e-6))*100)
print(f"  MAE(log-sec): {mae_c:.4f}   MAPE: {mape_c:.2f}%")
pickle.dump(mc, open(MODEL_DIR+"s3_runtime_completed.pkl","wb"))
metrics_out["s3"] = dict(mae_log=round(mae_c,4), mape_pct=round(mape_c,2))
del Xc_tr, Xc_te, yc_tr, yc_te

# ════════════════════════════════════════════════════════════════
# STAGE 3b — Failed Runtime Lookup
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STAGE 3b — Failed Runtime Lookup")
print("="*60)
rows = db.execute("""
    SELECT fail_type, COUNT(*) AS n,
           ROUND(MEDIAN(EXP(log_duration)-1),1) AS med,
           ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY EXP(log_duration)-1),1) AS p25,
           ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY EXP(log_duration)-1),1) AS p75
    FROM read_parquet('""" + TRAIN_PQ + """')
    WHERE failed=1 AND fail_type>0 GROUP BY fail_type ORDER BY fail_type
""").fetchall()
lookup = {}
names  = {1:"quick (<5min)",2:"medium (5min-2hr)",3:"slow (2hr+)"}
for r in rows:
    ft = int(r[0])
    lookup[str(ft)] = dict(median_s=r[2], p25_s=r[3], p75_s=r[4], n=int(r[1]))
    print(f"  fail_type={ft} {names[ft]:<18} median={r[2]:>8.0f}s  p25={r[3]:>7.0f}s  p75={r[4]:>7.0f}s  n={r[1]:,}")
json.dump(lookup, open(MODEL_DIR+"failed_runtime_lookup.json","w"), indent=2)

# ════════════════════════════════════════════════════════════════
# STAGE 4 — Energy Regressor
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STAGE 4 — Energy Regressor")
print("="*60)
me = lgb.train(
    {**BASE, "objective":"regression_l1", "metric":"mae"},
    lgb.Dataset(X_tr, label=y_tr["log_econ"], feature_name=FEATURES, free_raw_data=True),
    num_boost_round=1000,
    valid_sets=[lgb.Dataset(X_te, label=y_te["log_econ"], feature_name=FEATURES)],
    callbacks=CB,
)
pe     = me.predict(X_te)
mae_e  = float(np.mean(np.abs(pe - y_te["log_econ"])))
mask_e = y_te["log_econ"] > 4.6
mape_e = float(np.mean(np.abs(pe[mask_e]-y_te["log_econ"][mask_e])/(np.abs(y_te["log_econ"][mask_e])+1e-6))*100)
print(f"  MAE(log-J): {mae_e:.4f}   Filtered MAPE (econ>100J): {mape_e:.2f}%  [{mask_e.sum():,} jobs]")
pickle.dump(me, open(MODEL_DIR+"s4_energy.pkl","wb"))
metrics_out["s4"] = dict(mae_log=round(mae_e,4), filtered_mape=round(mape_e,2))

# ── Power lookup ─────────────────────────────────────────────────
pw_rows = db.execute("""
    SELECT is_compute_bound,
        CASE WHEN log_nnumr<0.693 THEN 0 WHEN log_nnumr<3.497 THEN 1
             WHEN log_nnumr<6.238 THEN 2 ELSE 3 END AS nb,
        ROUND(MEDIAN(avgpcon_clean),2) AS med_w,
        ROUND(MEDIAN(avgpcon_clean/NULLIF(EXP(log_nnumr)-1,0)),2) AS wpn,
        COUNT(*) AS n
    FROM read_parquet('""" + TRAIN_PQ + """')
    WHERE avgpcon_clean>0 GROUP BY 1,2 ORDER BY 1,2
""").fetchall()
pw_lookup = {}
for r in pw_rows:
    pw_lookup[str(int(r[0]))+"_"+str(int(r[1]))] = dict(median_w=r[2], w_per_node=r[3], n=int(r[4]))
json.dump(pw_lookup, open(MODEL_DIR+"power_lookup.json","w"), indent=2)

json.dump(metrics_out, open(MODEL_DIR+"metrics_v6.json","w"), indent=2)
json.dump({"features": FEATURES, "n_pca": N_PCA},
          open(MODEL_DIR+"feature_registry.json","w"), indent=2)

print(f"\nAll models → {MODEL_DIR}")
print("Done.")
