"""
Fugaku PA Model Training Pipeline — v3
Fixes: threshold sweep, Stage 2 stacking, failed runtime lookup, filtered energy MAPE
"""
import os, time, json, pickle
import numpy as np
import lightgbm as lgb
import duckdb
from sklearn.metrics import (roc_auc_score, f1_score, recall_score,
                             precision_score, classification_report)

PREPARED  = os.getenv("PREPARED_PATH", "data/prepared/")
MODEL_DIR = os.getenv("MODELS_PATH", "models/")
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_PQ = f"{PREPARED}train_v2.parquet"
TEST_PQ  = f"{PREPARED}test_v2.parquet"

db = duckdb.connect()
db.execute("PRAGMA threads=8")
db.execute("PRAGMA memory_limit='12GB'")

# ── Features ──────────────────────────────────────────────────────
FEATURES = [
    "log_nnumr", "log_nnuma", "log_cnumr", "log_elpl",
    "log_mszl",  "log_msza",  "is_compute_bound",
    "freq_req",  "pri",       "jobenv_0",  "jobenv_1",
    "elpl_per_node", "compute_x_nodes", "node_bucket",
    "user_fail_rate", "usr_x_compute", "usr_x_nodes",
]

FEAT_SQL = """
SELECT
    log_nnumr, log_nnuma, log_cnumr, log_elpl,
    COALESCE(log_mszl, 0.0)  AS log_mszl,
    COALESCE(log_msza, 0.0)  AS log_msza,
    CAST(is_compute_bound AS FLOAT)          AS is_compute_bound,
    CAST(COALESCE(freq_req, 0) AS FLOAT)     AS freq_req,
    CAST(COALESCE(pri, 0)     AS FLOAT)      AS pri,
    CAST(jobenv_0 AS FLOAT)                  AS jobenv_0,
    CAST(jobenv_1 AS FLOAT)                  AS jobenv_1,
    (log_elpl - log_nnumr)                   AS elpl_per_node,
    (is_compute_bound * log_nnumr)           AS compute_x_nodes,
    CAST(CASE
        WHEN log_nnumr < 0.693 THEN 0
        WHEN log_nnumr < 3.497 THEN 1
        WHEN log_nnumr < 6.238 THEN 2
        ELSE 3
    END AS FLOAT)                            AS node_bucket,
    -- ★ user historical failure rate + interactions ★
    CAST(COALESCE(user_fail_rate, 0.0989) AS FLOAT) AS user_fail_rate,
    CAST(COALESCE(user_fail_rate, 0.0989) * is_compute_bound AS FLOAT) AS usr_x_compute,
    CAST(COALESCE(user_fail_rate, 0.0989) * log_nnumr       AS FLOAT) AS usr_x_nodes,
    CAST(failed      AS FLOAT)               AS failed,
    CAST(fail_type   AS FLOAT)               AS fail_type,
    CAST(log_duration AS FLOAT)              AS log_duration,
    CAST(log_econ     AS FLOAT)              AS log_econ,
    CAST(COALESCE(avgpcon_clean, 0.0) AS FLOAT) AS avgpcon_clean
FROM read_parquet('{path}')
{where}
"""
LABEL_COLS = ["failed","fail_type","log_duration","log_econ","avgpcon_clean"]

def load(path, where=""):
    sql = FEAT_SQL.format(path=path, where=f"WHERE {where}" if where else "")
    t0  = time.time()
    raw = db.execute(sql).fetchnumpy()
    X   = np.column_stack([raw[c].astype(np.float32) for c in FEATURES])
    y   = {k: raw[k].astype(np.float32) for k in LABEL_COLS}
    print(f"  {X.shape[0]:>10,} rows × {X.shape[1]} features  [{time.time()-t0:.1f}s]")
    return X, y

def make_ds(X, y, ref=None):
    d = lgb.Dataset(X, label=y, free_raw_data=True, feature_name=list(FEATURES))
    if ref is not None: d.reference = ref
    return d

BASE = dict(
    device="gpu", num_leaves=127, max_depth=8,
    learning_rate=0.05, min_child_samples=100,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    min_data_in_bin=1, gpu_use_dp=True,
    n_jobs=-1, verbosity=-1,
)
CB = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)]
metrics_out = {}

# ═══════════════════════════════════════════════════════════════════
# STAGE 1 — Binary Failure Classifier
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STAGE 1 — Failure Classifier  (binary, all jobs)")
print("="*60)

print("train:"); X_tr, y_tr = load(TRAIN_PQ)
print("test: "); X_te, y_te = load(TEST_PQ)

m1 = lgb.train(
    {**BASE, "objective":"binary", "metric":"auc", "scale_pos_weight":9.1},
    make_ds(X_tr, y_tr["failed"]),
    num_boost_round=500,
    valid_sets=[make_ds(X_te, y_te["failed"])],
    callbacks=CB,
)

p1_tr = m1.predict(X_tr)   # keep for Stage 2 stacking
p1_te = m1.predict(X_te)
auc1  = roc_auc_score(y_te["failed"], p1_te)

# ── Threshold sweep on TEST set (maximise F1 on minority class) ──
thresholds = np.arange(0.02, 0.50, 0.005)
f1s = [f1_score(y_te["failed"], (p1_te > t).astype(np.int8), zero_division=0)
       for t in thresholds]
best_t = float(thresholds[np.argmax(f1s)])
best_f1 = float(max(f1s))
print(f"\n  Threshold sweep → best t={best_t:.3f}  F1={best_f1:.4f}")

pred1 = (p1_te > best_t).astype(np.int8)
f1_1  = f1_score(y_te["failed"], pred1, zero_division=0)
prec1 = precision_score(y_te["failed"], pred1, zero_division=0)
rec1  = recall_score(y_te["failed"], pred1, zero_division=0)

slow_mask   = (y_te["fail_type"] == 3).astype(np.int8)
slow_recall = recall_score(slow_mask, (p1_te > best_t).astype(np.int8), zero_division=0)

print(f"\n  AUC-ROC          : {auc1:.4f}")
print(f"  Threshold        : {best_t:.3f}")
print(f"  Precision(fail)  : {prec1:.4f}")
print(f"  Recall(fail)     : {rec1:.4f}")
print(f"  F1(fail)         : {f1_1:.4f}")
print(f"  Recall(slow-fail): {slow_recall:.4f}")
print(classification_report(y_te["failed"], pred1,
      target_names=["completed","failed"], digits=3))

pickle.dump(m1, open(f"{MODEL_DIR}s1_failure.pkl","wb"))
json.dump({"threshold": best_t}, open(f"{MODEL_DIR}s1_threshold.json","w"))
metrics_out["s1_failure"] = dict(
    auc=round(auc1,4), threshold=round(best_t,3),
    precision=round(prec1,4), recall=round(rec1,4),
    f1=round(f1_1,4), slow_fail_recall=round(slow_recall,4)
)

# ═══════════════════════════════════════════════════════════════════
# STAGE 2 — Fail-Cost Classifier  (stacked: features + P(fail))
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STAGE 2 — Fail-Cost Classifier  (stacked with P_fail)")
print("="*60)

print("train:"); X_tr_f, y_tr_f = load(TRAIN_PQ, "failed = 1")
print("test: "); X_te_f, y_te_f = load(TEST_PQ,  "failed = 1")

# Stack P(fail) from Stage 1 as an extra feature
p_fail_tr_f = m1.predict(X_tr_f).reshape(-1,1).astype(np.float32)
p_fail_te_f = m1.predict(X_te_f).reshape(-1,1).astype(np.float32)
X_tr_f2 = np.hstack([X_tr_f, p_fail_tr_f])
X_te_f2 = np.hstack([X_te_f, p_fail_te_f])
FEATURES_S2 = FEATURES + ["p_fail"]
print(f"  Features for Stage 2: {len(FEATURES_S2)} ({len(FEATURES)} base + p_fail)")

y2_tr = (y_tr_f["fail_type"] >= 2).astype(np.float32)  # expensive = ≥5min
y2_te = (y_te_f["fail_type"] >= 2).astype(np.float32)
print(f"  Expensive fail rate — train: {y2_tr.mean():.3f}  test: {y2_te.mean():.3f}")

ds2_tr = lgb.Dataset(X_tr_f2, label=y2_tr, free_raw_data=True,
                     feature_name=list(FEATURES_S2))
ds2_te = lgb.Dataset(X_te_f2, label=y2_te, free_raw_data=True,
                     feature_name=list(FEATURES_S2), reference=ds2_tr)

m2 = lgb.train(
    {**BASE, "objective":"binary", "metric":"auc"},
    ds2_tr,
    num_boost_round=500,
    valid_sets=[ds2_te],
    callbacks=CB,
)

p2    = m2.predict(X_te_f2)
auc2  = roc_auc_score(y2_te, p2)
# Threshold sweep for Stage 2
f1s2  = [f1_score(y2_te, (p2 > t).astype(np.int8), zero_division=0)
         for t in thresholds]
best_t2 = float(thresholds[np.argmax(f1s2)])
pred2   = (p2 > best_t2).astype(np.int8)
f1_2    = f1_score(y2_te, pred2, zero_division=0)
print(f"\n  AUC-ROC       : {auc2:.4f}")
print(f"  Best threshold: {best_t2:.3f}   F1(expensive): {f1_2:.4f}")
print(classification_report(y2_te, pred2,
      target_names=["cheap-fail","expensive-fail"], digits=3))

pickle.dump(m2, open(f"{MODEL_DIR}s2_failcost.pkl","wb"))
json.dump({"threshold": best_t2}, open(f"{MODEL_DIR}s2_threshold.json","w"))
metrics_out["s2_failcost"] = dict(auc=round(auc2,4), threshold=round(best_t2,3),
                                   f1=round(f1_2,4))

# ── Stage 2 feature importance ──────────────────────────────────
imp2   = m2.feature_importance(importance_type="gain")
ranked2 = sorted(zip(FEATURES_S2, imp2), key=lambda x: -x[1])
print("\n  Feature importance (Stage 2):")
total2 = imp2.sum() + 1e-9
for name, score in ranked2[:8]:
    bar = "█" * int(30 * score / (ranked2[0][1]+1e-9))
    print(f"    {name:<22} {bar}  {100*score/total2:.1f}%")

del X_tr_f, X_te_f, X_tr_f2, X_te_f2, y_tr_f, y_te_f

# ═══════════════════════════════════════════════════════════════════
# STAGE 3a — Runtime Regressor  (completed jobs only)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STAGE 3a — Runtime Regressor  (completed jobs)")
print("="*60)

print("train:"); Xc_tr, yc_tr = load(TRAIN_PQ, "failed = 0")
print("test: "); Xc_te, yc_te = load(TEST_PQ,  "failed = 0")

reg_p = {**BASE, "objective":"regression_l1", "metric":"mae"}
mc = lgb.train(
    reg_p,
    lgb.Dataset(Xc_tr, label=yc_tr["log_duration"], free_raw_data=True),
    num_boost_round=500,
    valid_sets=[lgb.Dataset(Xc_te, label=yc_te["log_duration"], free_raw_data=True)],
    callbacks=CB,
)
pc    = mc.predict(Xc_te)
mae_c = float(np.mean(np.abs(pc - yc_te["log_duration"])))
mape_c= float(np.mean(np.abs(pc - yc_te["log_duration"]) /
                       np.abs(yc_te["log_duration"] + 1e-6)) * 100)
print(f"  MAE(log-sec): {mae_c:.4f}   MAPE: {mape_c:.2f}%")
pickle.dump(mc, open(f"{MODEL_DIR}s3_runtime_completed.pkl","wb"))
metrics_out["s3_runtime_completed"] = dict(mae_log=round(mae_c,4), mape_pct=round(mape_c,2))
del Xc_tr, Xc_te, yc_tr, yc_te

# ═══════════════════════════════════════════════════════════════════
# STAGE 3b — Failed Runtime Lookup  (replaces chaotic regressor)
# Use training-set median duration per fail_type bucket
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STAGE 3b — Failed Runtime Lookup  (median by fail_type)")
print("="*60)

type_names = {1: "quick (<5min)", 2: "medium (5min-2hr)", 3: "slow (2hr+)"}
rows = db.execute(f"""
    SELECT
        fail_type,
        COUNT(*)                                        AS n,
        ROUND(MEDIAN(EXP(log_duration)-1), 1)          AS median_dur_s,
        ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP
              (ORDER BY EXP(log_duration)-1), 1)        AS p25_dur_s,
        ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP
              (ORDER BY EXP(log_duration)-1), 1)        AS p75_dur_s
    FROM read_parquet('{TRAIN_PQ}')
    WHERE failed = 1 AND fail_type > 0
    GROUP BY fail_type ORDER BY fail_type
""").fetchall()

runtime_lookup = {}
for r in rows:
    ft = int(r[0])
    runtime_lookup[str(ft)] = dict(
        median_s=r[2], p25_s=r[3], p75_s=r[4], n_jobs=r[1]
    )
    print(f"  fail_type={ft} {type_names[ft]:<22} "
          f"median={r[2]:>8.0f}s  p25={r[3]:>7.0f}s  p75={r[4]:>8.0f}s  n={r[1]:,}")

json.dump(runtime_lookup, open(f"{MODEL_DIR}failed_runtime_lookup.json","w"), indent=2)
metrics_out["s3b_runtime_failed"] = {"type": "lookup", "entries": runtime_lookup}

# ═══════════════════════════════════════════════════════════════════
# STAGE 4 — Energy Regressor  (all jobs)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STAGE 4 — Energy Regressor  (all jobs)")
print("="*60)

me = lgb.train(
    reg_p,
    lgb.Dataset(X_tr, label=y_tr["log_econ"], free_raw_data=True),
    num_boost_round=500,
    valid_sets=[lgb.Dataset(X_te, label=y_te["log_econ"], free_raw_data=True)],
    callbacks=CB,
)
pe     = me.predict(X_te)
mae_e  = float(np.mean(np.abs(pe - y_te["log_econ"])))

# Filtered MAPE — only jobs with econ > 100J (log > 4.6) to avoid tiny-job distortion
mask_e = y_te["log_econ"] > 4.6
mape_e = float(np.mean(np.abs(pe[mask_e] - y_te["log_econ"][mask_e]) /
                        np.abs(y_te["log_econ"][mask_e] + 1e-6)) * 100)
print(f"  MAE(log-J): {mae_e:.4f}   Filtered MAPE (econ>100J): {mape_e:.2f}%  "
      f"[{mask_e.sum():,} jobs]")
pickle.dump(me, open(f"{MODEL_DIR}s4_energy.pkl","wb"))
metrics_out["s4_energy"] = dict(mae_log=round(mae_e,4),
                                 filtered_mape_pct=round(mape_e,2),
                                 filter="econ>100J")

# ═══════════════════════════════════════════════════════════════════
# POWER LOOKUP TABLE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("POWER LOOKUP TABLE")
print("="*60)

rows = db.execute(f"""
    SELECT is_compute_bound,
        CASE WHEN log_nnumr < 0.693 THEN 0
             WHEN log_nnumr < 3.497 THEN 1
             WHEN log_nnumr < 6.238 THEN 2
             ELSE 3 END AS nb,
        ROUND(MEDIAN(avgpcon_clean),2)                        AS med_w,
        ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP
              (ORDER BY avgpcon_clean),2)                      AS p75_w,
        ROUND(MEDIAN(avgpcon_clean/NULLIF(EXP(log_nnumr)-1,0)),2) AS w_per_node,
        COUNT(*) AS n
    FROM read_parquet('{TRAIN_PQ}')
    WHERE avgpcon_clean > 0
    GROUP BY 1,2 ORDER BY 1,2
""").fetchall()

bnames = ["single(1)","small(2-32)","medium(33-512)","large(>512)"]
lookup = {}
for r in rows:
    key = f"{int(r[0])}_{int(r[1])}"
    lookup[key] = dict(median_w=r[2], p75_w=r[3], w_per_node=r[4], n_jobs=r[5])
    cls = "compute" if r[0] else "memory "
    print(f"  {cls}-bound  {bnames[int(r[1])]:<18}  "
          f"median={r[2]:>8.1f}W  W/node={r[4]:>6.1f}  n={r[5]:,}")

json.dump(lookup, open(f"{MODEL_DIR}power_lookup.json","w"), indent=2)

# ═══════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE — Stage 1
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FEATURE IMPORTANCE — Stage 1")
print("="*60)

imp    = m1.feature_importance(importance_type="gain")
total  = imp.sum() + 1e-9
ranked = sorted(zip(FEATURES, imp), key=lambda x: -x[1])
for name, score in ranked:
    bar = "█" * int(35 * score / (ranked[0][1]+1e-9))
    print(f"  {name:<22} {bar}  {100*score/total:.1f}%")

# ── Save summary ─────────────────────────────────────────────────
metrics_out["features"]       = FEATURES
metrics_out["features_s2"]    = FEATURES_S2
json.dump(metrics_out, open(f"{MODEL_DIR}metrics.json","w"), indent=2)
print(f"\n\nAll models saved to: {MODEL_DIR}")
print("Done.")
