"""
train_failure_classifier_v5.py

First run  : builds train/test parquets from scratch (slow, one-time)
Later runs : loads pre-built parquets directly (fast, under 2 min)

Quick test:
    SAMPLE_FRAC=0.01 python analytics/train_failure_classifier_v5.py
Full run:
    python analytics/train_failure_classifier_v5.py
"""
import duckdb, numpy as np, xgboost as xgb, json, os, time
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              confusion_matrix, classification_report,
                              precision_recall_curve)

# ── Config ────────────────────────────────────────────────────
RAW_GLOB     = os.path.join(os.getenv("FUGAKU_DATA_PATH", "data/fugaku"), "*.parquet")
PREPARED_DIR = os.getenv("PREPARED_PATH", "data/prepared/")
MODEL_DIR    = os.getenv("MODELS_PATH", "models/")
TMP_DIR      = os.getenv("DUCKDB_TMP", "/tmp/duckdb_tmp")
CUTOFF       = "2023-08-28 00:37:43+00:00"
TRAIN_FROM   = "2022-02-28 00:00:00+00:00"
MIN_PREC     = 0.30

SAMPLE_FRAC  = float(os.environ.get("SAMPLE_FRAC", "1.0"))
SAMPLE_TAG   = f"_sample{int(SAMPLE_FRAC*100)}" if SAMPLE_FRAC < 1.0 else ""
TRAIN_PARQ   = PREPARED_DIR + f"train_features{SAMPLE_TAG}.parquet"
TEST_PARQ    = PREPARED_DIR + f"test_features{SAMPLE_TAG}.parquet"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TMP_DIR,   exist_ok=True)

print(f"SAMPLE_FRAC : {SAMPLE_FRAC}  ({'FULL RUN' if SAMPLE_FRAC==1.0 else f'{int(SAMPLE_FRAC*100)}% SAMPLE'})")
print(f"Train cache : {TRAIN_PARQ}")
print(f"Test  cache : {TEST_PARQ}")

FEATURES = [
    "log_nnumr","log_nnuma","log_cnumr","log_elpl","log_mszl","log_msza",
    "is_compute_bound","freq_req","hour_of_day","day_of_week","log_queue_wait",
    "jnam_fail_rate_30d","usr_fail_rate_30d",
    "jnam_fail_rate_alltime","bucket_fail_rate","log_jnam_count"
]
N_PCA        = 32
PCA_COLS     = [f"emb_{i}" for i in range(N_PCA)]
ALL_FEATURES = FEATURES + PCA_COLS

def T(label):
    print(f"  {label}...", end=" ", flush=True)
    return time.time()
def done(t0): print(f"done ({time.time()-t0:.1f}s)")

# ══════════════════════════════════════════════════════════════
# FAST PATH — pre-built parquets exist
# ══════════════════════════════════════════════════════════════
if os.path.exists(TRAIN_PARQ) and os.path.exists(TEST_PARQ):
    print("\n[FAST PATH] Loading pre-built feature parquets...")
    import pandas as pd
    t0 = T("train")
    train_df = pd.read_parquet(TRAIN_PARQ)
    done(t0)
    t0 = T("test")
    test_df  = pd.read_parquet(TEST_PARQ)
    done(t0)
    print(f"  Train: {train_df.shape}  failure={train_df.failed.mean()*100:.1f}%")
    print(f"  Test : {test_df.shape}   failure={test_df.failed.mean()*100:.1f}%")

# ══════════════════════════════════════════════════════════════
# SLOW PATH — build from scratch and save
# ══════════════════════════════════════════════════════════════
else:
    print("\n[SLOW PATH] Building features from raw parquets (one-time)...")
    SAMPLE_CLAUSE = f"USING SAMPLE {SAMPLE_FRAC*100:.4g} PERCENT (bernoulli)" if SAMPLE_FRAC < 1.0 else ""

    db = duckdb.connect(config={
        "temp_directory":          TMP_DIR,
        "max_temp_directory_size": "500GiB",
        "memory_limit":            "160GB",
        "threads":                 8,
        "preserve_insertion_order": False,
    })
    db.execute(f"CREATE VIEW raw AS SELECT * FROM read_parquet(\'{RAW_GLOB}\')")

    # Step 1: base table (sampled if SAMPLE_FRAC < 1)
    print("Step 1: base table...")
    t0 = T("scan")
    db.execute(f"""
    CREATE OR REPLACE TABLE base AS
    SELECT jid, jnam, usr,
           CAST(sdt AS TIMESTAMPTZ) AS sdt_ts,
           CAST(qdt AS TIMESTAMPTZ) AS qdt_ts,
           CASE WHEN "exit state"=\'failed\' THEN 1 ELSE 0 END AS failed,
           nnumr, nnuma, cnumr, elpl, mszl, msza,
           pclass, freq_req, duration, econ, avgpcon,
           CASE WHEN nnumr=1 THEN \'single\'
                WHEN nnumr BETWEEN 2  AND 32  THEN \'small\'
                WHEN nnumr BETWEEN 33 AND 512 THEN \'medium\'
                ELSE \'large\' END AS node_bucket
    FROM raw
    WHERE sdt IS NOT NULL AND duration > 0 AND elpl > 0
    {SAMPLE_CLAUSE}
    """)
    n = db.execute("SELECT COUNT(*) FROM base").fetchone()[0]
    done(t0)
    print(f"  Rows: {n:,}")

    # Step 2: rolling 30-day rates → parquet → drop
    print("Step 2: rolling 30-day failure rates...")
    t0 = T("jnam rolling")
    db.execute("""
    CREATE OR REPLACE TABLE jnam_rolling AS
    SELECT jid,
        AVG(failed) OVER (PARTITION BY jnam ORDER BY sdt_ts
            RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND INTERVAL 1 SECOND PRECEDING
        ) AS jnam_fail_rate_30d
    FROM base
    """)
    db.execute(f"COPY jnam_rolling TO \'{PREPARED_DIR}jnam_rolling{SAMPLE_TAG}.parquet\' (FORMAT PARQUET, COMPRESSION ZSTD)")
    db.execute("DROP TABLE jnam_rolling")
    done(t0)

    t0 = T("usr rolling")
    db.execute("""
    CREATE OR REPLACE TABLE usr_rolling AS
    SELECT jid,
        AVG(failed) OVER (PARTITION BY usr ORDER BY sdt_ts
            RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND INTERVAL 1 SECOND PRECEDING
        ) AS usr_fail_rate_30d
    FROM base
    """)
    db.execute(f"COPY usr_rolling TO \'{PREPARED_DIR}usr_rolling{SAMPLE_TAG}.parquet\' (FORMAT PARQUET, COMPRESSION ZSTD)")
    db.execute("DROP TABLE usr_rolling")
    done(t0)

    # Step 3: lookup stats
    print("Step 3: lookup stats...")
    GLOBAL_FAIL = db.execute("SELECT AVG(failed) FROM base").fetchone()[0]
    print(f"  global_fail={GLOBAL_FAIL:.4f}")

    db.execute(f"""
    CREATE OR REPLACE TABLE jat AS
    SELECT jnam, AVG(failed) AS jnam_fail_rate_alltime, COUNT(*) AS jnam_count
    FROM base WHERE sdt_ts <= CAST(\'{CUTOFF}\' AS TIMESTAMPTZ) GROUP BY jnam
    """)
    db.execute(f"""
    CREATE OR REPLACE TABLE bat AS
    SELECT node_bucket, AVG(failed) AS bucket_fail_rate
    FROM base WHERE sdt_ts <= CAST(\'{CUTOFF}\' AS TIMESTAMPTZ) GROUP BY node_bucket
    """)

    # Step 4+5: stream directly to pandas — no intermediate materialization
    print("Step 4: streaming join → pandas (no intermediate table)...")

    JNAM_ROLL = PREPARED_DIR + f"jnam_rolling{SAMPLE_TAG}.parquet"
    USR_ROLL  = PREPARED_DIR + f"usr_rolling{SAMPLE_TAG}.parquet"
    EMB_PCA   = PREPARED_DIR + "embeddings_pca32.parquet"

    FEAT_EXPR = f"""
        LN(1+b.nnumr)  AS log_nnumr,
        LN(1+b.nnuma)  AS log_nnuma,
        LN(1+b.cnumr)  AS log_cnumr,
        LN(1+b.elpl)   AS log_elpl,
        LN(1+CASE WHEN b.mszl<0 OR b.mszl>1e18 THEN NULL ELSE b.mszl END) AS log_mszl,
        LN(1+CASE WHEN CAST(b.msza AS DOUBLE)>1e18 THEN NULL ELSE CAST(b.msza AS DOUBLE) END) AS log_msza,
        CASE WHEN b.pclass=\'compute-bound\' THEN 1 ELSE 0 END AS is_compute_bound,
        COALESCE(b.freq_req,0)  AS freq_req,
        HOUR(b.sdt_ts)          AS hour_of_day,
        DAYOFWEEK(b.sdt_ts)     AS day_of_week,
        LN(1+GREATEST(DATEDIFF(\'second\',b.qdt_ts,b.sdt_ts),0)) AS log_queue_wait,
        COALESCE(jr.jnam_fail_rate_30d,    {GLOBAL_FAIL}) AS jnam_fail_rate_30d,
        COALESCE(ur.usr_fail_rate_30d,     {GLOBAL_FAIL}) AS usr_fail_rate_30d,
        COALESCE(j.jnam_fail_rate_alltime, {GLOBAL_FAIL}) AS jnam_fail_rate_alltime,
        COALESCE(bt.bucket_fail_rate,      {GLOBAL_FAIL}) AS bucket_fail_rate,
        LN(1+COALESCE(j.jnam_count,1))                   AS log_jnam_count,
        {", ".join([f"e.emb_{i}" for i in range(N_PCA)])},
        b.failed
    """

    BASE_JOIN = f"""
        FROM base b
        LEFT JOIN read_parquet(\'{JNAM_ROLL}\') jr USING (jid)
        LEFT JOIN read_parquet(\'{USR_ROLL}\')  ur USING (jid)
        LEFT JOIN jat  j  USING (jnam)
        LEFT JOIN bat  bt ON b.node_bucket = bt.node_bucket
        INNER JOIN read_parquet(\'{EMB_PCA}\')  e  USING (jid)
    """

    t0 = T("train split")
    train_df = db.execute(f"""
        SELECT {FEAT_EXPR} {BASE_JOIN}
        WHERE b.sdt_ts > CAST(\'{TRAIN_FROM}\' AS TIMESTAMPTZ)
          AND b.sdt_ts <= CAST(\'{CUTOFF}\' AS TIMESTAMPTZ)
    """).df()
    done(t0)

    t0 = T("test split")
    test_df = db.execute(f"""
        SELECT {FEAT_EXPR} {BASE_JOIN}
        WHERE b.sdt_ts > CAST(\'{CUTOFF}\' AS TIMESTAMPTZ)
    """).df()
    done(t0)

    # Save for future runs
    t0 = T("saving train parquet")
    train_df.to_parquet(TRAIN_PARQ, index=False, compression="zstd")
    done(t0)
    t0 = T("saving test parquet")
    test_df.to_parquet(TEST_PARQ, index=False, compression="zstd")
    done(t0)
    print(f"  Saved: {TRAIN_PARQ}")
    print(f"  Saved: {TEST_PARQ}")
    print("  Future runs will skip this entire block.")

    db.execute("DROP TABLE jat"); db.execute("DROP TABLE bat"); db.close()

# ══════════════════════════════════════════════════════════════
# TRAINING (same for both paths)
# ══════════════════════════════════════════════════════════════
X_train = train_df[ALL_FEATURES].values.astype(np.float32)
X_test  = test_df[ALL_FEATURES].values.astype(np.float32)
y_train = train_df["failed"].values
y_test  = test_df["failed"].values
print(f"\nTrain: {X_train.shape}  failure={y_train.mean()*100:.1f}%")
print(f"Test : {X_test.shape}   failure={y_test.mean()*100:.1f}%")

scale = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale:.2f}")

print("\nTraining XGBoost v5...")
model = xgb.XGBClassifier(
    n_estimators=1000, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.6, min_child_weight=30,
    scale_pos_weight=25, eval_metric="aucpr",
    early_stopping_rounds=30, n_jobs=-1, random_state=42, device="cuda",
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

# Evaluate
y_prob  = model.predict_proba(X_test)[:, 1]
auc_roc = roc_auc_score(y_test, y_prob)
auc_pr  = average_precision_score(y_test, y_prob)

prec_arr, rec_arr, thresholds = precision_recall_curve(y_test, y_prob)

# Threshold @ recall >= 0.75
t_75 = float(thresholds[np.argmin(np.abs(rec_arr[:-1] - 0.75))])

# Threshold @ max F1 with precision >= MIN_PREC
best_f1, best_t = 0, 0.5
for p, r, t in zip(prec_arr, rec_arr, thresholds):
    if p >= MIN_PREC:
        f1 = 2*p*r/(p+r+1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

print(f"\n── EVALUATION ──")
print(f"  AUC-ROC : {auc_roc:.4f}")
print(f"  AUC-PR  : {auc_pr:.4f}  (baseline={y_test.mean():.3f})")

for label, t in [("recall=0.75", t_75), (f"max-F1 @ prec>={MIN_PREC}", best_t)]:
    y_pred = (y_prob >= t).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  ── threshold={t:.3f}  ({label}) ──")
    print(f"  TN={cm[0,0]:>8,}  FP={cm[0,1]:>8,}")
    print(f"  FN={cm[1,0]:>8,}  TP={cm[1,1]:>8,}")
    print(classification_report(y_test, y_pred, target_names=["completed","failed"]))

print("── TOP-K PRECISION ──")
for pct in [0.05, 0.10, 0.20]:
    k = int(len(y_prob)*pct)
    idx = np.argsort(y_prob)[::-1][:k]
    print(f"  Top {int(pct*100):2}% ({k:>7,}) → {y_test[idx].mean()*100:.1f}% real failures")

print("\n── TOP-10 FEATURE IMPORTANCES ──")
for name, sc in sorted(zip(ALL_FEATURES, model.feature_importances_), key=lambda x:-x[1])[:10]:
    print(f"  {name:<25}  {sc:.4f}  {'█'*int(sc*150)}")

tag = SAMPLE_TAG or "_full"
model.save_model(MODEL_DIR + f"failure_classifier_v5{tag}.json")
with open(MODEL_DIR + f"failure_classifier_v5{tag}_eval.json", "w") as f:
    json.dump({"auc_roc":round(auc_roc,4), "auc_pr":round(auc_pr,4),
               "t_75":round(t_75,4), "best_t":round(best_t,4),
               "best_f1":round(best_f1,4), "sample_frac":SAMPLE_FRAC,
               "features": ALL_FEATURES}, f, indent=2)
print(f"\nSaved: failure_classifier_v5{tag}.json")
print("Done.")
