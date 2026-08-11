"""
Step 2b — Enriched failure classifier with historical features.
New features: jnam/usr/node-bucket failure rates, time-of-day, queue wait time.
All historical rates computed from TRAIN only, looked up for TEST (no leakage).
"""
import duckdb
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
import json, os

RAW_GLOB     = os.path.join(os.getenv("FUGAKU_DATA_PATH", "data/fugaku"), "*.parquet")
PREPARED_DIR = os.getenv("PREPARED_PATH", "data/prepared/")
MODEL_DIR    = os.getenv("MODELS_PATH", "models/")
os.makedirs(MODEL_DIR, exist_ok=True)

CUTOFF = "2023-08-28 00:37:43+00:00"

db = duckdb.connect()
db.execute(f"CREATE VIEW raw AS SELECT * FROM read_parquet(\'{RAW_GLOB}\')")

# ── 1. Build enriched feature table from raw parquets ─────────
print("Building enriched feature table...")
db.execute(f"""
CREATE OR REPLACE TABLE enriched AS
SELECT
    -- Split
    CASE WHEN CAST(sdt AS TIMESTAMPTZ) <= CAST(\'{CUTOFF}\' AS TIMESTAMPTZ)
         THEN \'train\' ELSE \'test\' END AS split,

    -- Keys for historical lookups
    jnam,
    usr,

    -- Original submission features
    LN(1 + nnumr)    AS log_nnumr,
    LN(1 + nnuma)    AS log_nnuma,
    LN(1 + cnumr)    AS log_cnumr,
    LN(1 + elpl)     AS log_elpl,
    LN(1 + CASE WHEN mszl < 0 OR mszl > 1e18 THEN NULL ELSE mszl END) AS log_mszl,
    LN(1 + CASE WHEN CAST(msza AS DOUBLE) > 1e18 THEN NULL ELSE CAST(msza AS DOUBLE) END) AS log_msza,
    CASE WHEN pclass = \'compute-bound\' THEN 1 ELSE 0 END AS is_compute_bound,
    COALESCE(freq_req, 0) AS freq_req,

    -- Time features from qdt (submission timestamp)
    HOUR(CAST(qdt AS TIMESTAMPTZ))                           AS hour_of_day,
    DAYOFWEEK(CAST(qdt AS TIMESTAMPTZ))                      AS day_of_week,

    -- Queue wait time: sdt - qdt in seconds
    DATEDIFF(\'second\',
        CAST(qdt AS TIMESTAMPTZ),
        CAST(sdt AS TIMESTAMPTZ))                            AS queue_wait_secs,

    -- Node size bucket (for bucket-level failure rate lookup)
    CASE
        WHEN nnumr = 1            THEN \'single\'
        WHEN nnumr BETWEEN 2 AND 32  THEN \'small\'
        WHEN nnumr BETWEEN 33 AND 512 THEN \'medium\'
        ELSE                          \'large\'
    END AS node_bucket,

    -- Labels
    CASE WHEN "exit state" = \'failed\' THEN 1 ELSE 0 END AS failed,
    LN(1 + duration)  AS log_duration,
    LN(1 + econ)      AS log_econ,
    CASE WHEN avgpcon BETWEEN 10 AND 200000 THEN avgpcon ELSE NULL END AS avgpcon_clean,
    CASE WHEN "exit state" = \'failed\' THEN nnumr * duration / 3600.0 ELSE 0 END AS wasted_node_hours,
    CASE
        WHEN "exit state" = \'completed\' THEN 0
        WHEN "exit state" = \'failed\' AND duration < 300   THEN 1
        WHEN "exit state" = \'failed\' AND duration < 7200  THEN 2
        ELSE 3
    END AS fail_type

FROM raw
WHERE sdt IS NOT NULL AND duration > 0 AND elpl > 0
""")
print("  Done.")

# ── 2. Compute historical rates from TRAIN only ───────────────
print("Computing historical failure rates from training split...")

# jnam failure rate (smooth with global mean for rare jnames)
global_mean = db.execute(
    "SELECT AVG(failed) FROM enriched WHERE split=\'train\'"
).fetchone()[0]

db.execute(f"""
CREATE OR REPLACE TABLE jnam_stats AS
SELECT jnam,
       COUNT(*)        AS jnam_count,
       AVG(failed)     AS jnam_fail_rate
FROM enriched WHERE split = \'train\'
GROUP BY jnam
""")

db.execute(f"""
CREATE OR REPLACE TABLE usr_stats AS
SELECT usr,
       COUNT(*)        AS usr_count,
       AVG(failed)     AS usr_fail_rate
FROM enriched WHERE split = \'train\'
GROUP BY usr
""")

db.execute(f"""
CREATE OR REPLACE TABLE bucket_stats AS
SELECT node_bucket,
       AVG(failed)     AS bucket_fail_rate
FROM enriched WHERE split = \'train\'
GROUP BY node_bucket
""")

# ── 3. Join historical features back ─────────────────────────
print("Joining historical features...")
db.execute(f"""
CREATE OR REPLACE TABLE final AS
SELECT
    e.split,
    e.log_nnumr, e.log_nnuma, e.log_cnumr, e.log_elpl,
    e.log_mszl, e.log_msza, e.is_compute_bound, e.freq_req,
    e.hour_of_day, e.day_of_week,
    LN(1 + GREATEST(queue_wait_secs, 0)) AS log_queue_wait,

    -- Historical failure rates (fall back to global mean if unseen)
    COALESCE(j.jnam_fail_rate,    {global_mean}) AS jnam_fail_rate,
    COALESCE(u.usr_fail_rate,     {global_mean}) AS usr_fail_rate,
    COALESCE(b.bucket_fail_rate,  {global_mean}) AS bucket_fail_rate,

    -- jnam frequency (how busy is this job type)
    LN(1 + COALESCE(j.jnam_count, 1)) AS log_jnam_count,

    -- Labels
    e.failed, e.fail_type, e.log_duration, e.log_econ,
    e.avgpcon_clean, e.wasted_node_hours

FROM enriched e
LEFT JOIN jnam_stats  j USING (jnam)
LEFT JOIN usr_stats   u USING (usr)
LEFT JOIN bucket_stats b USING (node_bucket)
""")

# ── 4. Export for reuse ───────────────────────────────────────
print("Exporting enriched parquets...")
db.execute(f"""
    COPY (SELECT * FROM final WHERE split=\'train\')
    TO \'{PREPARED_DIR}train_v2.parquet\' (FORMAT PARQUET, COMPRESSION ZSTD)
""")
db.execute(f"""
    COPY (SELECT * FROM final WHERE split=\'test\')
    TO \'{PREPARED_DIR}test_v2.parquet\' (FORMAT PARQUET, COMPRESSION ZSTD)
""")

# ── 5. Train XGBoost ──────────────────────────────────────────
FEATURES_V2 = [
    "log_nnumr", "log_nnuma", "log_cnumr", "log_elpl",
    "log_mszl", "log_msza", "is_compute_bound", "freq_req",
    "hour_of_day", "day_of_week", "log_queue_wait",
    "jnam_fail_rate", "usr_fail_rate", "bucket_fail_rate",
    "log_jnam_count"
]

feat_cols = ", ".join(FEATURES_V2)
train_df = db.execute(
    f"SELECT {feat_cols}, failed FROM final WHERE split='train'"
).df()
test_df  = db.execute(
    f"SELECT {feat_cols}, failed FROM final WHERE split='test'"
).df()

X_train, y_train = train_df[FEATURES_V2].values, train_df["failed"].values
X_test,  y_test  = test_df[FEATURES_V2].values,  test_df["failed"].values

print(f"\nTrain: {X_train.shape}  failures={y_train.mean()*100:.1f}%")
print(f"Test:  {X_test.shape}   failures={y_test.mean()*100:.1f}%")

scale = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale:.2f}")

print("\nTraining XGBoost v2...")
model = xgb.XGBClassifier(
    n_estimators         = 1000,
    max_depth            = 8,
    learning_rate        = 0.05,
    subsample            = 0.8,
    colsample_bytree     = 0.8,
    min_child_weight     = 50,
    scale_pos_weight     = scale,
    eval_metric          = "aucpr",
    early_stopping_rounds= 30,
    n_jobs               = -1,
    random_state         = 42,
    device               = "cuda",
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

# ── 6. Evaluate ───────────────────────────────────────────────
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

auc_roc = roc_auc_score(y_test, y_prob)
auc_pr  = average_precision_score(y_test, y_prob)

print(f"\n── EVALUATION ──")
print(f"  AUC-ROC : {auc_roc:.4f}")
print(f"  AUC-PR  : {auc_pr:.4f}  (baseline={y_test.mean():.3f})")
print(classification_report(y_test, y_pred, target_names=["completed","failed"]))

cm = confusion_matrix(y_test, y_pred)
print(f"  TN={cm[0,0]:>8,}  FP={cm[0,1]:>8,}")
print(f"  FN={cm[1,0]:>8,}  TP={cm[1,1]:>8,}")

print("\n── FEATURE IMPORTANCE ──")
for name, score in sorted(zip(FEATURES_V2, model.feature_importances_), key=lambda x: -x[1]):
    bar = "█" * int(score * 300)
    print(f"  {name:<22}  {score:.4f}  {bar}")

# Save
model_path = MODEL_DIR + "failure_classifier_v2.json"
model.save_model(model_path)
eval_out = {
    "model": "failure_classifier_v2", "features": FEATURES_V2,
    "auc_roc": round(auc_roc,4), "auc_pr": round(auc_pr,4),
    "train_failure_rate": round(float(y_train.mean()),4),
    "test_failure_rate": round(float(y_test.mean()),4),
    "best_iteration": model.best_iteration
}
with open(MODEL_DIR + "failure_classifier_v2_eval.json", "w") as f:
    json.dump(eval_out, f, indent=2)
print(f"\nSaved: {model_path}")
print("Done.")
