"""
predict.py
Single-job inference. Load once, call many times.

Usage:
    from predict import Predictor
    p = Predictor()
    result = p.predict({
        "nnumr": 512, "nnuma": 512, "cnumr": 2048, "elpl": 86400,
        "pclass": "compute-bound", "mszl": 0, "msza": 0,
        "freq_req": 0, "pri": 0, "jobenv_req": "jobenv_req_0",
        "usr": "user_1234", "jnam": "md_simulation_water",
        "qdt": "2024-01-15 14:30:00",
    })
"""
import os, math, json, pickle
import numpy as np
import duckdb

MODEL_DIR   = os.getenv("MODELS_PATH", "models/")
PCA_PATH    = os.path.join(os.getenv("PREPARED_PATH", "data/prepared/"), "embedding_pca.pkl")
GLOBAL_RATE = 0.0989
GLOBAL_LOG_NNUMR = math.log(2)
GLOBAL_LOG_ELPL  = math.log1p(7200)


class Predictor:
    def __init__(self):
        print("Loading models...", end=" ", flush=True)
        self.m1   = pickle.load(open(MODEL_DIR + "s1_failure.pkl",           "rb"))
        self.iso1 = pickle.load(open(MODEL_DIR + "s1_calibrator.pkl",        "rb"))
        self.m2   = pickle.load(open(MODEL_DIR + "s2_failcost.pkl",          "rb"))
        self.iso2 = pickle.load(open(MODEL_DIR + "s2_calibrator.pkl",        "rb"))
        self.m3   = pickle.load(open(MODEL_DIR + "s3_runtime_completed.pkl", "rb"))
        self.m4   = pickle.load(open(MODEL_DIR + "s4_energy.pkl",            "rb"))
        self.pca  = pickle.load(open(PCA_PATH,                               "rb"))

        t = json.load(open(MODEL_DIR + "s1_thresholds.json"))
        self.t_caution = t["caution"]   # 0.10
        self.t_warning = t["warning"]   # 0.50

        reg           = json.load(open(MODEL_DIR + "feature_registry.json"))
        self.FEATURES = reg["features"]
        self.N_PCA    = reg["n_pca"]

        self.runtime_lk = json.load(open(MODEL_DIR + "failed_runtime_lookup.json"))

        # user stats
        db   = duckdb.connect()
        rows = db.execute(
            "SELECT * FROM read_parquet('" + MODEL_DIR + "user_stats.parquet')"
        ).fetchall()
        cols = [d[0] for d in db.execute(
            "SELECT * FROM read_parquet('" + MODEL_DIR + "user_stats.parquet') LIMIT 0"
        ).description]
        db.close()
        self.user_stats = {r[0]: dict(zip(cols[1:], r[1:])) for r in rows}

        # jnam embeddings
        self.jnam_emb = pickle.load(open(MODEL_DIR + "jnam_emb.pkl", "rb"))

        print(f"ready. ({len(self.user_stats):,} users, {len(self.jnam_emb):,} job names)")

    # ── feature construction ──────────────────────────────────────
    def _featurize(self, job: dict) -> np.ndarray:
        nnumr = max(int(job.get("nnumr", 1)), 1)
        nnuma = max(int(job.get("nnuma", nnumr)), 1)
        cnumr = max(int(job.get("cnumr", nnumr * 4)), 1)
        elpl  = max(float(job.get("elpl", 3600)), 1)
        mszl  = float(job.get("mszl", 0) or 0)
        msza  = float(job.get("msza", 0) or 0)
        pclass   = job.get("pclass", "memory-bound")
        is_cb    = 1.0 if pclass == "compute-bound" else 0.0
        freq_req = float(job.get("freq_req", 0) or 0)
        pri      = float(job.get("pri", 0) or 0)
        jenv     = str(job.get("jobenv_req", "jobenv_req_0"))

        log_nnumr = math.log1p(nnumr)
        log_nnuma = math.log1p(nnuma)
        log_cnumr = math.log1p(cnumr)
        log_elpl  = math.log1p(elpl)
        log_mszl  = math.log1p(mszl) if 0 < mszl < 1e18 else 0.0
        log_msza  = math.log1p(msza) if 0 < msza < 1e18 else 0.0

        elpl_per_node   = log_elpl - log_nnumr
        compute_x_nodes = is_cb * log_nnumr
        node_bucket = (0 if log_nnumr < 0.693 else
                       1 if log_nnumr < 3.497 else
                       2 if log_nnumr < 6.238 else 3)

        from datetime import datetime
        try:
            qdt_str = str(job.get("qdt", "2023-01-01 12:00:00"))
            qdt_str = qdt_str.split("+")[0].replace("Z", "")
            dt = datetime.fromisoformat(qdt_str)
        except Exception:
            dt = datetime(2023, 1, 1, 12, 0, 0)
        hour_of_day = float(dt.hour)
        day_of_week = float(dt.weekday())
        is_weekend  = 1.0 if day_of_week >= 5 else 0.0

        usr    = job.get("usr", "__unknown__")
        us     = self.user_stats.get(usr, {})
        ufr    = us.get("user_fail_rate",         GLOBAL_RATE)
        ucfr   = us.get("user_compute_fail_rate", GLOBAL_RATE)
        umfr   = us.get("user_memory_fail_rate",  GLOBAL_RATE)
        urrfr  = us.get("user_recent_fail_rate",  GLOBAL_RATE)
        ualnr  = us.get("user_avg_log_nnumr",     GLOBAL_LOG_NNUMR)
        ualel  = us.get("user_avg_log_elpl",      GLOBAL_LOG_ELPL)
        unj    = float(us.get("user_n_jobs",      0))
        nnumr_anomaly = log_nnumr - ualnr

        jnam = str(job.get("jnam", ""))
        emb  = (self.jnam_emb[jnam].astype(np.float32)
                if jnam in self.jnam_emb
                else np.zeros(self.N_PCA, dtype=np.float32))

        tab = {
            "log_nnumr": log_nnumr, "log_nnuma": log_nnuma,
            "log_cnumr": log_cnumr, "log_elpl":  log_elpl,
            "log_mszl":  log_mszl,  "log_msza":  log_msza,
            "is_compute_bound":   is_cb,
            "freq_req":  freq_req,  "pri": pri,
            "jobenv_0":  1.0 if jenv == "jobenv_req_0" else 0.0,
            "jobenv_1":  1.0 if jenv == "jobenv_req_1" else 0.0,
            "hour_of_day":  hour_of_day,
            "day_of_week":  day_of_week,
            "is_weekend":   is_weekend,
            "elpl_per_node":    elpl_per_node,
            "compute_x_nodes":  compute_x_nodes,
            "node_bucket":      float(node_bucket),
            "user_fail_rate":         ufr,
            "user_compute_fail_rate": ucfr,
            "user_memory_fail_rate":  umfr,
            "user_recent_fail_rate":  urrfr,
            "user_avg_log_nnumr":     ualnr,
            "user_avg_log_elpl":      ualel,
            "user_n_jobs":            unj,
            "nnumr_anomaly":          nnumr_anomaly,
        }
        for i in range(self.N_PCA):
            tab[f"emb_{i}"] = float(emb[i])

        return np.array([tab[f] for f in self.FEATURES], dtype=np.float32).reshape(1, -1)

    # ── plain-english reasons ─────────────────────────────────────
    def _reasons(self, job: dict) -> list:
        usr  = job.get("usr", "__unknown__")
        us   = self.user_stats.get(usr, {})
        nnumr  = int(job.get("nnumr", 1))
        pclass = job.get("pclass", "memory-bound")
        jnam   = str(job.get("jnam", ""))
        reasons = []

        ufr  = us.get("user_fail_rate",         GLOBAL_RATE)
        ucfr = us.get("user_compute_fail_rate",  GLOBAL_RATE)
        umfr = us.get("user_memory_fail_rate",   GLOBAL_RATE)
        urrfr = us.get("user_recent_fail_rate",  GLOBAL_RATE)
        ualnr = us.get("user_avg_log_nnumr",     GLOBAL_LOG_NNUMR)
        unj   = us.get("user_n_jobs", 0)

        if unj == 0:
            reasons.append("New user — no historical data (cold start, using global averages)")
        elif unj < 10:
            reasons.append(f"Limited user history ({int(unj)} jobs only)")
        elif ufr > 0.25:
            reasons.append(f"User historical failure rate is high ({100*ufr:.0f}%)")
        elif ufr < 0.03:
            reasons.append(f"User has very low failure rate ({100*ufr:.1f}%) — low base risk")

        if pclass == "compute-bound" and ucfr > 0.20:
            reasons.append(f"User's compute-bound jobs fail at {100*ucfr:.0f}%")
        if pclass == "memory-bound" and umfr > 0.20:
            reasons.append(f"User's memory-bound jobs fail at {100*umfr:.0f}%")

        if ualnr and math.log1p(nnumr) > ualnr + 1.5:
            ratio = math.exp(math.log1p(nnumr) - ualnr)
            reasons.append(f"Job is {ratio:.1f}× larger than this user's typical submission")

        if urrfr > ufr + 0.15:
            reasons.append(
                f"Recent fail rate ({100*urrfr:.0f}%) >> historical ({100*ufr:.0f}%) "
                f"— user may be actively debugging"
            )

        if jnam and jnam in self.jnam_emb:
            reasons.append(f"Job name '{jnam}' matches patterns seen in training data")
        elif jnam:
            reasons.append(f"Job name '{jnam}' is unseen — embedding defaults to neutral")

        return reasons[:4]

    # ── main predict ─────────────────────────────────────────────
    def predict(self, job: dict) -> dict:
        X = self._featurize(job)

        # Stage 1 — failure probability
        p_raw  = float(self.m1.predict(X)[0])
        p_fail = float(np.clip(self.iso1.transform([p_raw])[0], 0.0, 1.0))

        if   p_fail >= self.t_warning: risk = "WARNING"
        elif p_fail >= self.t_caution: risk = "CAUTION"
        else:                          risk = "OK"

        # Stage 2 — fail cost (only meaningful if flagged)
        X2          = np.hstack([X, [[p_fail]]]).astype(np.float32)
        p_exp_r     = float(self.m2.predict(X2)[0])
        p_expensive = float(np.clip(self.iso2.transform([p_exp_r])[0], 0.0, 1.0))

        if   p_expensive >= 0.70: fail_tier = "expensive — likely slow (>2hr)"
        elif p_expensive >= 0.40: fail_tier = "uncertain — could be slow or quick"
        else:                     fail_tier = "cheap — likely quick (<5min)"

        # Stage 3 — expected runtime if completed
        log_dur   = float(self.m3.predict(X)[0])
        exp_dur_s = math.expm1(max(log_dur, 0))

        # Stage 4 — expected energy
        # log_econ     = float(self.m4.predict(X)[0])
        # exp_energy_j = math.expm1(max(log_econ, 0))
        
        # Stage 4 — expected energy
        # Model is reliable for avg-range jobs (≤200 nodes).
        # For large jobs it underestimates massively due to training skew.
        # Floor: nnumr × elpl × MIN_WATTS_PER_NODE (conservative 50 W/node idle).
        # Ceil:  nnumr × elpl × MAX_WATTS_PER_NODE (aggressive 500 W/node peak).
        MIN_W_PER_NODE = 50.0
        MAX_W_PER_NODE = 500.0
        log_econ     = float(self.m4.predict(X)[0])
        exp_energy_j = math.expm1(max(log_econ, 0))
        physics_floor = job["nnumr"] * job["elpl"] * MIN_W_PER_NODE
        physics_ceil  = job["nnumr"] * job["elpl"] * MAX_W_PER_NODE
        if exp_energy_j < physics_floor:
            exp_energy_j = physics_floor   # model collapsed — use physics floor
        elif exp_energy_j > physics_ceil:
            exp_energy_j = physics_ceil    # model exploded — cap at physics ceiling

        # Estimated node-hours wasted if it fails slowly
        nnumr       = max(int(job.get("nnumr", 1)), 1)
        slow_median = self.runtime_lk.get("3", {}).get("median_s", 24716)
        wasted_nh   = nnumr * slow_median / 3600.0

        def fmt_dur(seconds: float) -> str:
            if seconds < 60:
                return f"{seconds:.1f} sec"
            elif seconds < 3600:
                minutes = seconds / 60
                return f"{minutes:.1f} min"
            else:
                hours = seconds / 3600
                return f"{hours:.2f} hr"

        def fmt_energy(j):
            if j < 1e3:   return f"{j:.0f} J"
            if j < 1e6:   return f"{j/1e3:.1f} kJ"
            if j < 3.6e9: return f"{j/1e6:.1f} MJ"
            return f"{j/3.6e9:.2f} MWh"

        return {
            "risk_level":              risk,
            "p_fail":                  round(p_fail, 3),
            "fail_type_if_fails":      fail_tier,
            "p_expensive_if_fails":    round(p_expensive, 3),
            "expected_runtime":        fmt_dur(exp_dur_s),
            "expected_energy":         fmt_energy(exp_energy_j),
            "wasted_node_hrs_if_slow": f"~{wasted_nh:,.0f}",
            "top_reasons":             self._reasons(job),
            "_raw": {
                "p_fail":       round(p_fail, 4),
                "p_expensive":  round(p_expensive, 4),
                "exp_dur_s":    round(exp_dur_s, 1),
                "exp_energy_j": round(exp_energy_j, 1),
                "wasted_nh":    round(wasted_nh, 2),
            }
        }


# ── self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint
    p = Predictor()

    tests = [
        {
            "label": "MD simulation — large compute (expect WARNING)",
            "job":   dict(nnumr=512, nnuma=512, cnumr=2048, elpl=86400,
                          pclass="compute-bound", usr="user_0042",
                          jnam="md_simulation_water", qdt="2024-01-15 14:30:00"),
        },
        {
            "label": "Small postprocess job (expect OK)",
            "job":   dict(nnumr=4, nnuma=4, cnumr=16, elpl=3600,
                          pclass="memory-bound", usr="user_0100",
                          jnam="data_postprocess", qdt="2024-01-15 09:00:00"),
        },
        {
            "label": "New user cold start (expect CAUTION + cold-start reason)",
            "job":   dict(nnumr=64, nnuma=64, cnumr=256, elpl=43200,
                          pclass="compute-bound", usr="brand_new_user_xyz",
                          jnam="unknown_application", qdt="2024-01-16 22:00:00"),
        },
    ]

    for t in tests:
        print(f"\n{'='*62}")
        print(f"  {t['label']}")
        print(f"{'='*62}")
        pprint.pprint(p.predict(t["job"]), width=72, sort_dicts=False)
