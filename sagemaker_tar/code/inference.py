import json
import pandas as pd
import joblib
import os
import sys
import subprocess

# Allow local imports
sys.path.append(os.path.dirname(__file__))
subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])

from feature_engineering import prepare_features


# =========================
# LOAD MODEL (ONCE)
# =========================
def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model_bundle.pkl")

    print("Loading model from:", model_path)
    print("Files:", os.listdir(model_dir))

    bundle = joblib.load(model_path)
    return bundle


# =========================
# INPUT
# =========================
def input_fn(request_body, request_content_type):
    data = json.loads(request_body)
    return pd.DataFrame(data)


# =========================
# HELPER
# =========================
def get_severity_label(severity):
    if severity < 1.25:
        return "Low"
    elif severity < 2.5:
        return "⚠️ Medium"
    else:
        return "🚨 Critical"


# =========================
# PREDICT
# =========================
def predict_fn(df, bundle):

    models = bundle["models"]
    features = bundle["features"]
    targets = bundle["targets"]
    thresholds = bundle.get("thresholds", {})

    MIN_PERCENT = 0.05
    LATENCY_MULT = 1.2
    SUCCESS_VOL_MULT = 1.2
    FAIL_VOL_MULT = 1.2

    df_raw = df.copy()
    df = prepare_features(df)

    all_required_features = set()
    for t in targets:
        all_required_features.update(features[t])

    for col in all_required_features:
        if col not in df.columns:
            df[col] = 0

    results = []

    for idx, row in df.iterrows():

        op = df_raw.loc[idx, "operation"]
        hour = pd.to_datetime(df_raw.loc[idx, "timestamp"]).hour

        success_vol = df_raw.loc[idx, "success_vol"]
        fail_vol = df_raw.loc[idx, "fail_vol"]
        success_rt_avg = df_raw.loc[idx, "success_rt_avg"]
        fail_rt_avg = df_raw.loc[idx, "fail_rt_avg"]

        detected_anomaly = False
        max_severity = 0
        root_causes = []

        preds = {}

        for t in targets:

            feat = features[t]

            pred = models[t].predict(row[feat].values.reshape(1, -1))[0]
            actual = row[t]

            preds[f"pred_{t}"] = pred

            # =========================
            # THRESHOLD LOGIC 
            # =========================
            rule = thresholds.get(op, {}).get(hour, {}).get(t)

            if rule is None:
                continue

            threshold_val = max(
                pred * rule["percent_threshold"],
                rule["abs_threshold"],
                MIN_PERCENT * pred
            )

            deviation = None

            # =========================
            # LATENCY
            # =========================
            if t in ["success_rt_avg", "fail_rt_avg"]:
                threshold_val *= LATENCY_MULT

                deviation = actual - pred
                if deviation <= 0:
                    continue

            # =========================
            # SUCCESS VOL
            # =========================
            elif t == "success_vol":
                threshold_val *= SUCCESS_VOL_MULT
                deviation = abs(actual - pred)

            # =========================
            # FAIL VOL
            # =========================
            elif t == "fail_vol":
                threshold_val *= FAIL_VOL_MULT

                deviation = actual - pred
                if deviation <= 0:
                    continue

            # =========================
            # CHECK
            # =========================
            if deviation is not None and deviation > threshold_val:

                detected_anomaly = True

                severity = deviation / (threshold_val + 1e-6)
                root_causes.append((t, severity))
                max_severity = max(max_severity, severity)

        # Sort root causes
        root_causes = sorted(root_causes, key=lambda x: x[1], reverse=True)

        results.append({
            "operation": op,
            "hour": hour,

            "success_vol": success_vol,
            "fail_vol": fail_vol,
            "success_rt_avg": success_rt_avg,
            "fail_rt_avg": fail_rt_avg,

            **preds,

            "Status": "Anomaly" if detected_anomaly else "Normal ✅",

            "Root_Cause": root_causes[0][0] if root_causes else None,
            "All_Causes": ", ".join([f"{k}:{v:.2f}" for k, v in root_causes]) if root_causes else None,
            "Severity": round(max_severity, 3),
            "Severity_Label": get_severity_label(max_severity)
        })

    return pd.DataFrame(results)


# =========================
# OUTPUT
# =========================
def output_fn(prediction, accept):
    return prediction.to_json(orient="records")