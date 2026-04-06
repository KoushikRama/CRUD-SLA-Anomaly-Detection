import pandas as pd
import joblib

from src.common.feature_engineering import prepare_features
from src.common.load_main_config import get_model_path, load_data_config, load_threshold_config
from datetime import datetime
from src.data.generate_test_data import generate_test_data


# =========================================
# LOAD MODEL
# =========================================
def load_model():
    return joblib.load(get_model_path())


def load_bundle_parameters(bundle):
    models = bundle["models"]
    features = bundle["features"]
    targets = bundle["targets"]
    thresholds = bundle.get("thresholds", {})

    return models, features, targets, thresholds


# =========================================
# MAIN INFERENCE
# =========================================
def run_inference(df):

    bundle = load_model()
    models, features, targets, thresholds = load_bundle_parameters(bundle)

    # ✅ Load threshold config ONCE (optimization)
    threshold_cfg = load_threshold_config()

    # Keep raw copy
    df_raw = df.copy()

    # Feature engineering
    df = prepare_features(df)

    results = []

    for idx, row in df.iterrows():

        op = df_raw.loc[idx, "operation"]
        hour = pd.to_datetime(df_raw.loc[idx, "timestamp"]).hour

        success_vol = df_raw.loc[idx, "success_vol"]
        fail_vol = df_raw.loc[idx, "fail_vol"]
        success_rt_avg = df_raw.loc[idx, "success_rt_avg"]
        fail_rt_avg = df_raw.loc[idx, "fail_rt_avg"]

        is_anomaly = row.get("is_anomaly", None)
        anomaly_type = row.get("anomaly_type", None)

        detected_anomaly = False
        max_severity = 0
        root_causes = []

        preds = {}

        for t in targets:

            feat = features[t]

            pred = models[t].predict(row[feat].values.reshape(1, -1))[0]
            actual = row[t]

            preds[f"pred_{t}"] = pred

            rule = thresholds.get(op, {}).get(hour, {}).get(t)
            if rule is None:
                continue

            min_pct = threshold_cfg["threshold_rules"]["min_percent_of_pred"]

            threshold_val = max(
                pred * rule["percent_threshold"],
                rule["abs_threshold"],
                min_pct * pred
            )

            deviation = 0

            # =========================
            # LATENCY
            # =========================
            if t in ["success_rt_avg", "fail_rt_avg"]:

                latency_cfg = threshold_cfg["threshold_rules"]["latency"]

                threshold_val *= latency_cfg["multiplier"]

                deviation = actual - pred
                if deviation <= 0:
                    continue

            # =========================
            # SUCCESS VOL
            # =========================
            elif t == "success_vol":
                threshold_val *= threshold_cfg["threshold_rules"]["success_vol"]["multiplier"]
                deviation = abs(actual - pred)

            # =========================
            # FAIL VOL
            # =========================
            elif t == "fail_vol":
                threshold_val *= threshold_cfg["threshold_rules"]["fail_vol"]["multiplier"]

                deviation = actual - pred
                if deviation <= 0:
                    continue

            # =========================
            # ABSOLUTE CHECK
            # =========================
            if deviation > threshold_val:

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

            "Status": "Anomaly 🚨" if detected_anomaly else "Normal ✅",

            # ✅ SAFE ACCESS FIX
            "Root_Cause": root_causes[0][0] if root_causes else None,
            "All_Causes": root_causes if root_causes else None,
            "Severity": round(max_severity, 3),

            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type
        })

    return pd.DataFrame(results)


# =========================================
# MAIN
# =========================================
def main():

    config = load_data_config()

    df_test = generate_test_data(datetime(2025, 4, 1), 48, config)

    results = run_inference(df_test)

    print(results.head())


if __name__ == "__main__":
    main()