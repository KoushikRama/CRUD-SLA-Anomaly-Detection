import pandas as pd


def evaluate(results_df):

    df = results_df.copy()

    # ----------------------------
    # SAFETY CHECK
    # ----------------------------
    if "is_anomaly" not in df.columns:
        raise ValueError("Missing 'is_anomaly' column in results")

    # ----------------------------
    # CORE METRICS
    # ----------------------------
    df["predicted"] = df["Status"] != "Normal ✅"
    df["actual"] = df["is_anomaly"]

    total = len(df)
    alerts = df["predicted"].sum()
    actual = df["actual"].sum()

    tp = ((df["predicted"]) & (df["actual"])).sum()
    fp = ((df["predicted"]) & (~df["actual"])).sum()
    fn = ((~df["predicted"]) & (df["actual"])).sum()
    tn = ((~df["predicted"]) & (~df["actual"])).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    alert_rate = alerts / total

    # ----------------------------
    # PRINT SUMMARY
    # ----------------------------
    print("\n===== PERFORMANCE =====")
    print(f"Total: {total}")
    print(f"Actual Anomalies: {actual}")
    print(f"Detected Alerts: {alerts}")

    print(f"\nPrecision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Alert Rate: {alert_rate:.3f}")

    print("\n===== CONFUSION MATRIX =====")
    print(f"TP: {tp} | FP: {fp}")
    print(f"FN: {fn} | TN: {tn}")

    # ----------------------------
    # 🔥 PER-ANOMALY TYPE ANALYSIS
    # ----------------------------
    if "anomaly_type" in df.columns:

        print("\n===== PER ANOMALY TYPE =====")

        anomaly_types = df[df["actual"]]["anomaly_type"].unique()

        for atype in anomaly_types:
            subset = df[df["anomaly_type"] == atype]

            if len(subset) == 0:
                continue

            tp_t = ((subset["predicted"]) & (subset["actual"])).sum()
            fn_t = ((~subset["predicted"]) & (subset["actual"])).sum()

            recall_t = tp_t / (tp_t + fn_t + 1e-6)

            print(f"{atype}: Recall = {recall_t:.3f}")

    # ----------------------------
    # RETURN METRICS (IMPORTANT)
    # ----------------------------
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alert_rate": alert_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }