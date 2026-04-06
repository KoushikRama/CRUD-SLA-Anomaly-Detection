import pandas as pd

from src.common.feature_engineering import prepare_features
from src.common.load_main_config import get_model_path, get_data_filepath, load_threshold_config
import joblib


def compute_errors(df, targets):
    for t in targets:
        df[f"{t}_error"] = abs(df[t] - df[f"{t}_pred"])
        df[f"{t}_ratio"] = df[f"{t}_error"] / (df[f"{t}_pred"] + 1e-6)
    return df


def decode_operation(df):
    op_cols = [c for c in df.columns if c.startswith("operation_")]

    df["operation"] = df[op_cols].idxmax(axis=1)
    df["operation"] = df["operation"].str.replace("operation_", "")

    return df

def generate_thresholds(df, targets):

    cfg = load_threshold_config()

    thresholds = {}
    grouped = df.groupby(["operation", "hour"])

    for (op, hour), group in grouped:

        thresholds.setdefault(op, {})
        thresholds[op].setdefault(hour, {})

        for t in targets:

            q = cfg["quantile"]["per_target"].get(
                t, cfg["quantile"]["default"]
            )

            thresholds[op][hour][t] = {
                "percent_threshold": float(group[f"{t}_ratio"].quantile(q)),
                "abs_threshold": float(group[f"{t}_error"].quantile(q)),
            }

    return thresholds


def main():

    bundle = joblib.load(get_model_path())
    models = bundle["models"]
    features = bundle["features"]
    targets = bundle["targets"]

    df = pd.read_csv(get_data_filepath())  # processed data

    df = prepare_features(df)

    # predictions
    for t in targets:
        df[f"{t}_pred"] = models[t].predict(df[features[t]])

    df = compute_errors(df, targets)
    df = decode_operation(df)

    thresholds = generate_thresholds(df, targets)

    bundle["thresholds"] = thresholds

    joblib.dump(bundle, get_model_path())

    print("Thresholds added to bundle")

# =========================================
# RUN
# =========================================
if __name__ == "__main__":
    main()
