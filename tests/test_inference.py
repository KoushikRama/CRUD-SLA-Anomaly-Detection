import pandas as pd

from src.xgboost.inference.infer import run_inference, load_bundle_parameters, load_model
from src.common.feature_engineering import prepare_features


def test_inference(df):

    # Load bundle once
    bundle = load_model()
    _, features, targets, _ = load_bundle_parameters(bundle)

    # Keep raw
    df_raw = df.copy()

    # Apply feature engineering
    df_feat = prepare_features(df_raw)

    # 🔥 FIX: align features
    all_required_features = set()
    for t in targets:
        all_required_features.update(features[t])

    for col in all_required_features:
        if col not in df_feat.columns:
            df_feat[col] = 0

    # Optional: enforce order
    df_feat = df_feat[list(all_required_features)]

    print("Expected:", list(all_required_features)[:5])
    print("Actual:", list(df_feat.columns)[:5])

    # Run original inference
    return run_inference(df_feat)