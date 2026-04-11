import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib
from src.common.s3_utils import upload_to_s3
from src.common.load_main_config import load_main_config

from src.common.feature_engineering import (
    prepare_features,
    get_features_for_target
)

from src.common.load_main_config import get_data_filepath , get_model_path, load_data_config, load_model_config

# =========================================
# LOAD DATA
# =========================================
def load_training_data():

    filepath = get_data_filepath()

    return pd.read_csv(filepath)

# =========================================
# TRAIN MODEL
# =========================================
def train_model(X_train, y_train, X_val, y_val):

    model_cfg = load_model_config()["xgboost"]
    
    model = XGBRegressor(**model_cfg)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model

# =========================================
# TRAIN ALL TARGETS
# =========================================
def train_all_models(df, TARGETS):
    models = {}
    metrics = {}
    feature_map={}

    df = prepare_features(df)

    ALL_FEATURES = [c for c in df.columns]

    model_cfg = load_model_config()["training"]

    for target in TARGETS:
        
        print(f"\nTraining model for: {target}")
        
        FEATURES = get_features_for_target(target, ALL_FEATURES)
        
        X = df[FEATURES]
        y = df[target]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=model_cfg["test_size"], random_state=model_cfg["random_state"]
        )
        
        model = train_model(X_train, y_train, X_val, y_val)
        
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        
        models[target] = model
        feature_map[target] = FEATURES
        metrics[target] = mae
        print(f"{target} MAE: {mae:.4f}")
    
    return models, feature_map

# =========================================
# SAVE MODEL
# =========================================
def save_model(models,features,targets):
    bundle = {
        "models": models,
        "features": features,
        "targets": targets,
    }

    model_path = get_model_path()
    # create dir
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    print("Model Saved to:",model_path)


# =========================================
# MAIN
# =========================================
def main():

    cfg = load_data_config()
    targets = cfg["targets"]
    df = load_training_data()

    models, features = train_all_models(df, targets)

    save_model(models,features,targets)

    print("Training commplete")

    config = load_main_config()

    if config["flags"]["upload_to_s3"]:
        bucket = config["s3"]["bucket"]
        prefix = config["s3"]["prefix"]

        model_path = get_model_path()

        upload_to_s3(model_path, bucket, f"{prefix}/model_bundle.pkl")


# =========================================
# RUN
# =========================================
if __name__ == "__main__":
    main()
