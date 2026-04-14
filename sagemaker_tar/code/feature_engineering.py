import pandas as pd
import numpy as np

def add_time_features(df):
    """
    Add cyclical hour features

    Args:
        df: input dataframe with 'timestamp'

    Returns:
        df with hour, hour_sin, hour_cos
    """

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df


# =========================================
# OPERATION ENCODING
# =========================================
def encode_operation(df):
    """
    One-hot encode Operations column

    Args:
        df: input dataframe
        prefix: prefix for columns

    Returns:
        df with encoded columns
    """

    df = df.copy()

    df = pd.get_dummies(df, columns=["operation"])

    return df

# =========================================
# FEATURE PREPARATION (CORE ENTRY POINT)
# =========================================
def prepare_features(df):
    """
    Full feature pipeline

    Args:
        df: raw dataframe

    Returns:
        df with features
    """

    df = add_time_features(df)
    df = encode_operation(df)
    df = df.drop(columns=["timestamp"])

    return df

# =========================================
# FEATURES FOR TRAINING MODEL
# =========================================
def get_features_for_target(target, features):
    """
    Features: ALL_FEATURES

    Args:
        target: target metric
        features: All features

    Returns:
        features without target feature
    """

    return [f for f in features if f != target]