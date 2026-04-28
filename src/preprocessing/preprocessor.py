import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.ingestion.data_loader import get_sensor_columns, load_config, load_raw_data


def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    drop_cols = [c for c in config["data"]["drop_columns"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    sensor_cols = get_sensor_columns(df, config)

    # Drop rows where all sensor values are NaN
    df = df.dropna(subset=sensor_cols, how="all")

    # Fill remaining NaNs with column median
    df[sensor_cols] = df[sensor_cols].fillna(df[sensor_cols].median())

    # Remove constant columns (no variance)
    constant_cols = [c for c in sensor_cols if df[c].std() == 0]
    if constant_cols:
        logger.warning(f"Dropping {len(constant_cols)} constant sensor columns")
        df = df.drop(columns=constant_cols)

    logger.info(f"After cleaning: {len(df)} rows, {len(df.columns)} columns")
    return df


def encode_labels(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    target = config["data"]["target_column"]
    normal_label = config["data"]["normal_label"]
    df["label"] = (df[target] != normal_label).astype(int)  # 0=normal, 1=anomaly
    return df


def split_data(df: pd.DataFrame, config: dict):
    sensor_cols = get_sensor_columns(df, config)
    X = df[sensor_cols].values
    y = df["label"].values

    test_size = config["data"]["test_size"]
    val_size = config["data"]["val_size"]
    seed = config["data"]["random_seed"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=seed, stratify=y
    )
    relative_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - relative_val), random_state=seed, stratify=y_temp
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_scaler(X_train: np.ndarray, config: dict) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler_path = Path(config["model"]["scaler_path"])
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {scaler_path}")
    return scaler


def load_scaler(config: dict) -> StandardScaler:
    with open(config["model"]["scaler_path"], "rb") as f:
        return pickle.load(f)


def prepare_drift_simulation_data(df: pd.DataFrame, config: dict) -> dict:
    """
    Creates time-ordered data chunks for drift simulation.
    Returns reference window and a series of detection windows.
    """
    sensor_cols = get_sensor_columns(df, config)
    ref_size = config["drift_detection"]["reference_window_size"]
    win_size = config["drift_detection"]["detection_window_size"]

    normal_df = df[df["label"] == 0][sensor_cols].values
    anomaly_df = df[df["label"] == 1][sensor_cols].values

    reference = normal_df[:ref_size]

    # Create windows: first half normal, second half gradually drifting
    windows = []
    n_normal_windows = 10
    n_drift_windows = 10

    for i in range(n_normal_windows):
        start = ref_size + i * win_size
        end = start + win_size
        if end <= len(normal_df):
            windows.append({"data": normal_df[start:end], "has_drift": False, "window_idx": i})

    for i in range(n_drift_windows):
        start = i * win_size
        end = start + win_size
        if end <= len(anomaly_df):
            windows.append({"data": anomaly_df[start:end], "has_drift": True, "window_idx": n_normal_windows + i})

    out_dir = Path(config["data"]["drift_simulation_path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "reference.npy", reference)

    logger.info(f"Drift simulation: reference={len(reference)}, windows={len(windows)}")
    return {"reference": reference, "windows": windows}


def run_preprocessing(config: dict = None):
    if config is None:
        config = load_config()

    df = load_raw_data(config)
    df = clean_data(df, config)
    df = encode_labels(df, config)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, config)
    scaler = fit_scaler(X_train, config)

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    processed = {
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": get_sensor_columns(df, config),
    }

    out_path = Path(config["data"]["processed_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(processed, f)
    logger.info(f"Processed data saved to {out_path}")

    drift_data = prepare_drift_simulation_data(df, config)
    return processed, drift_data


if __name__ == "__main__":
    cfg = load_config()
    run_preprocessing(cfg)
