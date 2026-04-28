import os
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_dataset(config: dict) -> None:
    """Download Pump Sensor dataset from Kaggle."""
    raw_path = Path(config["data"]["raw_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        logger.info(f"Dataset already exists at {raw_path}")
        return

    try:
        import kaggle
        logger.info("Downloading pump sensor dataset from Kaggle...")
        kaggle.api.dataset_download_files(
            "nphantawee/pump-sensor-data",
            path=str(raw_path.parent),
            unzip=True
        )
        # Rename to expected filename if needed
        downloaded = list(raw_path.parent.glob("*.csv"))
        if downloaded and downloaded[0] != raw_path:
            downloaded[0].rename(raw_path)
        logger.info("Dataset downloaded successfully.")
    except Exception as e:
        logger.warning(f"Kaggle download failed: {e}. Generating synthetic data instead.")
        _generate_synthetic_data(config)


def _generate_synthetic_data(config: dict) -> None:
    """Generate synthetic pump sensor data for development/testing."""
    logger.info("Generating synthetic pump sensor data...")
    np.random.seed(config["data"]["random_seed"])

    n_normal = 180000
    n_broken = 20000
    n_recovering = 20000
    n_sensors = 52

    # Normal operating conditions
    normal = pd.DataFrame(
        np.random.normal(loc=50, scale=5, size=(n_normal, n_sensors)),
        columns=[f"sensor_{i:02d}" for i in range(n_sensors)]
    )
    normal["machine_status"] = "NORMAL"

    # Broken: sensors drift significantly (simulate concept drift)
    broken = pd.DataFrame(
        np.random.normal(loc=80, scale=15, size=(n_broken, n_sensors)),
        columns=[f"sensor_{i:02d}" for i in range(n_sensors)]
    )
    broken["machine_status"] = "BROKEN"

    # Recovering: between normal and broken
    recovering = pd.DataFrame(
        np.random.normal(loc=60, scale=8, size=(n_recovering, n_sensors)),
        columns=[f"sensor_{i:02d}" for i in range(n_sensors)]
    )
    recovering["machine_status"] = "RECOVERING"

    df = pd.concat([normal, broken, recovering], ignore_index=True)
    df.insert(0, "timestamp", pd.date_range("2018-04-01", periods=len(df), freq="1min"))

    raw_path = Path(config["data"]["raw_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=True)
    logger.info(f"Synthetic data saved to {raw_path} ({len(df)} rows)")


def load_raw_data(config: dict) -> pd.DataFrame:
    raw_path = Path(config["data"]["raw_path"])
    if not raw_path.exists():
        download_dataset(config)
    logger.info(f"Loading data from {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def get_sensor_columns(df: pd.DataFrame, config: dict) -> list:
    exclude = set(config["data"]["drop_columns"] + [config["data"]["target_column"], "label", "timestamp"])
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    cfg = load_config()
    df = load_raw_data(cfg)
    print(df.head())
    print(df["machine_status"].value_counts())
