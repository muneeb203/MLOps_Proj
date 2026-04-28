import numpy as np
import pandas as pd
import pytest

from src.ingestion.data_loader import load_config, get_sensor_columns
from src.preprocessing.preprocessor import clean_data, encode_labels, split_data


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def sample_df(config):
    np.random.seed(42)
    n = 500
    sensor_cols = [f"sensor_{i:02d}" for i in range(10)]
    df = pd.DataFrame(np.random.normal(50, 5, (n, 10)), columns=sensor_cols)
    df["machine_status"] = ["NORMAL"] * 400 + ["BROKEN"] * 70 + ["RECOVERING"] * 30
    df.insert(0, "timestamp", pd.date_range("2020-01-01", periods=n, freq="1min"))
    return df


def test_clean_data_removes_constant_columns(config, sample_df):
    sample_df["constant_sensor"] = 5.0  # constant column
    cleaned = clean_data(sample_df, config)
    assert "constant_sensor" not in cleaned.columns


def test_clean_data_fills_nans(config, sample_df):
    sample_df.iloc[0, 1] = np.nan
    cleaned = clean_data(sample_df, config)
    assert cleaned.isnull().sum().sum() == 0


def test_encode_labels_binary(config, sample_df):
    cleaned = clean_data(sample_df, config)
    labeled = encode_labels(cleaned, config)
    assert set(labeled["label"].unique()).issubset({0, 1})
    assert labeled[labeled["machine_status"] == "NORMAL"]["label"].eq(0).all()
    assert labeled[labeled["machine_status"] == "BROKEN"]["label"].eq(1).all()


def test_split_sizes(config, sample_df):
    cleaned = clean_data(sample_df, config)
    labeled = encode_labels(cleaned, config)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(labeled, config)
    total = len(X_train) + len(X_val) + len(X_test)
    assert total == len(labeled)
    assert len(X_train) > len(X_test)
