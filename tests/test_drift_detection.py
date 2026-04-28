import numpy as np
import pytest

from src.drift_detection.drift_detector import (
    ADWINDetector,
    DDMDetector,
    KSTestDetector,
    KSWINDetector,
    PSIDetector,
    PageHinkleyDetector,
    DriftDetectionBenchmark,
)
from src.ingestion.data_loader import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def normal_window():
    np.random.seed(0)
    return np.random.normal(50, 5, (200, 5))


@pytest.fixture
def drift_window():
    np.random.seed(1)
    return np.random.normal(90, 15, (200, 5))  # significantly shifted


def test_ks_test_detects_drift(config, normal_window, drift_window):
    detector = KSTestDetector(config)
    detected, _ = detector.detect(drift_window, normal_window)
    assert detected is True


def test_ks_test_no_false_positive(config, normal_window):
    detector = KSTestDetector(config)
    # Same distribution should not trigger drift
    reference = normal_window[:100]
    window = normal_window[100:]
    detected, _ = detector.detect(window, reference)
    assert detected is False


def test_psi_detects_drift(config, normal_window, drift_window):
    detector = PSIDetector(config)
    detected, extra = detector.detect(drift_window, normal_window)
    assert detected is True
    assert extra["statistic"] > config["drift_detection"]["methods"]["psi"]["threshold"]


def test_psi_no_false_positive(config, normal_window):
    detector = PSIDetector(config)
    reference = normal_window[:100]
    window = normal_window[100:]
    detected, _ = detector.detect(window, reference)
    assert detected is False


def test_adwin_returns_bool(config, normal_window):
    detector = ADWINDetector(config)
    detected, extra = detector.detect(normal_window)
    assert isinstance(detected, bool)
    assert "cpu_time_ms" in extra


def test_benchmark_returns_results_for_all_methods(config):
    np.random.seed(42)
    reference = np.random.normal(50, 5, (1000, 5))
    windows = [
        {"data": np.random.normal(50, 5, (200, 5)), "has_drift": False, "window_idx": 0},
        {"data": np.random.normal(90, 15, (200, 5)), "has_drift": True, "window_idx": 1},
    ]
    benchmark = DriftDetectionBenchmark(config)
    results = benchmark.run(reference, windows)

    methods_seen = {r.method for r in results}
    expected = {"ADWIN", "DDM", "KSWIN", "PageHinkley", "KS_Test", "PSI"}
    assert expected == methods_seen

    for r in results:
        assert isinstance(r.drift_detected, bool)
        assert r.cpu_time_ms >= 0
