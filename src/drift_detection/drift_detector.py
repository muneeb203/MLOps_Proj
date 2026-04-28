"""
Empirical comparison of 6 drift detection methods:
  1. ADWIN       - Adaptive Windowing (river)
  2. DDM         - Drift Detection Method (river)
  3. KSWIN       - Kolmogorov-Smirnov Windowing (river)
  4. Page-Hinkley - Page-Hinkley test (river)
  5. KS-Test     - Kolmogorov-Smirnov two-sample test (scipy)
  6. PSI         - Population Stability Index (custom)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from scipy import stats


@dataclass
class DriftResult:
    method: str
    window_idx: int
    drift_detected: bool
    detection_latency: float       # seconds
    false_positive: bool
    false_negative: bool
    p_value: Optional[float] = None
    statistic: Optional[float] = None
    cpu_time_ms: float = 0.0


class ADWINDetector:
    def __init__(self, config: dict):
        from river.drift import ADWIN
        self.delta = config["drift_detection"]["methods"]["adwin"]["delta"]
        self._detector = ADWIN(delta=self.delta)

    def reset(self):
        from river.drift import ADWIN
        self._detector = ADWIN(delta=self.delta)

    def detect(self, window_data: np.ndarray) -> tuple[bool, dict]:
        start = time.perf_counter()
        # Use first principal component as the scalar stream
        stream = window_data[:, 0] if window_data.ndim > 1 else window_data
        drift_detected = False
        for val in stream:
            self._detector.update(float(val))
            if self._detector.drift_detected:
                drift_detected = True
                break
        elapsed_ms = (time.perf_counter() - start) * 1000
        return drift_detected, {"cpu_time_ms": elapsed_ms}


class DDMDetector:
    """
    Drift Detection Method (Gama et al., 2004) — implemented from scratch.
    Monitors error rate p and std s; signals drift when p+s exceeds
    p_min + drift_level * s_min.
    """
    def __init__(self, config: dict):
        cfg = config["drift_detection"]["methods"]["ddm"]
        self.min_num_instances = cfg["min_num_instances"]
        self.warning_level = cfg["warning_level"]
        self.drift_level = cfg["drift_level"]
        self.reset()

    def reset(self):
        self._n = 0
        self._p = 1.0      # error rate
        self._s = 0.0      # std dev
        self._p_min = float("inf")
        self._s_min = float("inf")

    def _update(self, error: int) -> bool:
        self._n += 1
        self._p += (error - self._p) / self._n
        self._s = (self._p * (1 - self._p) / self._n) ** 0.5

        if self._n >= self.min_num_instances:
            if self._p + self._s < self._p_min + self._s_min:
                self._p_min = self._p
                self._s_min = self._s

            if self._p + self._s > self._p_min + self.drift_level * self._s_min:
                return True  # drift detected
        return False

    def detect(self, window_data: np.ndarray) -> tuple[bool, dict]:
        start = time.perf_counter()
        self.reset()
        stream = window_data[:, 0] if window_data.ndim > 1 else window_data
        mn, mx = stream.min(), stream.max()
        normalized = (stream - mn) / (mx - mn) if mx > mn else np.zeros_like(stream)

        drift_detected = False
        for val in normalized:
            if self._update(int(val > 0.5)):
                drift_detected = True
                break
        elapsed_ms = (time.perf_counter() - start) * 1000
        return drift_detected, {"cpu_time_ms": elapsed_ms}


class KSWINDetector:
    def __init__(self, config: dict):
        from river.drift import KSWIN
        cfg = config["drift_detection"]["methods"]["kswin"]
        self.alpha = cfg["alpha"]
        self.window_size = cfg["window_size"]
        self.stat_size = cfg["stat_size"]
        self._make_detector()

    def _make_detector(self):
        from river.drift import KSWIN
        self._detector = KSWIN(
            alpha=self.alpha,
            window_size=self.window_size,
            stat_size=self.stat_size,
        )

    def reset(self):
        self._make_detector()

    def detect(self, window_data: np.ndarray) -> tuple[bool, dict]:
        start = time.perf_counter()
        stream = window_data[:, 0] if window_data.ndim > 1 else window_data
        self._make_detector()
        drift_detected = False
        for val in stream:
            self._detector.update(float(val))
            if self._detector.drift_detected:
                drift_detected = True
                break
        elapsed_ms = (time.perf_counter() - start) * 1000
        return drift_detected, {"cpu_time_ms": elapsed_ms}


class PageHinkleyDetector:
    def __init__(self, config: dict):
        cfg = config["drift_detection"]["methods"]["page_hinkley"]
        self.min_instances = cfg["min_instances"]
        self.delta = cfg["delta"]
        self.threshold = cfg["threshold"]
        self.alpha = cfg["alpha"]

    def _make_detector(self):
        from river.drift import PageHinkley
        return PageHinkley(
            min_instances=self.min_instances,
            delta=self.delta,
            threshold=self.threshold,
            alpha=self.alpha,
        )

    def reset(self):
        pass

    def detect(self, window_data: np.ndarray) -> tuple[bool, dict]:
        start = time.perf_counter()
        detector = self._make_detector()
        stream = window_data[:, 0] if window_data.ndim > 1 else window_data
        drift_detected = False
        for val in stream:
            detector.update(float(val))
            if detector.drift_detected:
                drift_detected = True
                break
        elapsed_ms = (time.perf_counter() - start) * 1000
        return drift_detected, {"cpu_time_ms": elapsed_ms}


class KSTestDetector:
    def __init__(self, config: dict):
        cfg = config["drift_detection"]["methods"]["ks_test"]
        self.alpha = cfg["alpha"]

    def reset(self):
        pass

    def detect(self, window_data: np.ndarray, reference: np.ndarray) -> tuple[bool, dict]:
        start = time.perf_counter()
        # Run KS test on each feature, flag drift if majority features drift
        n_features = window_data.shape[1] if window_data.ndim > 1 else 1
        if window_data.ndim == 1:
            window_data = window_data.reshape(-1, 1)
            reference = reference.reshape(-1, 1)

        drift_count = 0
        p_values = []
        statistics = []
        for i in range(n_features):
            stat, p_val = stats.ks_2samp(reference[:, i], window_data[:, i])
            p_values.append(p_val)
            statistics.append(stat)
            if p_val < self.alpha:
                drift_count += 1

        drift_detected = drift_count > (n_features * 0.3)  # >30% features drifted
        elapsed_ms = (time.perf_counter() - start) * 1000
        return drift_detected, {
            "cpu_time_ms": elapsed_ms,
            "p_value": float(np.median(p_values)),
            "statistic": float(np.median(statistics)),
            "n_drifted_features": drift_count,
        }


class PSIDetector:
    def __init__(self, config: dict):
        cfg = config["drift_detection"]["methods"]["psi"]
        self.buckets = cfg["buckets"]
        self.threshold = cfg["threshold"]

    def reset(self):
        pass

    @staticmethod
    def _psi_single(reference: np.ndarray, current: np.ndarray, buckets: int) -> float:
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        if max_val == min_val:
            return 0.0
        bins = np.linspace(min_val, max_val, buckets + 1)
        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)
        ref_pct = ref_counts / max(len(reference), 1)
        cur_pct = cur_counts / max(len(current), 1)
        # Avoid division by zero / log(0)
        ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
        cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def detect(self, window_data: np.ndarray, reference: np.ndarray) -> tuple[bool, dict]:
        start = time.perf_counter()
        if window_data.ndim == 1:
            window_data = window_data.reshape(-1, 1)
            reference = reference.reshape(-1, 1)

        n_features = window_data.shape[1]
        psi_values = [
            self._psi_single(reference[:, i], window_data[:, i], self.buckets)
            for i in range(n_features)
        ]
        mean_psi = float(np.mean(psi_values))
        drift_detected = mean_psi > self.threshold
        elapsed_ms = (time.perf_counter() - start) * 1000
        return drift_detected, {
            "cpu_time_ms": elapsed_ms,
            "statistic": mean_psi,
        }


class DriftDetectionBenchmark:
    """Run all 6 drift detection methods on the same windows and collect results."""

    def __init__(self, config: dict):
        self.config = config
        self.detectors = {
            "ADWIN": ADWINDetector(config),
            "DDM": DDMDetector(config),
            "KSWIN": KSWINDetector(config),
            "PageHinkley": PageHinkleyDetector(config),
            "KS_Test": KSTestDetector(config),
            "PSI": PSIDetector(config),
        }

    def run(self, reference: np.ndarray, windows: list) -> List[DriftResult]:
        all_results: List[DriftResult] = []

        for window_info in windows:
            window_data = window_info["data"]
            has_drift = window_info["has_drift"]
            window_idx = window_info["window_idx"]

            logger.info(f"Window {window_idx} | ground_truth_drift={has_drift}")

            for method_name, detector in self.detectors.items():
                detector.reset()
                t_start = time.perf_counter()

                if method_name in ("KS_Test", "PSI"):
                    drift_detected, extra = detector.detect(window_data, reference)
                else:
                    drift_detected, extra = detector.detect(window_data)

                latency = time.perf_counter() - t_start

                result = DriftResult(
                    method=method_name,
                    window_idx=window_idx,
                    drift_detected=drift_detected,
                    detection_latency=latency,
                    false_positive=(drift_detected and not has_drift),
                    false_negative=(not drift_detected and has_drift),
                    p_value=extra.get("p_value"),
                    statistic=extra.get("statistic"),
                    cpu_time_ms=extra.get("cpu_time_ms", 0.0),
                )
                all_results.append(result)
                logger.debug(
                    f"  {method_name}: detected={drift_detected} "
                    f"FP={result.false_positive} FN={result.false_negative} "
                    f"time={latency*1000:.2f}ms"
                )

        return all_results
