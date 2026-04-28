"""
Runs the full drift detection benchmark and logs results to MLflow.
This is the core research experiment of the project.
"""

import json
from pathlib import Path
from typing import List

import mlflow
import numpy as np
import pandas as pd
from loguru import logger

from src.drift_detection.drift_detector import DriftDetectionBenchmark, DriftResult
from src.ingestion.data_loader import load_config
from src.preprocessing.preprocessor import run_preprocessing


def compute_method_summary(results: List[DriftResult]) -> pd.DataFrame:
    records = []
    for r in results:
        records.append({
            "method": r.method,
            "window_idx": r.window_idx,
            "drift_detected": r.drift_detected,
            "false_positive": r.false_positive,
            "false_negative": r.false_negative,
            "latency_ms": r.detection_latency * 1000,
            "cpu_time_ms": r.cpu_time_ms,
        })
    df = pd.DataFrame(records)

    summary = df.groupby("method").agg(
        total_windows=("window_idx", "count"),
        detections=("drift_detected", "sum"),
        false_positives=("false_positive", "sum"),
        false_negatives=("false_negative", "sum"),
        avg_latency_ms=("latency_ms", "mean"),
        avg_cpu_ms=("cpu_time_ms", "mean"),
    ).reset_index()

    summary["fpr"] = summary["false_positives"] / summary["total_windows"]
    summary["fnr"] = summary["false_negatives"] / summary["total_windows"]
    summary["precision"] = (
        (summary["detections"] - summary["false_positives"]) /
        summary["detections"].replace(0, 1)
    )
    return summary


def run_benchmark(config: dict = None) -> pd.DataFrame:
    if config is None:
        config = load_config()

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    logger.info("Running preprocessing for drift simulation data...")
    _, drift_data = run_preprocessing(config)

    reference = drift_data["reference"]
    windows = drift_data["windows"]

    benchmark = DriftDetectionBenchmark(config)

    with mlflow.start_run(run_name="drift_detection_benchmark") as run:
        mlflow.log_params({
            "reference_window_size": config["drift_detection"]["reference_window_size"],
            "detection_window_size": config["drift_detection"]["detection_window_size"],
            "n_windows": len(windows),
            "n_features": reference.shape[1],
            "methods": "ADWIN,DDM,KSWIN,PageHinkley,KS_Test,PSI",
        })

        logger.info("Starting drift detection benchmark across 6 methods...")
        results = benchmark.run(reference, windows)

        summary = compute_method_summary(results)
        logger.info(f"\nBenchmark Summary:\n{summary.to_string(index=False)}")

        # Log per-method metrics to MLflow
        for _, row in summary.iterrows():
            method = row["method"]
            mlflow.log_metrics({
                f"{method}_fpr": row["fpr"],
                f"{method}_fnr": row["fnr"],
                f"{method}_avg_latency_ms": row["avg_latency_ms"],
                f"{method}_avg_cpu_ms": row["avg_cpu_ms"],
                f"{method}_precision": row["precision"],
            })

        # Save and log full results
        results_dir = Path("mlflow_tracking/benchmark_results")
        results_dir.mkdir(parents=True, exist_ok=True)

        raw_df = pd.DataFrame([vars(r) for r in results])
        raw_path = results_dir / "raw_results.csv"
        summary_path = results_dir / "summary.csv"
        raw_df.to_csv(raw_path, index=False)
        summary.to_csv(summary_path, index=False)

        mlflow.log_artifact(str(raw_path))
        mlflow.log_artifact(str(summary_path))

        logger.info(f"Benchmark run ID: {run.info.run_id}")

    return summary


if __name__ == "__main__":
    cfg = load_config()
    summary = run_benchmark(cfg)
    print(summary)
