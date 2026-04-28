"""
Full end-to-end pipeline runner.
Usage:
  python scripts/run_pipeline.py --stage all
  python scripts/run_pipeline.py --stage data
  python scripts/run_pipeline.py --stage train
  python scripts/run_pipeline.py --stage benchmark
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.ingestion.data_loader import load_config, load_raw_data
from src.preprocessing.preprocessor import run_preprocessing
from src.training.trainer import run_training
from src.drift_detection.benchmark_runner import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="MLOps Pipeline Runner")
    parser.add_argument(
        "--stage",
        choices=["data", "train", "benchmark", "all"],
        default="all",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.stage in ("data", "all"):
        logger.info("=== Stage 1: Data Ingestion & Preprocessing ===")
        run_preprocessing(config)

    if args.stage in ("train", "all"):
        logger.info("=== Stage 2: Model Training ===")
        run_id = run_training(config)
        logger.info(f"Training complete. MLflow run: {run_id}")

    if args.stage in ("benchmark", "all"):
        logger.info("=== Stage 3: Drift Detection Benchmark ===")
        summary = run_benchmark(config)
        logger.info(f"\n{summary.to_string(index=False)}")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
