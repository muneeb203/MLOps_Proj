import pickle
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.ingestion.data_loader import load_config
from src.preprocessing.preprocessor import run_preprocessing


def setup_mlflow(config: dict) -> None:
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])


def train_isolation_forest(X_train: np.ndarray, config: dict) -> IsolationForest:
    model_cfg = config["model"]
    model = IsolationForest(
        contamination=model_cfg["contamination"],
        n_estimators=model_cfg["n_estimators"],
        random_state=model_cfg["random_state"],
        n_jobs=-1,
    )
    logger.info("Training Isolation Forest...")
    start = time.time()
    model.fit(X_train)
    elapsed = time.time() - start
    logger.info(f"Training complete in {elapsed:.2f}s")
    return model, elapsed


def evaluate_model(model: IsolationForest, X: np.ndarray, y_true: np.ndarray) -> dict:
    # IsolationForest: predict returns -1 (anomaly) or 1 (normal)
    raw_preds = model.predict(X)
    y_pred = (raw_preds == -1).astype(int)  # 1=anomaly, 0=normal

    scores = model.decision_function(X)
    # decision_function: lower = more anomalous; invert for roc_auc
    y_score = -scores

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0,
    }
    return metrics, y_pred


def save_model(model: IsolationForest, config: dict) -> str:
    model_path = Path(config["model"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")
    return str(model_path)


def load_model(config: dict) -> IsolationForest:
    with open(config["model"]["model_path"], "rb") as f:
        return pickle.load(f)


def run_training(config: dict = None) -> str:
    if config is None:
        config = load_config()

    setup_mlflow(config)

    processed, _ = run_preprocessing(config)
    X_train = processed["X_train"]
    X_val = processed["X_val"]
    X_test = processed["X_test"]
    y_val = processed["y_val"]
    y_test = processed["y_test"]

    with mlflow.start_run(run_name="isolation_forest_baseline") as run:
        # Log parameters
        mlflow.log_params({
            "model_type": config["model"]["type"],
            "contamination": config["model"]["contamination"],
            "n_estimators": config["model"]["n_estimators"],
            "random_state": config["model"]["random_state"],
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "n_features": X_train.shape[1],
        })

        model, train_time = train_isolation_forest(X_train, config)

        val_metrics, _ = evaluate_model(model, X_val, y_val)
        test_metrics, _ = evaluate_model(model, X_test, y_test)

        # Log metrics
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.log_metric("train_time_seconds", train_time)

        logger.info(f"Val metrics: {val_metrics}")
        logger.info(f"Test metrics: {test_metrics}")

        model_path = save_model(model, config)

        # Log model artifact to MLflow
        mlflow.sklearn.log_model(model, "isolation_forest_model")
        mlflow.log_artifact(config["model"]["scaler_path"])
        mlflow.log_artifact("configs/config.yaml")

        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")

    return run_id


if __name__ == "__main__":
    cfg = load_config()
    run_id = run_training(cfg)
    print(f"Training complete. Run ID: {run_id}")
