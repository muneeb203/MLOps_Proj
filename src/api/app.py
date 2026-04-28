import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from loguru import logger
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field

from src.drift_detection.drift_detector import (
    ADWINDetector,
    KSTestDetector,
    PSIDetector,
)
from src.monitoring.metrics import (
    ANOMALY_DETECTED,
    ANOMALY_SCORE,
    DRIFT_DETECTED,
    DRIFT_DETECTION_LATENCY,
    MODEL_VERSION,
    PREDICTION_THROUGHPUT,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Global state ─────────────────────────────────────────────────────────────
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    _state["config"] = config

    model_path = Path(config["model"]["model_path"])
    scaler_path = Path(config["model"]["scaler_path"])

    if model_path.exists():
        with open(model_path, "rb") as f:
            _state["model"] = pickle.load(f)
        logger.info("Model loaded.")
    else:
        logger.warning("Model file not found. Run training first.")
        _state["model"] = None

    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            _state["scaler"] = pickle.load(f)
    else:
        _state["scaler"] = None

    _state["reference_window"] = []
    _state["adwin"] = ADWINDetector(config)
    _state["ks_test"] = KSTestDetector(config)
    _state["psi"] = PSIDetector(config)
    _state["model_version"] = 1
    MODEL_VERSION.set(1)

    yield
    _state.clear()


app = FastAPI(
    title="Predictive Maintenance MLOps API",
    description="Sensor anomaly detection with real-time drift monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ── Schemas ───────────────────────────────────────────────────────────────────
class SensorReading(BaseModel):
    sensor_values: List[float] = Field(..., description="List of sensor readings")
    timestamp: Optional[str] = None


class PredictionResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    latency_ms: float
    drift_detected: bool
    drift_methods_triggered: List[str]


class BatchPredictionRequest(BaseModel):
    readings: List[SensorReading]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: int
    reference_window_size: int


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_state.get("model") is not None,
        model_version=_state.get("model_version", 0),
        reference_window_size=len(_state.get("reference_window", [])),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    t_start = time.perf_counter()
    endpoint = "predict"

    model = _state.get("model")
    scaler = _state.get("scaler")
    config = _state["config"]

    if model is None:
        REQUEST_COUNT.labels(endpoint=endpoint, status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        x = np.array(reading.sensor_values).reshape(1, -1)
        if scaler is not None:
            x = scaler.transform(x)

        raw_pred = model.predict(x)[0]
        score = float(-model.decision_function(x)[0])
        is_anomaly = raw_pred == -1

        ANOMALY_SCORE.set(score)
        if is_anomaly:
            ANOMALY_DETECTED.inc()

        # Update reference window for drift detection
        ref_win = _state["reference_window"]
        ref_win.append(x[0])
        ref_size = config["drift_detection"]["reference_window_size"]
        if len(ref_win) > ref_size:
            _state["reference_window"] = ref_win[-ref_size:]

        # Real-time drift detection via ADWIN
        drift_methods = []
        adwin: ADWINDetector = _state["adwin"]
        t_drift = time.perf_counter()
        adwin._detector.update(score)
        if adwin._detector.drift_detected:
            drift_methods.append("ADWIN")
            DRIFT_DETECTED.labels(method="ADWIN").inc()
        DRIFT_DETECTION_LATENCY.labels(method="ADWIN").observe(
            (time.perf_counter() - t_drift) * 1000
        )

        latency_ms = (time.perf_counter() - t_start) * 1000
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency_ms / 1000)
        REQUEST_COUNT.labels(endpoint=endpoint, status="success").inc()

        return PredictionResponse(
            is_anomaly=is_anomaly,
            anomaly_score=round(score, 4),
            latency_ms=round(latency_ms, 2),
            drift_detected=len(drift_methods) > 0,
            drift_methods_triggered=drift_methods,
        )

    except Exception as e:
        REQUEST_COUNT.labels(endpoint=endpoint, status="error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    t_start = time.perf_counter()
    results = []
    for reading in request.readings:
        results.append(predict(reading))
    PREDICTION_THROUGHPUT.observe(len(request.readings))
    return {"predictions": results, "count": len(results),
            "total_latency_ms": round((time.perf_counter() - t_start) * 1000, 2)}


@app.post("/drift/check")
def check_drift_batch(request: BatchPredictionRequest):
    """Run all 3 reference-based drift detectors on a batch."""
    config = _state["config"]
    ref_win = _state.get("reference_window", [])

    if len(ref_win) < 100:
        raise HTTPException(status_code=400, detail="Insufficient reference data. Send more /predict requests first.")

    scaler = _state.get("scaler")
    window_data = np.array([r.sensor_values for r in request.readings])
    if scaler is not None:
        window_data = scaler.transform(window_data)
    reference = np.array(ref_win)

    results = {}
    ks: KSTestDetector = _state["ks_test"]
    psi: PSIDetector = _state["psi"]

    t = time.perf_counter()
    ks_detected, ks_extra = ks.detect(window_data, reference)
    DRIFT_DETECTION_LATENCY.labels(method="KS_Test").observe((time.perf_counter() - t) * 1000)
    if ks_detected:
        DRIFT_DETECTED.labels(method="KS_Test").inc()
    results["KS_Test"] = {"drift_detected": ks_detected, **ks_extra}

    t = time.perf_counter()
    psi_detected, psi_extra = psi.detect(window_data, reference)
    DRIFT_DETECTION_LATENCY.labels(method="PSI").observe((time.perf_counter() - t) * 1000)
    if psi_detected:
        DRIFT_DETECTED.labels(method="PSI").inc()
    results["PSI"] = {"drift_detected": psi_detected, **psi_extra}

    return {"drift_results": results, "reference_size": len(ref_win), "window_size": len(window_data)}


@app.get("/")
def root():
    return {"message": "Predictive Maintenance MLOps API", "docs": "/docs", "metrics": "/metrics"}
