from prometheus_client import Counter, Gauge, Histogram, Summary

REQUEST_COUNT = Counter(
    "predictive_maintenance_requests_total",
    "Total prediction requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "predictive_maintenance_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

ANOMALY_SCORE = Gauge(
    "predictive_maintenance_anomaly_score",
    "Latest anomaly score from the model",
)

ANOMALY_DETECTED = Counter(
    "predictive_maintenance_anomalies_detected_total",
    "Total anomalies detected",
)

DRIFT_DETECTED = Counter(
    "predictive_maintenance_drift_detected_total",
    "Total drift events detected",
    ["method"],
)

DRIFT_DETECTION_LATENCY = Histogram(
    "predictive_maintenance_drift_detection_latency_ms",
    "Drift detection latency in milliseconds",
    ["method"],
    buckets=[0.1, 0.5, 1, 5, 10, 50, 100, 500],
)

MODEL_VERSION = Gauge(
    "predictive_maintenance_model_version",
    "Currently loaded model version",
)

PREDICTION_THROUGHPUT = Summary(
    "predictive_maintenance_prediction_throughput",
    "Predictions processed per batch",
)
