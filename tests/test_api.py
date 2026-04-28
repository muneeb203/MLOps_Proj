import pickle
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture
def client(tmp_path):
    # Create a mock model and scaler
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    model = IsolationForest(contamination=0.05, random_state=42)
    X_train = np.random.normal(50, 5, (500, 10))
    model.fit(X_train)

    scaler = StandardScaler()
    scaler.fit(X_train)

    model_path = tmp_path / "anomaly_detector.pkl"
    scaler_path = tmp_path / "scaler.pkl"
    config_path = tmp_path / "config.yaml"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    import yaml
    from src.ingestion.data_loader import load_config
    cfg = load_config()
    cfg["model"]["model_path"] = str(model_path)
    cfg["model"]["scaler_path"] = str(scaler_path)

    (tmp_path / "configs").mkdir(exist_ok=True)
    with open(tmp_path / "configs" / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    with patch("src.api.app.load_config", return_value=cfg):
        from src.api.app import app
        with TestClient(app) as c:
            yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_predict_normal(client):
    sensor_values = [50.0] * 10
    response = client.post("/predict", json={"sensor_values": sensor_values})
    assert response.status_code == 200
    data = response.json()
    assert "is_anomaly" in data
    assert "anomaly_score" in data
    assert "latency_ms" in data
    assert isinstance(data["drift_detected"], bool)


def test_predict_anomaly_high_values(client):
    # Extremely high values should be anomalous
    sensor_values = [500.0] * 10
    response = client.post("/predict", json={"sensor_values": sensor_values})
    assert response.status_code == 200
    data = response.json()
    assert data["is_anomaly"] is True


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Predictive Maintenance" in response.json()["message"]
