import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("models")

    model = IsolationForest(contamination=0.05, random_state=42)
    scaler = StandardScaler()
    X_train = np.random.normal(50, 5, (500, 10))
    model.fit(X_train)
    scaler.fit(X_train)

    import pickle, yaml
    from src.ingestion.data_loader import load_config

    cfg = load_config()
    cfg["model"]["model_path"] = str(tmp_path / "anomaly_detector.pkl")
    cfg["model"]["scaler_path"] = str(tmp_path / "scaler.pkl")

    with open(cfg["model"]["model_path"], "wb") as f:
        pickle.dump(model, f)
    with open(cfg["model"]["scaler_path"], "wb") as f:
        pickle.dump(scaler, f)

    with patch("src.api.app.load_config", return_value=cfg):
        from src.api.app import app
        with TestClient(app) as c:
            yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_expected_fields(client):
    response = client.post("/predict", json={"sensor_values": [50.0] * 10})
    assert response.status_code == 200
    data = response.json()
    assert "is_anomaly" in data
    assert "anomaly_score" in data
    assert "latency_ms" in data
    assert isinstance(data["drift_detected"], bool)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
