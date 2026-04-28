"""
Sends a mix of normal and anomalous requests to the API
to populate Prometheus metrics and Grafana dashboards.
"""
import random
import time
import requests

API = "http://localhost:8080"
N_REQUESTS = 200


def normal_reading():
    return [random.gauss(50, 5) for _ in range(52)]


def anomalous_reading():
    return [random.gauss(120, 30) for _ in range(52)]


def main():
    # Check API is up
    try:
        r = requests.get(f"{API}/health", timeout=5)
        print(f"API health: {r.json()}")
    except Exception as e:
        print(f"API not reachable: {e}")
        return

    print(f"\nSending {N_REQUESTS} requests (mix of normal + anomalous)...\n")

    for i in range(N_REQUESTS):
        # 80% normal, 20% anomalous
        if random.random() < 0.2:
            payload = {"sensor_values": anomalous_reading()}
            kind = "ANOMALOUS"
        else:
            payload = {"sensor_values": normal_reading()}
            kind = "normal  "

        try:
            r = requests.post(f"{API}/predict", json=payload, timeout=5)
            data = r.json()
            print(
                f"[{i+1:03d}] {kind} | anomaly={data['is_anomaly']} "
                f"score={data['anomaly_score']:.3f} "
                f"latency={data['latency_ms']:.1f}ms "
                f"drift={data['drift_detected']}"
            )
        except Exception as e:
            print(f"[{i+1:03d}] Error: {e}")

        time.sleep(0.1)

    print("\nDone! Refresh Grafana at http://localhost:3000")


if __name__ == "__main__":
    main()
