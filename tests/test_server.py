import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from iforest_server.main import app
from iforest_server.models.schemas import Signal, Discharge

client = TestClient(app)

def test_full_flow():
    # start training
    resp = client.post("/train", json={"totalDischarges": 3, "timeoutSeconds": 10})
    assert resp.status_code == 200
    assert resp.json()["expectedDischarges"] == 3

    # send first discharge
    d1 = {
        "id": "d1",
        "signals": {"filename": "f", "values": [0.1, 0.2]},
        "times": [0.0, 1.0],
        "length": 2,
    }
    resp = client.post("/train/1", json=d1)
    assert resp.status_code == 200

    # send second discharge marked as anomaly
    d2 = {
        "id": "d2",
        "signals": {"filename": "f", "values": [0.2, 0.3]},
        "times": [0.0, 1.0],
        "length": 2,
        "anomalyTime": 0.5,
    }
    resp = client.post("/train/2", json=d2)
    assert resp.status_code == 200

    # send third discharge triggers training using only normal data
    d3 = {
        "id": "d3",
        "signals": {"filename": "f", "values": [0.15, 0.25]},
        "times": [0.0, 1.0],
        "length": 2,
    }
    resp = client.post("/train/3", json=d3)
    assert resp.status_code == 200

    # prediction after training
    resp = client.post("/predict", json=d1)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in ["Normal", "Anomaly"]
    assert 0 <= body["confidence"] <= 1
