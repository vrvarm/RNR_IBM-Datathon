import pytest
from fastapi.testclient import TestClient

from fraud.model import train
from fraud.service import app

DALLAS = (32.78, -96.80)
NYC = (40.71, -74.01)


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    model_dir = tmp_path_factory.mktemp("model")
    metrics = train(out_dir=model_dir, n_synth=20_000, fraud_rate=0.02, max_iter=60,
                    force_synthetic=True, export_replay=False)
    assert metrics["avg_precision"] > 0.3  # far above the ~0.02 random baseline
    monkeypatch_module.setenv("FRAUD_MODEL_DIR", str(model_dir))
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


def body(card, ts, store, device=None, category="grocery", amount=25.0):
    device = device or store
    return {"card_id": card, "amount": amount, "merchant_category": category, "timestamp": ts,
            "store_lat": store[0], "store_lon": store[1], "device_lat": device[0], "device_lon": device[1]}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_normal_purchase_response_shape(client):
    client.post("/reset")
    r = client.post("/score", json=body("A1", 1000, DALLAS))
    assert r.status_code == 200
    out = r.json()
    assert out["decision"] in {"APPROVE", "FLAG_2FA", "DECLINE"}
    assert set(out["layers"]) == {"location", "velocity", "merchant"}
    assert 0 <= out["fraud_probability"] <= 1


def test_impossible_travel_is_not_approved(client):
    client.post("/reset")
    client.post("/score", json=body("B1", 1000, DALLAS))
    r = client.post("/score", json=body("B1", 1600, NYC, device=DALLAS, category="electronics", amount=900))
    out = r.json()
    assert out["decision"] != "APPROVE"
    assert out["layers"]["velocity"]["status"] == "alert"
    assert out["layers"]["location"]["status"] == "alert"


def test_dry_run_does_not_update_history(client):
    client.post("/reset")
    client.post("/score?commit=false", json=body("C1", 1000, DALLAS))
    assert client.get("/stats").json()["cards_tracked"] == 0


def test_unknown_category_rejected(client):
    r = client.post("/score", json=body("D1", 1000, DALLAS, category="casino"))
    assert r.status_code == 422
