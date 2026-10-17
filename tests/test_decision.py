from fraud.decision import APPROVE, DECLINE, FLAG, decide

TH = {"flag": 0.2, "decline": 0.8}
BASE = 0.005


def feats(**over):
    f = {"dev_store_km": 0.05, "km_from_last": 3.0, "hours_since_last": 2.0, "speed_kmh": 1.5,
         "cat_risk": 0.004, "amount_z_cat": 0.1, "new_cat": 0.0}
    f.update(over)
    return f


def test_benign_is_approved():
    out = decide(0.01, feats(), TH, BASE)
    assert out["decision"] == APPROVE and out["reasons"] == []


def test_model_thresholds():
    assert decide(0.3, feats(), TH, BASE)["decision"] == FLAG
    assert decide(0.9, feats(), TH, BASE)["decision"] == DECLINE


def test_single_layer_alert_escalates_to_2fa():
    out = decide(0.01, feats(km_from_last=2200, hours_since_last=0.5, speed_kmh=4400), TH, BASE)
    assert out["decision"] == FLAG
    assert out["layers"]["velocity"]["status"] == "alert"


def test_two_layer_alerts_decline():
    out = decide(0.01, feats(dev_store_km=2000, km_from_last=2200, hours_since_last=0.5, speed_kmh=4400), TH, BASE)
    assert out["decision"] == DECLINE
    assert any("Multiple layers" in r for r in out["reasons"])


def test_gps_jitter_does_not_trigger_speed_alert():
    out = decide(0.01, feats(km_from_last=2.0, hours_since_last=0.001, speed_kmh=1200), TH, BASE)
    assert out["layers"]["velocity"]["status"] == "ok"
