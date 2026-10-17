import math

from fraud.features import CardState, MerchantProfile, featurize, haversine_km

PROFILE = MerchantProfile(0.005, {"grocery": {"risk": 0.005, "log_mu": 3.4, "log_sd": 1.2}})
DALLAS = (32.78, -96.80)
NYC = (40.71, -74.01)


def txn(ts, loc, device=None):
    device = device or loc
    return {"ts": ts, "amount": 20.0, "category": "grocery", "store_lat": loc[0], "store_lon": loc[1],
            "device_lat": device[0], "device_lon": device[1]}


def test_haversine_nyc_la():
    assert 3900 < haversine_km(40.7128, -74.0060, 34.0522, -118.2437) < 3980


def test_first_transaction_has_no_history():
    f = featurize(txn(0, DALLAS), CardState(), PROFILE)
    assert math.isnan(f["speed_kmh"]) and math.isnan(f["km_from_last"])
    assert f["txn_last_hour"] == 0.0 and f["dev_store_km"] < 0.01


def test_impossible_travel_speed():
    st = CardState()
    st.update(0, *DALLAS, "grocery")
    f = featurize(txn(1800, NYC), st, PROFILE)  # Dallas -> New York in 30 minutes
    assert f["km_from_last"] > 2000
    assert f["speed_kmh"] > 4000


def test_device_far_from_store():
    f = featurize(txn(0, NYC, device=DALLAS), CardState(), PROFILE)
    assert f["dev_store_km"] > 2000


def test_velocity_counts_recent_transactions():
    st = CardState()
    for t in (0, 600, 1200):
        st.update(t, *DALLAS, "grocery")
    assert featurize(txn(1500, DALLAS), st, PROFILE)["txn_last_hour"] == 3.0
    assert featurize(txn(9000, DALLAS), st, PROFILE)["txn_last_hour"] == 0.0
