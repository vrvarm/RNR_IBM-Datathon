"""Per-card state and the three real-time feature layers.

The SAME code path builds features for training (replaying history in time order) and for the
live API (one transaction at a time), so there is no train/serve skew.

  Layer 1  location  : device vs store distance, store vs the card's usual area
  Layer 2  velocity  : km / hours since the previous transaction -> implied speed
  Layer 3  merchant  : category risk, amount vs category norm, card's history in that category
"""
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import PCA_COLS

EARTH_KM = 6371.0088
NAN = float("nan")

BASE = ["log_amount", "hour_of_day", "n_prior"]
LAYER1 = ["dev_store_km", "store_from_usual_km"]
LAYER2 = ["km_from_last", "hours_since_last", "speed_kmh", "txn_last_hour"]
LAYER3 = ["cat_risk", "amount_z_cat", "cat_affinity", "new_cat"]
CORE_FEATURES = BASE + LAYER1 + LAYER2 + LAYER3


def feature_columns(use_pca: bool = True):
    return CORE_FEATURES + (PCA_COLS if use_pca else [])


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_KM * math.asin(min(1.0, math.sqrt(a)))


@dataclass
class CardState:
    n: int = 0
    sum_lat: float = 0.0
    sum_lon: float = 0.0
    last_lat: float = 0.0
    last_lon: float = 0.0
    last_ts: float = 0.0
    cats: dict = field(default_factory=dict)
    recent: deque = field(default_factory=lambda: deque(maxlen=100))

    def update(self, ts, lat, lon, category):
        self.n += 1
        self.sum_lat += lat
        self.sum_lon += lon
        self.last_lat, self.last_lon, self.last_ts = lat, lon, ts
        self.cats[category] = self.cats.get(category, 0) + 1
        self.recent.append(ts)


@dataclass
class MerchantProfile:
    """Per-category stats learned on the training split only."""
    base_rate: float
    cats: dict  # name -> {"risk": smoothed fraud rate, "log_mu", "log_sd": log1p(amount) stats}

    @classmethod
    def fit(cls, df: pd.DataFrame, prior_strength: float = 200.0):
        base = float(df["Class"].mean())
        cats = {}
        for name, g in df.groupby("merchant_category"):
            la = np.log1p(g["Amount"].to_numpy())
            cats[name] = {
                "risk": float((g["Class"].sum() + base * prior_strength) / (len(g) + prior_strength)),
                "log_mu": float(la.mean()),
                "log_sd": float(la.std()) or 1.0,
            }
        return cls(base, cats)


def featurize(txn: dict, state: CardState, profile: MerchantProfile) -> dict:
    """txn keys: ts, amount, category, store_lat, store_lon, device_lat, device_lon."""
    prof = profile.cats[txn["category"]]
    ts, cat = txn["ts"], txn["category"]
    log_amt = math.log1p(txn["amount"])

    f = {
        "log_amount": log_amt,
        "hour_of_day": (ts / 3600.0) % 24,
        "n_prior": float(state.n),
        # layer 1
        "dev_store_km": haversine_km(txn["device_lat"], txn["device_lon"], txn["store_lat"], txn["store_lon"]),
        "store_from_usual_km": NAN,
        # layer 2
        "km_from_last": NAN, "hours_since_last": NAN, "speed_kmh": NAN, "txn_last_hour": 0.0,
        # layer 3
        "cat_risk": prof["risk"],
        "amount_z_cat": (log_amt - prof["log_mu"]) / prof["log_sd"],
        "cat_affinity": NAN,
        "new_cat": 0.0,
    }
    if state.n:
        f["store_from_usual_km"] = haversine_km(
            state.sum_lat / state.n, state.sum_lon / state.n, txn["store_lat"], txn["store_lon"])
        km = haversine_km(state.last_lat, state.last_lon, txn["store_lat"], txn["store_lon"])
        hours = max(ts - state.last_ts, 0.0) / 3600.0
        f["km_from_last"], f["hours_since_last"] = km, hours
        f["speed_kmh"] = min(km / max(hours, 1 / 60), 5000.0)  # floor of 1 minute avoids divide blow-ups
        f["cat_affinity"] = state.cats.get(cat, 0) / state.n
        f["new_cat"] = float(cat not in state.cats and state.n >= 5)
        count = 0
        for t in reversed(state.recent):
            if ts - t > 3600:
                break
            count += 1
        f["txn_last_hour"] = float(count)
    return f


def featurize_frame(df: pd.DataFrame, profile: MerchantProfile):
    """Replay a time-sorted frame through per-card state. Returns (features frame, final states)."""
    states, rows = {}, []
    cols = ("card_id", "Time", "Amount", "merchant_category",
            "store_lat", "store_lon", "device_lat", "device_lon")
    for cid, ts, amt, cat, slat, slon, dlat, dlon in zip(*(df[c].to_numpy() for c in cols)):
        st = states.get(cid)
        if st is None:
            st = states[cid] = CardState()
        rows.append(featurize(
            {"ts": float(ts), "amount": float(amt), "category": cat, "store_lat": slat,
             "store_lon": slon, "device_lat": dlat, "device_lon": dlon}, st, profile))
        st.update(float(ts), slat, slon, cat)
    X = pd.DataFrame(rows, columns=CORE_FEATURES)
    pca = df[PCA_COLS].reset_index(drop=True)
    return pd.concat([X, pca], axis=1), states
