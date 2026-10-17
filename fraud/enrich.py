"""Add the fields the Kaggle data lacks: card id, merchant category, store and device coordinates.

The original dataset is PCA-anonymised (V1..V28) and has no geography or merchant info, so these
columns are SIMULATED. Legit cards stay near home and shop their usual categories; a share of
fraud is used in another city while the cardholder's phone stays home. Velocity is NOT injected:
it emerges from the per-card sequence of store locations.
"""
import numpy as np
import pandas as pd

from .config import CATEGORIES, CITIES, FRAUD_CAT_WEIGHTS, LEGIT_CAT_WEIGHTS, N_CARDS, SEED


def enrich(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    n, k = len(df), len(CATEGORIES)
    fraud = df["Class"].to_numpy() == 1

    city_lat = np.array([c[1] for c in CITIES])
    city_lon = np.array([c[2] for c in CITIES])

    card = rng.integers(0, N_CARDS, n)
    home_city = rng.integers(0, len(CITIES), N_CARDS)
    home_lat = city_lat[home_city] + rng.normal(0, 0.08, N_CARDS)
    home_lon = city_lon[home_city] + rng.normal(0, 0.08, N_CARDS)
    h_lat, h_lon = home_lat[card], home_lon[card]

    # merchant category: each card favours a few categories; fraud skews to high-risk ones
    legit_w = np.array(LEGIT_CAT_WEIGHTS) / sum(LEGIT_CAT_WEIGHTS)
    fraud_w = np.array(FRAUD_CAT_WEIGHTS) / sum(FRAUD_CAT_WEIGHTS)
    pref = rng.dirichlet(legit_w * 12, N_CARDS)
    cdf = np.cumsum(pref[card], axis=1)
    cat_idx = np.minimum((rng.random(n)[:, None] > cdf).sum(axis=1), k - 1)
    cat_idx = np.where(fraud, rng.choice(k, size=n, p=fraud_w), cat_idx)

    # store location: near home, except ~60% of fraud which happens in another city
    store_lat = h_lat + rng.normal(0, 0.12, n)
    store_lon = h_lon + rng.normal(0, 0.15, n)
    away = fraud & (rng.random(n) < 0.6)
    other = (home_city[card] + rng.integers(1, len(CITIES), n)) % len(CITIES)
    store_lat = np.where(away, city_lat[other] + rng.normal(0, 0.1, n), store_lat)
    store_lon = np.where(away, city_lon[other] + rng.normal(0, 0.1, n), store_lon)

    # device (cardholder's phone) location: at the store (~90 m GPS noise)...
    dev_lat = store_lat + rng.normal(0, 0.0008, n)
    dev_lon = store_lon + rng.normal(0, 0.0008, n)
    # ...3% of the time a coarse fix (VPN / cell tower, up to tens of km off)...
    coarse = rng.random(n) < 0.03
    dev_lat = np.where(coarse, store_lat + rng.normal(0, 0.25, n), dev_lat)
    dev_lon = np.where(coarse, store_lon + rng.normal(0, 0.25, n), dev_lon)
    # ...and when fraud happens away from home the real phone is still at home
    phone_home = away & (rng.random(n) < 0.85)
    dev_lat = np.where(phone_home, h_lat + rng.normal(0, 0.01, n), dev_lat)
    dev_lon = np.where(phone_home, h_lon + rng.normal(0, 0.01, n), dev_lon)

    out = df.copy()
    out["card_id"] = card
    out["merchant_category"] = np.array(CATEGORIES)[cat_idx]
    out["store_lat"], out["store_lon"] = store_lat, store_lon
    out["device_lat"], out["device_lon"] = dev_lat, dev_lon
    return out
