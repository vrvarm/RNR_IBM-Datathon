"""Shared constants for the fraud pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
MODEL_FILE = "model.joblib"
SEED = 42

N_CARDS = 2000
PCA_COLS = [f"V{i}" for i in range(1, 29)]

# --- Merchant categories (simulated: the Kaggle dataset has none) ------------
CATEGORIES = [
    "grocery", "gas", "restaurant", "retail", "pharmacy",
    "entertainment", "travel", "electronics", "jewelry", "atm_cash",
]
LEGIT_CAT_WEIGHTS = [0.26, 0.18, 0.20, 0.14, 0.07, 0.06, 0.03, 0.04, 0.01, 0.01]
FRAUD_CAT_WEIGHTS = [0.04, 0.08, 0.03, 0.10, 0.02, 0.05, 0.14, 0.28, 0.13, 0.13]

# --- Home cities used to simulate card / store / device coordinates ----------
CITIES = [
    ("Dallas", 32.78, -96.80), ("New York", 40.71, -74.01),
    ("Los Angeles", 34.05, -118.24), ("Chicago", 41.88, -87.63),
    ("Miami", 25.76, -80.19), ("Seattle", 47.61, -122.33),
    ("Atlanta", 33.75, -84.39), ("Denver", 39.74, -104.99),
    ("Boston", 42.36, -71.06), ("Phoenix", 33.45, -112.07),
]

# --- Rule thresholds for the three layers ------------------------------------
LOCATION_WARN_KM = 10      # device this far from the store -> warn
LOCATION_ALERT_KM = 50     # ... -> alert
SPEED_WARN_KMH = 250       # implied travel speed (needs >= MIN_WARN_KM)
SPEED_ALERT_KMH = 900      # faster than a commercial jet (needs >= MIN_ALERT_KM)
MIN_WARN_KM = 50
MIN_ALERT_KM = 100
AMOUNT_Z_WARN = 2.5        # log-amount z-score within the merchant category
AMOUNT_Z_ALERT = 3.0
