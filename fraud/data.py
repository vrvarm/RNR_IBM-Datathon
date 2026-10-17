"""Load the Kaggle credit-card fraud data, or build a synthetic stand-in with the same schema."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_PATH, PCA_COLS, SEED

REQUIRED = ["Time", "Amount", "Class", *PCA_COLS]


def synth_raw(n=100_000, fraud_rate=0.005, seed=SEED):
    """Synthetic data shaped like mlg-ulb/creditcardfraud (Time, V1..V28, Amount, Class)."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 172_792, n)).round(0)
    y = (rng.random(n) < fraud_rate).astype(int)
    fraud = y == 1

    v = rng.normal(size=(n, 28))
    # shift the same components that separate fraud in the real dataset (V1,V3,V4,V10,V11,V12,V14,V17)
    for j, shift in {0: -0.9, 2: -1.1, 3: 0.9, 9: -1.0, 10: 0.8, 11: -1.1, 13: -1.2, 16: -0.9}.items():
        v[fraud, j] += shift

    amount = np.where(fraud, rng.lognormal(4.2, 1.5, n), rng.lognormal(3.4, 1.2, n))
    amount = np.clip(amount, 0.01, 5000).round(2)

    df = pd.DataFrame(v, columns=PCA_COLS)
    df.insert(0, "Time", t)
    df["Amount"] = amount
    df["Class"] = y
    return df


def load_raw(path=None, n_synth=100_000, fraud_rate=0.005, seed=SEED, force_synthetic=False):
    """Return (dataframe, is_real). Uses the Kaggle CSV when present, otherwise synthetic data."""
    path = Path(path) if path else DATA_PATH
    if not force_synthetic and path.exists():
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        return df, True
    return synth_raw(n_synth, fraud_rate, seed), False
