"""Train the fraud model, pick decision thresholds, evaluate the full engine, save the artifact."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .config import MODEL_DIR, MODEL_FILE, PCA_COLS, SEED
from .data import load_raw
from .decision import APPROVE, DECLINE, FLAG, decide
from .enrich import enrich
from .features import MerchantProfile, feature_columns, featurize_frame


def pick_thresholds(y, p, decline_precision=0.90, flag_precision=0.30):
    """Decline where validation precision >= 90%; ask for 2FA where it is still >= 30%."""
    prec, _, thr = precision_recall_curve(y, p)
    prec = prec[:-1]

    def lowest(min_prec, fallback):
        ok = np.where(prec >= min_prec)[0]
        return float(thr[ok].min()) if len(ok) else fallback

    t_decline = lowest(decline_precision, 0.90)
    t_flag = min(lowest(flag_precision, 0.05), t_decline)
    return {"flag": round(t_flag, 4), "decline": round(t_decline, 4)}


def _evaluate(y, p, X, thresholds, base_rate):
    decisions = np.array([decide(pi, f, thresholds, base_rate)["decision"]
                          for pi, f in zip(p, X.to_dict("records"))])
    out = {
        "n_test": int(len(y)),
        "n_fraud_test": int(y.sum()),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "avg_precision": round(float(average_precision_score(y, p)), 4),
        "avg_precision_random_baseline": round(float(y.mean()), 4),
    }
    for label, name in ((1, "fraud"), (0, "legit")):
        m = decisions[y == label]
        for d in (APPROVE, FLAG, DECLINE):
            out[f"{name}_{d.lower()}_pct"] = round(100 * float((m == d).mean()), 2)
    return out


def train(data_path=None, out_dir=MODEL_DIR, use_pca=True, n_synth=100_000, fraud_rate=0.005,
          max_iter=200, seed=SEED, force_synthetic=False, export_replay=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, is_real = load_raw(data_path, n_synth, fraud_rate, seed, force_synthetic)
    source = "kaggle" if is_real else "synthetic"
    if not is_real:
        print("WARNING: data/creditcard.csv not found -> training on SYNTHETIC data. "
              "Run scripts/get_data.sh for the real Kaggle dataset.")
    df = enrich(raw.sort_values("Time", kind="stable").reset_index(drop=True), seed)

    n = len(df)
    i_tr, i_va = int(n * 0.70), int(n * 0.85)  # chronological 70 / 15 / 15 split
    profile = MerchantProfile.fit(df.iloc[:i_tr])
    X_all, _ = featurize_frame(df, profile)
    cols = feature_columns(use_pca)
    X, y = X_all[cols], df["Class"].to_numpy()

    model = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0, random_state=seed)
    model.fit(X.iloc[:i_tr], y[:i_tr])

    p_va = model.predict_proba(X.iloc[i_tr:i_va])[:, 1]
    thresholds = pick_thresholds(y[i_tr:i_va], p_va)

    p_te = model.predict_proba(X.iloc[i_va:])[:, 1]
    metrics = _evaluate(y[i_va:], p_te, X.iloc[i_va:].reset_index(drop=True), thresholds, profile.base_rate)
    metrics.update({"source": source, "rows": int(n), "use_pca": use_pca,
                    "thresholds": thresholds, "replay_eval_start": int(i_va - i_tr)})

    joblib.dump({
        "model": model, "features": cols, "use_pca": use_pca, "thresholds": thresholds,
        "profile": {"base_rate": profile.base_rate, "cats": profile.cats},
        "trained_on": source, "metrics": metrics,
    }, out_dir / MODEL_FILE)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if export_replay:  # validation + test rows, in order, as API payloads for scripts/simulate.py
        keep = ["card_id", "Time", "Amount", "merchant_category", "store_lat", "store_lon",
                "device_lat", "device_lon", *PCA_COLS, "Class"]
        df.iloc[i_tr:][keep].to_csv(out_dir / "replay.csv", index=False)
    return metrics


def load_artifact(model_dir=MODEL_DIR):
    path = Path(model_dir) / MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m fraud.train` first")
    return joblib.load(path)
