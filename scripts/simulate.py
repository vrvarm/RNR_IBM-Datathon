"""Replay held-out transactions against a running API and report accuracy + latency.

    python scripts/simulate.py --url http://localhost:8000 --limit 5000

The first half of models/replay.csv is only used to warm up per-card history; metrics cover the
held-out test portion.
"""
import argparse
import json
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PCA = [f"V{i}" for i in range(1, 29)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--replay", default=str(ROOT / "models" / "replay.csv"))
    ap.add_argument("--limit", type=int, default=0, help="only send the first N rows (0 = all)")
    args = ap.parse_args()

    df = pd.read_csv(args.replay)
    start = json.loads((Path(args.replay).parent / "metrics.json").read_text())["replay_eval_start"]
    if args.limit:
        df = df.head(args.limit)
        start = min(start, len(df) // 2)

    decisions, latencies = [], []
    with httpx.Client(base_url=args.url, timeout=10) as client:
        client.post("/reset").raise_for_status()
        t_start = time.perf_counter()
        for r in df.to_dict("records"):
            payload = {
                "card_id": str(int(r["card_id"])), "amount": r["Amount"],
                "merchant_category": r["merchant_category"], "timestamp": r["Time"],
                "store_lat": r["store_lat"], "store_lon": r["store_lon"],
                "device_lat": r["device_lat"], "device_lon": r["device_lon"],
                "pca": [r[c] for c in PCA],
            }
            t0 = time.perf_counter()
            resp = client.post("/score", json=payload)
            latencies.append((time.perf_counter() - t0) * 1000)
            resp.raise_for_status()
            decisions.append(resp.json()["decision"])
        elapsed = time.perf_counter() - t_start

    y = df["Class"].to_numpy()[start:]
    d = np.array(decisions[start:])
    lat = np.array(latencies[start:])
    print(f"\nScored {len(df)} transactions in {elapsed:.1f}s ({len(df) / elapsed:.0f} tx/s); "
          f"evaluating the last {len(y)} ({int(y.sum())} fraud)")
    print(f"Latency ms  p50={np.percentile(lat, 50):.1f}  p95={np.percentile(lat, 95):.1f}  "
          f"p99={np.percentile(lat, 99):.1f}")
    print(pd.crosstab(pd.Series(y, name="actual (1=fraud)"), pd.Series(d, name="decision")))


if __name__ == "__main__":
    main()
