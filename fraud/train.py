"""CLI: python -m fraud.train [--synthetic] [--no-pca]"""
import argparse
import json

from .config import DATA_PATH, MODEL_DIR
from .model import train


def main():
    ap = argparse.ArgumentParser(description="Train the fraud model")
    ap.add_argument("--data", default=str(DATA_PATH), help="path to creditcard.csv")
    ap.add_argument("--out", default=str(MODEL_DIR), help="output directory for the model artifact")
    ap.add_argument("--synthetic", action="store_true", help="ignore the CSV and use synthetic data")
    ap.add_argument("--no-pca", action="store_true", help="drop V1..V28; use only the three layers + amount")
    ap.add_argument("--max-iter", type=int, default=200)
    args = ap.parse_args()

    metrics = train(args.data, args.out, use_pca=not args.no_pca, max_iter=args.max_iter,
                    force_synthetic=args.synthetic)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
