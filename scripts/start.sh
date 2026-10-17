#!/usr/bin/env bash
# Train if no model exists yet, then serve. Used by the Dockerfile and by bare-metal deploys.
# Keep a single worker: per-card history is held in process memory.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f models/model.joblib ]; then
  echo "No trained model found - training now..."
  python -m fraud.train
fi

exec python -m uvicorn fraud.service:app --host 0.0.0.0 --port "${PORT:-8000}"
