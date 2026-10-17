"""FastAPI scoring service. Run: uvicorn fraud.service:app --host 0.0.0.0 --port 8000

Card history lives in process memory, so run ONE worker (use Redis if you need to scale out).
"""
import math
import os
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from .config import MODEL_DIR, PCA_COLS
from .decision import decide
from .features import CardState, MerchantProfile, featurize
from .model import load_artifact


class Transaction(BaseModel):
    card_id: str
    amount: float = Field(ge=0)
    merchant_category: str
    store_lat: float = Field(ge=-90, le=90)
    store_lon: float = Field(ge=-180, le=180)
    device_lat: float = Field(ge=-90, le=90)
    device_lon: float = Field(ge=-180, le=180)
    timestamp: Optional[float] = Field(default=None, description="epoch seconds; defaults to now")
    pca: Optional[list[float]] = Field(default=None, min_length=28, max_length=28,
                                       description="V1..V28 from the Kaggle data; zeros if omitted")


class Engine:
    def __init__(self, artifact):
        self.model = artifact["model"]
        self.features = artifact["features"]
        self.thresholds = artifact["thresholds"]
        self.profile = MerchantProfile(**artifact["profile"])
        self.info = {"trained_on": artifact["trained_on"], "use_pca": artifact["use_pca"],
                     "thresholds": self.thresholds, "metrics": artifact["metrics"]}
        self.states: dict[str, CardState] = {}
        self.counts: Counter = Counter()
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self.states.clear()
            self.counts.clear()

    def score(self, t: Transaction, commit: bool = True) -> dict:
        if t.merchant_category not in self.profile.cats:
            raise ValueError(f"unknown merchant_category {t.merchant_category!r}; "
                             f"valid: {sorted(self.profile.cats)}")
        t0 = time.perf_counter()
        ts = t.timestamp if t.timestamp is not None else time.time()
        txn = {"ts": ts, "amount": t.amount, "category": t.merchant_category,
               "store_lat": t.store_lat, "store_lon": t.store_lon,
               "device_lat": t.device_lat, "device_lon": t.device_lon}

        with self.lock:
            state = self.states.get(t.card_id) or CardState()
            f = featurize(txn, state, self.profile)
            if commit:
                state.update(ts, t.store_lat, t.store_lon, t.merchant_category)
                self.states[t.card_id] = state

        row = {**f, **dict(zip(PCA_COLS, t.pca or [0.0] * 28))}
        p = float(self.model.predict_proba(pd.DataFrame([row])[self.features])[0, 1])
        out = decide(p, f, self.thresholds, self.profile.base_rate)
        out["fraud_probability"] = round(p, 4)
        out["signals"] = {k: None if math.isnan(f[k]) else round(f[k], 2)
                          for k in ("dev_store_km", "km_from_last", "hours_since_last", "speed_kmh")}
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        if commit:
            with self.lock:
                self.counts[out["decision"]] += 1
        return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.environ.get("FRAUD_MODEL_DIR", MODEL_DIR))
    app.state.engine = Engine(load_artifact(model_dir))
    yield


app = FastAPI(title="Real-Time Hyper-Localized Fraud Detection", lifespan=lifespan)


def _engine(request: Request) -> Engine:
    return request.app.state.engine


@app.get("/health")
def health(request: Request):
    e = _engine(request)
    return {"status": "ok", "cards_tracked": len(e.states), **e.info}


@app.get("/categories")
def categories(request: Request):
    return sorted(_engine(request).profile.cats)


@app.post("/score")
def score(txn: Transaction, request: Request,
          commit: bool = Query(True, description="false = dry run, don't update card history")):
    try:
        return _engine(request).score(txn, commit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/stats")
def stats(request: Request):
    e = _engine(request)
    return {"decisions": dict(e.counts), "cards_tracked": len(e.states)}


@app.post("/reset")
def reset(request: Request):
    _engine(request).reset()
    return {"status": "reset"}
