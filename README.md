# Real-Time Hyper-Localized Fraud Detection

IBM Datathon project. Every card transaction is scored in milliseconds and gets one of three decisions: **APPROVE**, **FLAG_2FA** (step-up authentication) or **DECLINE**.

The service evaluates three real-time layers per transaction, then combines them with a gradient-boosted model:

| Layer | Question | Signals |
|---|---|---|
| 1. Location | Is the cardholder's device where the store is? | device-to-store distance, store vs. the card's usual area |
| 2. Velocity | Could they physically have travelled here? | km and minutes since the card's previous transaction, implied km/h, transactions in the last hour |
| 3. Merchant profile | Does this fit how this card shops? | category fraud risk, amount vs. category norm, share of the card's history in this category, first purchase in a new category |

Decision logic (`fraud/decision.py`): the model's fraud probability sets the base action using two thresholds tuned on a validation split. Each layer can also raise a rule-based alert: one alert escalates an approval to 2FA, two alerts decline. Every response lists the reasons.

## Quick start

```bash
pip install -r requirements.txt
./scripts/get_data.sh            # real Kaggle data (needs Kaggle credentials); optional
python -m fraud.train            # uses data/creditcard.csv if present, else synthetic data
python -m pytest -q
./scripts/start.sh               # API on :8000, docs at /docs
python scripts/simulate.py       # replay held-out transactions, report accuracy + latency
```

Example request:

```bash
curl -X POST localhost:8000/score -H 'content-type: application/json' -d '{
  "card_id": "4242", "amount": 899.99, "merchant_category": "electronics",
  "store_lat": 40.71, "store_lon": -74.01, "device_lat": 32.78, "device_lon": -96.80
}'
```

Endpoints: `POST /score` (add `?commit=false` for a dry run), `GET /health`, `GET /stats`, `GET /categories`, `POST /reset`.

## Deploy to LinuxONE

```bash
git remote add origin git@github.com:<you>/<repo>.git
git push -u origin main
DEPLOY_HOST=user@your-linuxone-host ./scripts/deploy.sh
```

`deploy.sh` clones or pulls the repo on the server, then runs `docker compose up -d --build` (or a venv + nohup if docker is missing). The container trains the model on first start, so the artifact is built natively on the server's architecture (s390x is big-endian). Put `creditcard.csv` in the server's `data/` folder before the first start to train on the real data.

## Important caveats

- **Geography and merchants are simulated.** The Kaggle dataset is PCA-anonymised (`V1..V28`, `Amount`, `Time`, `Class`) and has no locations, card IDs or merchant categories. `fraud/enrich.py` adds them, with a share of fraud occurring in another city while the cardholder's phone stays home. Velocity is not injected: it comes from each card's sequence of store locations. Say this in your presentation; swap in real geo/merchant data if you can get it.
- **Metrics on synthetic data are optimistic.** Treat `models/metrics.json` as a pipeline check, not a performance claim. Retrain on the real CSV for real numbers.
- **`V1..V28` don't exist in a live system.** The API accepts them as an optional `pca` array (zeros if omitted) so the Kaggle replay works. `python -m fraud.train --no-pca` trains a model on the three layers plus amount only.
- **One worker only.** Per-card history is held in memory. To scale out, move `CardState` to Redis.
- **Not yet tested on s390x.** All numpy/pandas/scikit-learn packages must install there; the Dockerfile includes build tools as a fallback if a wheel is missing (a source build can be slow).

## Results

Trained on the real Kaggle credit-card fraud dataset (284,807 transactions, 492 fraud). Chronological split: 70% train, 15% validation (decision thresholds tuned here), 15% test. Test set: 42,722 transactions, 52 fraud.

| Metric | Value |
|---|---|
| ROC AUC | 0.975 |
| Average precision (random baseline 0.0012) | 0.898 |
| Fraud declined | 90.4% (47 of 52) |
| Fraud sent to 2FA | 3.8% (2 of 52) |
| Fraud approved (missed) | 5.8% (3 of 52) |
| Legit approved | 98.8% |
| Legit sent to 2FA | 1.2% |
| Legit declined | 0.02% |
| Median scoring latency (API, LinuxONE s390x) | about 7 ms |

Thresholds: 2FA at fraud probability 0.0012, decline at 0.1267, plus rule alerts from the three layers (two alerts decline).

Caveats: the test set has only 52 fraud cases, so rates are noisy. The Kaggle data has no locations, card IDs or merchant categories, so those fields are simulated on top of the real transactions (see fraud/enrich.py). Treat the results as a working end-to-end pipeline on real transaction data, not a claim about performance on real geography.
