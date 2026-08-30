# How to Run — Churn Predictor API

## Prerequisites
- Python 3.11+
- `data/raw/telco_customer_churn.xlsx` (already in repo)

## Quick Start

```bash
# 1. Create venv and install dependencies
make setup install-dev

# 2. Build trusted + refined datasets
make data

# 3. Run linter
make lint

# 4. Run tests with coverage
make test-cov

# 5. Start development server
make run
# → http://localhost:8000
```

## Environment Variables

Copy `.env.example` to `.env` and adjust as needed:

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/champion_v2_logreg.joblib` | Path to champion model |
| `THRESHOLD` | `0.6717` | Classification threshold |
| `LOG_LEVEL` | `INFO` | Logging level |
| `PROMETHEUS_ENABLED` | `0` | Enable `/metrics` endpoint |

## Generate test payloads

```bash
make sample
# or
python scripts/generate_test_cases.py --n 20 --seed 42 > /tmp/cases.jsonl
```

## Docker

Docker is planned for a future iteration. Use `make run-prod` for production-like startup (uvicorn with 2 workers):

```bash
make run-prod
```

## Troubleshooting

- **Model not found**: ensure `models/champion_v2_logreg.joblib` exists.
- **422 errors**: check enum values in `docs/api_contract.md`.
- **Import errors**: run `make install-dev` to install all dependencies.
