# Churn Predictor — FIAP POS Tech Challenge 1

API de predição de churn de clientes de telecomunicações, servida via **FastAPI** com modelo campeão `LogisticRegression + SMOTE-NC` (AUC-ROC 0.84, Precision 0.62, threshold 0.6717).

---

## Arquitetura

```
data/raw/          → data/trusted/ → data/refined/
                        ↓                ↓
                   ChurnPredictor ← FeatureEngineer
                        ↓
              FastAPI /predict + /health
                        ↓
                  static/index.html
```

- **Track B** — Domain (enums, schemas Pydantic v2) + Feature Engineering
- **Track A** — Data Pipeline (raw → trusted → refined)
- **Track C** — ChurnPredictor + Explainer (contribuições LogReg)
- **Track D** — API FastAPI + HTML
- **Track E** — Tooling (Makefile, pyproject, requirements)
- **Track F** — Observabilidade (structlog JSON)

---

## Quickstart

```bash
# Setup
make setup install-dev

# Build datasets
make data

# Lint + tests
make lint
make test-cov

# Start server
make run
# → http://localhost:8000
```

---

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Status, versão do modelo, threshold |
| POST | `/predict` | Prediz churn para um cliente |
| POST | `/predict/batch` | Prediz em lote |
| GET | `/` | Página HTML interativa |
Veja `docs/api_contract.md` para exemplos completos.

---

## Modelo Campeão

| Métrica | Valor |
|---|---|
| AUC-ROC | 0.8394 |
| Precision | 0.616 |
| Recall | 0.575 |
| F1 | 0.595 |
| Threshold | 0.6717 |
| Latência/amostra | 0.008 ms |

---

## Observabilidade

- **Logs JSON** (structlog): cada requisição loga `request_id`, `latency_ms`, `input_hash` (SHA256 — nunca PII), `probability`, `top_contributor`.

---

## Estrutura

```
src/churn_predictor/
├── config.py            pydantic-settings
├── domain/              enums + schemas Pydantic
├── features/            FeatureEngineer
├── data/                DataPipeline (raw→trusted→refined)
├── models/              ChurnPredictor + Explainer
├── api/                 FastAPI app, routes, middleware
└── observability/       structlog
```

---

## Documentação

- `docs/api_contract.md` — schemas, exemplos, enums
- `docs/how_to_run.md` — setup, variáveis de ambiente
- `docs/validation_report.md` — checklist executado com evidências
