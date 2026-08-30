# Churn Predictor — FIAP POS Tech Challenge 1

API de predição de churn de clientes de telecomunicações, servida via **FastAPI** com modelo campeão `LogisticRegression + SMOTE-NC` (AUC-ROC 0.84, Precision 0.62, threshold 0.6717).

---

## Estrutura do Repositório

```
├── data/                  raw, trusted, refined
├── docs/                  documentação adicional
├── models/                modelos treinados
├── notebooks/             notebooks de EDA e modelagem
├── scripts/               scripts auxiliares
├── src/                   código-fonte do projeto
│   └── churn_predictor/   módulo principal
├── tests/                 testes unitários e de integração
├── .env.example           exemplo de arquivo de variáveis de ambiente
├── Makefile               comandos make para facilitar execução
├── README.md              este arquivo
└── requirements-dev.txt   dependências de desenvolvimento
```

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

---

## Pré-requisitos

- Python 3.11+
- `data/raw/telco_customer_churn.xlsx` (já incluso no repositório)
- `make` instalado (opcional — os comandos equivalentes são listados abaixo)

---

## Como rodar localmente

### 1. Criar o ambiente virtual e instalar dependências

```bash
# Com make
make setup install-dev

# Sem make
python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pip install -e .
```

### 2. Construir os datasets

```bash
# Com make
make data

# Sem make
.venv/bin/python scripts/build_datasets.py
```

### 3. Lint e testes

```bash
# Com make
make lint
make test-cov

# Sem make
.venv/bin/ruff check src/ tests/ scripts/
.venv/bin/python -m pytest --cov=src/churn_predictor --cov-report=term-missing tests/
```

### 4. Iniciar o servidor

```bash
# Com make
make run

# Sem make
.venv/bin/python -m uvicorn churn_predictor.api.app:app \
  --host 0.0.0.0 --port 8000 --reload --app-dir src
```

Acesse: **http://localhost:8000**

---

## Variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto (opcional):

| Variável | Padrão | Descrição |
|---|---|---|
| `MODEL_PATH` | `models/champion_v2_logreg.joblib` | Caminho para o modelo campeão |
| `LOG_LEVEL` | `INFO` | Nível de log |

---

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Status, versão do modelo, threshold |
| POST | `/predict` | Prediz churn para um cliente |
| POST | `/predict/batch` | Prediz em lote |
| GET | `/` | Página HTML interativa |

Veja `docs/api_contract.md` para schemas e exemplos completos.

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

- **Logs JSON** (structlog): cada requisição loga `input_hash` (SHA256 — nunca PII), `latency`, `probability`, `top_contributor`, `model_version`.

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

## Troubleshooting

- **`make: command not found`** — use os comandos equivalentes listados em cada etapa acima.
- **Modelo não encontrado** — verifique se `models/champion_v2_logreg.joblib` existe.
- **Erros 422** — confira os valores de enum em `docs/api_contract.md`.
- **Erros de import** — execute o passo de instalação de dependências novamente.
