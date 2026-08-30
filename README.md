# Churn Predictor — FIAP POS Tech Challenge 1

**Autor:** Jonathan Costa — FIAP Pós-Graduação em Machine Learning  
**Data:** Agosto/2026  
**Modelo campeão:** `LogisticRegression + SMOTE-NC` · AUC-ROC 0,84 · Precision 0,62 · threshold 0,672

---

## Problema de Negócio

A Telco é uma operadora de telecomunicações enfrentando aceleração na taxa de cancelamentos. O Custo de Aquisição de Clientes (CAC) é elevado — envolve marketing, vendas e infraestrutura — tornando a **retenção mais rentável que a aquisição**. A diretoria solicitou um modelo preditivo capaz de identificar clientes em risco de churn _antes_ do cancelamento, permitindo intervenções proativas da equipe de retenção (promoções, upgrades, contato direto).

**Objetivo do repositório:** entregar um pipeline de ML completo — da ingestão de dados brutos até uma API REST em produção — que responda à pergunta: _"Este cliente vai cancelar no próximo ciclo?"_

---

## Fluxo End-to-End

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DADOS                                                                      │
│                                                                             │
│  data/raw/                   data/trusted/            data/refined/         │
│  telco_customer_churn.xlsx ──► telco_churn_trusted.csv ──► telco_churn_     │
│  (7.043 clientes, 20 col.)    (limpeza, tipos, NaNs)      refined.csv       │
│                                                           (feature eng.)    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │  scripts/build_datasets.py
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  MODELAGEM  (notebooks/)                                                    │
│                                                                             │
│  01_eda_telco_churn.ipynb          02_modeling_telco_churn.ipynb            │
│  · Análise exploratória            · Baseline LogReg                        │
│  · Correlações, distribuições      · 5 candidatos com SMOTE-NC              │
│  · Identificação de leakage        · Tuning RandomizedSearchCV              │
│  · Decisões de feature eng.        · Calibração de threshold (PR curve)     │
│                                    · Benchmark de latência                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │  artefato salvo em
                                       ▼  models/champion_v2_logreg.joblib
┌─────────────────────────────────────────────────────────────────────────────┐
│  PIPELINE DE PREDIÇÃO  (src/churn_predictor/)                               │
│                                                                             │
│  CustomerProfile (Pydantic)                                                 │
│       │                                                                     │
│       ▼                                                                     │
│  FeatureEngineer                                                            │
│  · qtd_addons · contrato_longo · pagamento_automatico                       │
│  · internet_fibra · novo_cliente · drop Total Charges                       │
│       │                                                                     │
│       ▼                                                                     │
│  ImbPipeline (joblib)                                                       │
│  impute ──► [SMOTE-NC só no fit] ──► encode ──► LogReg                     │
│       │                                                                     │
│       ▼                                                                     │
│  Explainer (coef × valor escalado) ──► top_contributors                    │
│       │                                                                     │
│       ▼                                                                     │
│  PredictionResponse (JSON)                                                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  API REST  (FastAPI · uvicorn)                                              │
│                                                                             │
│  GET  /health      POST /predict      POST /predict/batch                   │
│  GET  /            (UI interativa HTML)                                     │
│                                                                             │
│  Logs JSON estruturados (structlog) · input_hash SHA-256 (sem PII)          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Estrutura do Repositório

```
postech-fiap-tech-challenge-1/
│
├── data/
│   ├── raw/                  Dataset original (telco_customer_churn.xlsx)
│   ├── trusted/              Após limpeza e tipagem (telco_churn_trusted.csv)
│   └── refined/              Após feature engineering (telco_churn_refined.csv)
│
├── docs/
│   ├── ml_canvas.md          Definição do problema, decisões e KPIs
│   ├── baseline_model_card.md  Documentação do modelo baseline
│   ├── champion_model_card.md  Documentação do modelo campeão (decisão + métricas)
│   └── api_contract.md       Contratos de request/response da API
│
├── models/
│   ├── champion_v2_logreg.joblib      ← MODELO EM PRODUÇÃO
│   └── baseline_logistic_regression.joblib
│
├── notebooks/
│   ├── 01_eda_telco_churn.ipynb       Análise exploratória de dados
│   └── 02_modeling_telco_churn.ipynb  Experimentos, tuning e seleção do campeão
│
├── scripts/
│   ├── build_datasets.py     Executa pipeline raw → trusted → refined
│   └── generate_test_cases.py  Gera payloads de teste aleatórios (JSON)
│
├── src/churn_predictor/
│   ├── config.py             Configurações via variáveis de ambiente (pydantic-settings)
│   ├── domain/               Enums e schemas Pydantic (CustomerProfile, PredictionResponse)
│   ├── features/             FeatureEngineer: trusted → model-ready
│   ├── data/                 DataPipeline: raw → trusted → refined
│   ├── models/               ChurnPredictor + Explainer
│   ├── api/                  FastAPI app, rotas, middleware, dependências
│   └── observability/        Logging estruturado JSON (structlog)
│
├── static/index.html         Interface web interativa
├── tests/                    Testes unitários e de integração (pytest)
├── .env.example              Variáveis de ambiente disponíveis
├── Makefile                  Atalhos para operações comuns
├── pyproject.toml            Configuração do projeto, ruff e pytest
└── requirements.txt / requirements-dev.txt
```

---

## Notebooks

| Notebook | Conteúdo |
|---|---|
| `01_eda_telco_churn.ipynb` | Análise exploratória: distribuições, correlações (Cramér's V, Pearson), identificação de leakage, decisões de feature engineering, análise de desbalanceamento |
| `02_modeling_telco_churn.ipynb` | Baseline LogReg · comparação de 5 candidatos com pipeline equivalente (SMOTE-NC + CV) · tuning `RandomizedSearchCV` · calibração de threshold pela curva PR · benchmark de latência · seleção do campeão |

Para executar os notebooks, certifique-se de que os datasets refined existem (`make data`) e abra com Jupyter ou VS Code.

---

## Decisão do Modelo Campeão

### Comparação dos Candidatos (Cross-Validation 5-fold, SMOTE-NC)

| Modelo | AUC-ROC | PR-AUC | F1 | Recall | Precision | ms/amostra |
|---|---|---|---|---|---|---|
| **LogReg L2** ★ | 0,852 | 0,666 | **0,643** | **0,751** | 0,563 | **0,009** |
| GradientBoosting | **0,858** | **0,673** | 0,639 | 0,672 | 0,609 | 0,016 |
| RandomForest | 0,844 | 0,622 | 0,615 | 0,627 | 0,604 | 0,039 |
| HistGradBoost | 0,844 | 0,643 | 0,608 | 0,612 | 0,605 | 0,017 |
| ExtraTrees | 0,832 | 0,603 | 0,600 | 0,623 | 0,580 | 0,050 |

**Por que LogReg e não GradientBoosting?**
- GradientBoosting tem AUC +0,006 — diferença estatisticamente irrelevante no CV
- LogReg vence em **F1** (critério primário) e **Recall** — fundamental para capturar churners
- LogReg é **~2× mais rápido** em inferência (0,009 vs 0,016 ms/amostra) e ~4× mais rápido que RandomForest
- Coeficientes diretamente interpretáveis para o time de retenção
- Seleção por AUC sem critério de negócio levou ao fracasso da v1 (GradientBoosting, Precision 0,44)

### Resultado no Teste (threshold = 0,672)

| Indicador | Baseline | v1 GradBoost | **v2 Campeão** | Δ vs baseline |
|---|---|---|---|---|
| Precision (Churn) | 0,511 | 0,444 | **0,616** | +10,5 pp |
| Recall (Churn) | 0,781 | **0,928** | 0,575 | −20,6 pp |
| AUC-ROC | 0,849 | 0,855 | 0,839 | −1,0 pp |
| Clientes acionados/ciclo | 573 | 784 | **349** | **−39%** |
| Falsos positivos/ciclo | 280 | 437 | **134** | **−52%** |
| Acertos (TP) | 293 | 347 | 215 | −78 |

> **Trade-off explícito:** a v2 troca capturar 78 churners a mais (baseline) por poupar 146 acionamentos desnecessários por ciclo. Justifica-se quando o custo unitário de uma ação de retenção (desconto, ligação, brinde) é alto em relação ao valor do cliente que escapa. Se o custo por ação for baixo, o **baseline ainda é competitivo**.

### Onde está o modelo final

```
models/champion_v2_logreg.joblib
```

O artefato é um dict serializado com `joblib` contendo:

| Chave | Conteúdo |
|---|---|
| `pipeline` | `ImbPipeline` completo (impute → SMOTE-NC → encode → LogReg) |
| `threshold` | 0,672 (calibrado via curva PR, piso Precision ≥ 0,65) |
| `feature_cols` | Lista das 20 colunas de entrada |
| `num_features` / `cat_features` | Schema de features numéricas e categóricas |
| `metrics_test` | AUC, PR-AUC, Precision, Recall, F1 no teste |
| `best_params` | Hiperparâmetros do RandomizedSearchCV |
| `inference_latency_ms_per_sample` | 0,009 ms/amostra (batch) |
| `random_state` | 42 |

---

## Como Rodar Localmente

### Pré-requisitos

- Python 3.11+
- `make` instalado (opcional — comandos equivalentes listados abaixo)

### 1. Criar ambiente virtual e instalar dependências

```bash
# Com make
make setup install-dev

# Sem make
python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pip install -e .
```

### 2. Configurar variáveis de ambiente (opcional)

```bash
cp .env.example .env
# Edite .env se necessário — os padrões funcionam para uso local
```

Variáveis disponíveis:

| Variável | Padrão | Descrição |
|---|---|---|
| `MODEL_PATH` | `models/champion_v2_logreg.joblib` | Caminho para o artefato do modelo |
| `LOG_LEVEL` | `INFO` | Nível de log (`DEBUG`, `INFO`, `WARNING`) |
| `RAW_DATA_PATH` | `data/raw/telco_customer_churn.xlsx` | Dados brutos de entrada |
| `TRUSTED_DATA_PATH` | `data/trusted/telco_churn_trusted.csv` | Saída da etapa trusted |
| `REFINED_DATA_PATH` | `data/refined/telco_churn_refined.csv` | Saída da etapa refined |

> O threshold **não é configurável por variável de ambiente** — é carregado diretamente do artefato `champion_v2_logreg.joblib` para garantir rastreabilidade.

### 3. Construir os datasets

```bash
# Com make
make data

# Sem make
.venv/bin/python scripts/build_datasets.py
```

### 4. Lint e testes

```bash
# Com make
make lint          # ruff check
make format        # ruff format (auto-fix)
make test          # pytest sem cobertura
make test-cov      # pytest com cobertura (exige ≥ 85%)

# Sem make
.venv/bin/ruff check src/ tests/ scripts/
.venv/bin/python -m pytest --cov=src/churn_predictor --cov-report=term-missing tests/
```

### 5. Iniciar o servidor

```bash
# Com make (desenvolvimento, com hot-reload)
make run

# Com make (produção, 2 workers)
make run-prod

# Sem make
.venv/bin/python -m uvicorn churn_predictor.api.app:app \
  --host 0.0.0.0 --port 8000 --reload --app-dir src
```

Acesse: **http://localhost:8000** (interface web interativa)  
Documentação interativa: **http://localhost:8000/docs**

### 6. Gerar payloads de teste

```bash
# Com make (5 exemplos)
make sample

# Sem make (20 exemplos, seed fixo)
.venv/bin/python scripts/generate_test_cases.py --n 20 --seed 42
```

### Referência completa do Makefile

| Comando | O que faz |
|---|---|
| `make setup` | Cria `.venv` e atualiza pip |
| `make install` | Instala dependências de produção |
| `make install-dev` | Instala dependências de desenvolvimento (inclui produção) |
| `make data` | Executa o pipeline raw → trusted → refined |
| `make lint` | Verifica estilo com ruff |
| `make format` | Corrige estilo automaticamente com ruff |
| `make test` | Roda pytest sem relatório de cobertura |
| `make test-cov` | Roda pytest com cobertura (falha se < 85%) |
| `make run` | Sobe API com hot-reload (desenvolvimento) |
| `make run-prod` | Sobe API com 2 workers (produção local) |
| `make sample` | Gera 5 payloads de teste no terminal |
| `make clean` | Remove `__pycache__`, `.pytest_cache`, `.coverage` |

---

## API — Endpoints e Exemplos

### Base URL

```
http://localhost:8000
```

### Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Status, versão do modelo, threshold e uptime |
| `POST` | `/predict` | Predição individual com explicabilidade |
| `POST` | `/predict/batch` | Predição em lote (array de perfis) |
| `GET` | `/` | Interface web interativa |
| `GET` | `/docs` | Documentação Swagger (OpenAPI) |

---

### GET /health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_version": "d5e6b522",
  "threshold": 0.672,
  "feature_contract_hash": "a1b2c3d4",
  "uptime_s": 42.1
}
```

---

### POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "Tenure Months": 2,
    "Monthly Charges": 99.65,
    "Total Charges": 199.30,
    "Senior Citizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "Multiple Lines": "No",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "No",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "No",
    "Streaming Movies": "No",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check"
  }'
```

**Resposta (200)**

```json
{
  "customer_id": null,
  "churn_probability": 0.9582,
  "churn_prediction": true,
  "threshold": 0.672,
  "model_version": "d5e6b522",
  "decision_reason": "Alta probabilidade de churn (95.8%). Principal fator: Sem dependentes.",
  "top_contributors": [
    {
      "feature": "Dependents_No",
      "feature_label": "Sem dependentes",
      "contribution": 0.966,
      "direction": "aumenta risco de churn"
    },
    {
      "feature": "Contract_Month-to-month",
      "feature_label": "Tipo de contrato: mensal",
      "contribution": 0.847,
      "direction": "aumenta risco de churn"
    },
    {
      "feature": "novo_cliente",
      "feature_label": "Cliente novo (≤ 3 meses)",
      "contribution": 0.778,
      "direction": "aumenta risco de churn"
    },
    {
      "feature": "Payment Method_Electronic check",
      "feature_label": "Pagamento: cheque eletrônico",
      "contribution": 0.656,
      "direction": "aumenta risco de churn"
    },
    {
      "feature": "Multiple Lines_No",
      "feature_label": "Linha telefônica única",
      "contribution": -0.474,
      "direction": "reduz risco de churn"
    }
  ]
}
```

**Campos da resposta:**

| Campo | Tipo | Descrição |
|---|---|---|
| `churn_probability` | float | Probabilidade de churn (0–1) |
| `churn_prediction` | bool | `true` se probabilidade ≥ threshold |
| `threshold` | float | Limiar de decisão (do artefato) |
| `model_version` | string | Hash SHA-256 dos primeiros 8 chars do modelo |
| `decision_reason` | string | Resumo textual da decisão com o fator principal |
| `top_contributors[].feature_label` | string | Nome legível do fator em português |
| `top_contributors[].contribution` | float | Peso (coef × valor escalado); positivo = aumenta risco |
| `top_contributors[].direction` | string | `"aumenta risco de churn"` ou `"reduz risco de churn"` |

---

### POST /predict/batch

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H 'Content-Type: application/json' \
  -d '[
    {"Tenure Months": 2, "Monthly Charges": 99.65, "Total Charges": 199.30, ...},
    {"Tenure Months": 48, "Monthly Charges": 45.00, "Total Charges": 2160.00, ...}
  ]'
```

Retorna um array de objetos `PredictionResponse` na mesma ordem dos inputs.

---

### Enums e Restrições de Input

| Campo | Valores aceitos |
|---|---|
| `Senior Citizen` | `"Yes"`, `"No"` |
| `Partner` | `"Yes"`, `"No"` |
| `Dependents` | `"Yes"`, `"No"` |
| `Multiple Lines` | `"Yes"`, `"No"` |
| `Internet Service` | `"DSL"`, `"Fiber optic"`, `"No"` |
| `Online Security` | `"Yes"`, `"No"` |
| `Online Backup` | `"Yes"`, `"No"` |
| `Device Protection` | `"Yes"`, `"No"` |
| `Tech Support` | `"Yes"`, `"No"` |
| `Streaming TV` | `"Yes"`, `"No"` |
| `Streaming Movies` | `"Yes"`, `"No"` |
| `Contract` | `"Month-to-month"`, `"One year"`, `"Two year"` |
| `Paperless Billing` | `"Yes"`, `"No"` |
| `Payment Method` | `"Electronic check"`, `"Mailed check"`, `"Bank transfer (automatic)"`, `"Credit card (automatic)"` |

**Restrições numéricas:**
- `Tenure Months` ≥ 0
- `Monthly Charges` > 0
- `Total Charges` ≥ 0; deve ser 0 quando `Tenure Months` = 0

**Erro 422** (exemplo de campo inválido):
```json
{"detail": [{"loc": ["body", "Contract"], "msg": "Input should be 'Month-to-month', 'One year' or 'Two year'", "type": "enum"}]}
```

---

## Observabilidade

Cada requisição emite dois eventos JSON no stdout (structlog):

```json
{"event": "prediction", "input_hash": "f17d81bcd2ec", "latency": 0.034,
 "probability": 0.958, "label": true, "top_contributor": "Sem dependentes",
 "model_version": "d5e6b522", "route": "/predict", "request_id": "...", "level": "info"}

{"event": "request", "route": "/predict", "status": 200,
 "latency_ms": 36.4, "request_id": "...", "level": "info"}
```

- `input_hash`: SHA-256 do payload (12 chars) — rastreabilidade sem expor PII
- `request_id`: UUID por requisição — correlação entre os dois eventos

---

## Possíveis Vieses e Riscos

### Vieses identificados

| Viés | Descrição | Mitigação atual |
|---|---|---|
| **Viés de seleção** | Dataset Kaggle/IBM é sintético/histórico — não reflete comportamento atual de uma operadora real | Retreino com dados reais assim que disponíveis |
| **Viés demográfico** | `Senior Citizen` é uma das features mais influentes; o modelo pode sistematicamente sub-ponderar clientes sênior como não-churners | Auditoria de fairness por subgrupo antes de produção real |
| **Viés de sobrevivência** | Dataset contém apenas clientes ativos ou que já cancelaram — não inclui clientes que nunca chegaram a converter | Ampliar escopo do dataset |
| **Estabilidade de coeficientes** | `Tenure Months` e `Total Charges` têm correlação 0,826; coeficientes isolados são instáveis | `Total Charges` removido do pipeline de produção |
| **SMOTE sintético** | Amostras sintéticas geradas em regiões densas podem não representar comportamento real de churners | Monitorar distribuição dos scores em produção |

### Riscos atuais do projeto

| Risco | Impacto | Probabilidade | Mitigação |
|---|---|---|---|
| Dataset estático (Kaggle) | Drift não detectado em produção real | Alta | Retreino trimestral + monitoramento mensal de AUC |
| Threshold fixo no artefato | Threshold ótimo muda com o perfil da base | Média | Recalibrar na curva PR a cada retreino |
| Sem calibração de probabilidades | `predict_proba` pode ser sobre-confiante após SMOTE | Média | Aplicar `CalibratedClassifierCV(method='isotonic')` |
| Sem autenticação na API | Qualquer cliente pode consultar o modelo | Alta (produção) | Adicionar API key ou OAuth antes de deploy externo |
| Sem versionamento de modelo na API | Impossível rollback rápido | Média | Adicionar rota `/models` com artefatos disponíveis |
| Ausência de testes de drift | Degradação silenciosa | Alta | Implementar PSI / KS test nos logs de produção |

---

## Próximos Passos e Evoluções

### Curto prazo (próximo ciclo)

- [ ] **Análise de custo de FP vs FN com stakeholder** — definir empiricamente o `PRECISION_FLOOR` (atualmente 0,65 a priori) baseado no custo real de uma ação de retenção vs. receita perdida por churn
- [ ] **Calibração de probabilidades** — `CalibratedClassifierCV(method='isotonic')` pós-SMOTE para scores mais confiáveis
- [ ] **Autenticação na API** — API key mínima antes de qualquer exposição externa
- [ ] **Auditoria de fairness** — métricas de Precision/Recall por subgrupo (`Senior Citizen`, faixa de `Monthly Charges`)

### Médio prazo

- [ ] **Dados reais** — substituir dataset Kaggle por dados da operadora com histórico atualizado
- [ ] **Monitoramento de drift** — PSI (Population Stability Index) e KS test nos inputs e scores
- [ ] **BorderlineSMOTE / ADASYN** — testar alternativas ao SMOTE-NC para reduzir amostras sintéticas em regiões densas
- [ ] **Containerização** — Dockerfile + docker-compose para deploy reprodutível
- [ ] **CI/CD** — pipeline de retreino automatizado com validação de métricas antes de promote

### Longo prazo

- [ ] **Features comportamentais** — histórico de chamadas ao suporte, variação mensal de uso, eventos de billing
- [ ] **Modelo de uplift** — estimar não só _quem_ vai cancelar, mas _quem_ responde positivamente à intervenção
- [ ] **A/B test de intervenção** — medir impacto real da predição na taxa de retenção

---

## Documentação Adicional

| Documento | Conteúdo |
|---|---|
| `docs/ml_canvas.md` | Definição do problema, decisions, value proposition e KPIs |
| `docs/baseline_model_card.md` | Pipeline, métricas e interpretabilidade do baseline |
| `docs/champion_model_card.md` | Motivação da v2, comparação de candidatos, métricas de teste, threshold |
| `docs/api_contract.md` | Contratos detalhados de request/response |

---

## Troubleshooting

| Erro | Causa provável | Solução |
|---|---|---|
| `make: command not found` | `make` não instalado | Use os comandos equivalentes listados em cada etapa acima |
| `ModuleNotFoundError: churn_predictor` | Pacote não instalado em modo editable | `make install-dev` ou `.venv/bin/pip install -e .` |
| `FileNotFoundError: champion_v2_logreg.joblib` | Modelo ausente | Verifique se o arquivo existe em `models/` |
| `FileNotFoundError: telco_customer_churn.xlsx` | Dataset bruto ausente | Verifique se o arquivo existe em `data/raw/` |
| `422 Unprocessable Entity` | Valor de enum inválido ou constraint violada | Consulte a tabela de enums acima |
| Datasets não existem (`/trusted`, `/refined`) | Pipeline de dados não executado | `make data` |
