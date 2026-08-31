# Churn Predictor — FIAP POS Tech Challenge 1

**Autor:** Jonathan Costa — FIAP Pós-Graduação em Machine Learning  
**Data:** Agosto/2026  
**Modelo em produção:** `LogisticRegression (Baseline v2)` · AUC-ROC 0,849 · Recall 0,783 · threshold 0,50

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
│  · Correlações, distribuições      · 6 candidatos com SMOTE-NC              │
│  · Identificação de leakage        · Tuning RandomizedSearchCV              │
│  · Decisões de feature eng.        · Calibração de threshold (PR curve)     │
│                                    · Benchmark de latência                  │
│                                    · Seleção via SELECTED_MODEL             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │  artefato salvo em
                                       ▼  models/selected_v2_*.joblib
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
│  GET  /health      POST /predict                                            │
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
│   ├── selected_v2_baseline.joblib        ← MODELO EM PRODUÇÃO
│   ├── champion_v2_logreg.joblib          Campeão tunado (referência)
│   └── baseline_logistic_regression.joblib
│
├── notebooks/
│   ├── 01_eda_telco_churn.ipynb       Análise exploratória de dados
│   └── 02_modeling_telco_churn.ipynb  Experimentos, tuning e seleção do modelo final
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
| `02_modeling_telco_churn.ipynb` | Baseline LogReg · comparação de 6 candidatos (incluindo MLPClassifier) com pipeline equivalente (SMOTE-NC + CV) · tuning `RandomizedSearchCV` (com `smote__k_neighbors`) · diagnóstico e calibração de threshold pela curva PR · benchmark de latência · seleção do modelo final via `SELECTED_MODEL` |

Para executar os notebooks, certifique-se de que os datasets refined existem (`make data`) e abra com Jupyter ou VS Code.

---

## Decisão do Modelo Final

### Comparação dos Candidatos (Cross-Validation 5-fold, SMOTE-NC)

| Modelo | AUC-ROC | PR-AUC | F1 | Recall | Precision | ms/amostra |
|---|---|---|---|---|---|---|
| **LogReg L2** ★ | **0,855** | **0,673** | **0,650** | 0,757 | 0,570 | **0,016** |
| GradientBoosting | 0,854 | 0,671 | 0,635 | 0,666 | 0,607 | 0,020 |
| MLP | 0,849 | 0,654 | 0,627 | 0,697 | 0,570 | 0,058 |
| RandomForest | 0,845 | 0,635 | 0,616 | 0,629 | 0,603 | 0,133 |
| ExtraTrees | 0,835 | 0,611 | 0,608 | 0,633 | 0,584 | 0,123 |
| HistGradBoost | 0,843 | 0,646 | 0,606 | 0,606 | 0,605 | 0,031 |

### Baseline v2 vs. Campeão v2 no Teste

| Métrica | Baseline v2 (thr=0,50) | Campeão v2 (thr=0,5506) | Δ |
|---|---|---|---|
| AUC-ROC | **0,8487** | 0,8383 | −0,0104 |
| **Recall** | **0,7834** | 0,6791 | **−0,1043** |
| F1 | **0,6188** | 0,6120 | −0,0068 |
| Precision | 0,5113 | **0,5570** | +0,0457 |
| PR-AUC | 0,6447 | **0,6503** | +0,0056 |

**Modelo escolhido: Baseline v2.** O campeão tunado ganha apenas em Precision (+4,6 pp) e PR-AUC marginalmente (+0,6 pp), mas perde **10 pp de Recall** — a métrica mais crítica para churn. O custo de não identificar um churner (FN) é maior que o custo de acionar um cliente que não ia cancelar (FP). O ajuste de threshold do campeão simplesmente trocou recall por precisão de forma desvantajosa para o problema de negócio.

### Seleção do modelo final via parâmetro

O notebook `02_modeling_telco_churn.ipynb` expõe um parâmetro na célula de setup para alternar o modelo persistido sem alterar o código:

```python
# 'champion' → usa o modelo tunado neste notebook
# 'baseline' → usa o baseline carregado de ../models/baseline_logistic_regression.joblib
SELECTED_MODEL = 'baseline'
```

O artefato gerado é nomeado automaticamente:
- `SELECTED_MODEL = 'baseline'` → `models/selected_v2_baseline.joblib`
- `SELECTED_MODEL = 'champion'` → `models/selected_v2_champion_logreg.joblib`

### Onde está o modelo final

```
models/selected_v2_baseline.joblib
```

O artefato é um dict serializado com `joblib` contendo:

| Chave | Conteúdo |
|---|---|
| `pipeline` | `ImbPipeline` completo (impute → SMOTE-NC → encode → LogReg) |
| `threshold` | 0,50 |
| `selected_model` | `"baseline"` |
| `feature_cols` | Lista das colunas de entrada |
| `num_features` / `cat_features` | Schema de features numéricas e categóricas |
| `metrics_test` | AUC, PR-AUC, Precision, Recall, F1 no teste |
| `best_params` | `{}` (baseline não passa por tuning) |
| `inference_latency_ms_per_sample` | 0,016 ms/amostra (batch) |
| `random_state` | 42 |

---

## Como Rodar Localmente

### Pré-requisitos

- Python 3.11+
- `make` instalado (opcional — comandos equivalentes listados abaixo)

### 1. Criar ambiente virtual e instalar dependências

```bash
# Com make
make setup install

# Sem make
python -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

> **Nota:** o `make setup` atualiza `pip`, `setuptools` e `wheel` juntos — necessário para que o build backend `setuptools.build_meta` (declarado em `pyproject.toml`) funcione corretamente em ambientes com Python 3.13+.

### 2. Configurar variáveis de ambiente (opcional)

```bash
cp .env.example .env
# Edite .env se necessário — os padrões funcionam para uso local
```

Variáveis disponíveis:

| Variável | Padrão | Descrição |
|---|---|---|
| `MODEL_PATH` | `models/selected_v2_baseline.joblib` | Caminho para o artefato do modelo |
| `LOG_LEVEL` | `INFO` | Nível de log (`DEBUG`, `INFO`, `WARNING`) |
| `RAW_DATA_PATH` | `data/raw/telco_customer_churn.xlsx` | Dados brutos de entrada |
| `TRUSTED_DATA_PATH` | `data/trusted/telco_churn_trusted.csv` | Saída da etapa trusted |
| `REFINED_DATA_PATH` | `data/refined/telco_churn_refined.csv` | Saída da etapa refined |

> O threshold **não é configurável por variável de ambiente** — é carregado diretamente do artefato para garantir rastreabilidade.

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

# Sem make
.venv/bin/python -m uvicorn churn_predictor.api.app:app \
  --host 0.0.0.0 --port 8000 --reload --app-dir src
```

Acesse: **http://localhost:8000** (interface web interativa)  
Documentação interativa: **http://localhost:8000/docs**

### 6. Gerar payloads de teste

```bash
# Sem make (20 exemplos, seed fixo)
.venv/bin/python scripts/generate_test_cases.py --n 20 --seed 42
```

### Referência completa do Makefile

| Comando | O que faz |
|---|---|
| `make setup` | Cria `.venv` e atualiza pip, setuptools e wheel |
| `make install` | Instala dependências de produção e o pacote em modo editable |
| `make data` | Executa o pipeline raw → trusted → refined |
| `make lint` | Verifica estilo com ruff |
| `make format` | Corrige estilo automaticamente com ruff |
| `make test` | Roda pytest sem relatório de cobertura |
| `make test-cov` | Roda pytest com cobertura (falha se < 85%) |
| `make run` | Sobe API com hot-reload (desenvolvimento) |
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
| `GET` | `/` | Interface web interativa |
| `GET` | `/docs` | Documentação Swagger interativa (OpenAPI) |

---

### GET /health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_version": "d5e6b522",
  "threshold": 0.5,
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
  "threshold": 0.5,
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
| `model_version` | string | Hash SHA-256 (primeiros 8 chars) do artefato carregado |
| `decision_reason` | string | Resumo textual da decisão com o fator principal |
| `top_contributors[].feature_label` | string | Nome legível do fator em português |
| `top_contributors[].contribution` | float | Peso (coef × valor escalado); positivo = aumenta risco |
| `top_contributors[].direction` | string | `"aumenta risco de churn"` ou `"reduz risco de churn"` |

---

### Enums e Restrições de Input

| Campo | Valores aceitos |
|---|---|
| `Senior Citizen` | `"Yes"`, `"No"` |
| `Partner` | `"Yes"`, `"No"` |
| `Dependents` | `"Yes"`, `"No"` |
| `Multiple Lines` | `"Yes"`, `"No"` |
| `Internet Service` | `"DSL"`, `"Fiber optic"`, `"Cable"`, `"No"` |
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

## Possíveis Vieses e Riscos

### Vieses identificados

| Viés | Descrição | Mitigação atual |
|---|---|---|
| **Viés de seleção** | Dataset Kaggle/IBM é sintético/histórico — não reflete comportamento atual de uma operadora real | Retreino com dados reais assim que disponíveis |
| **Viés demográfico** | `Senior Citizen` é uma das features mais influentes; o modelo pode sistematicamente sub-ponderar clientes sênior | Auditoria de fairness por subgrupo antes de produção real |
| **Viés de sobrevivência** | Dataset contém apenas clientes ativos ou que já cancelaram | Ampliar escopo do dataset |
| **Estabilidade de coeficientes** | `Tenure Months` e `Total Charges` têm correlação 0,826 | `Total Charges` removido do pipeline de produção |
| **SMOTE sintético** | Amostras sintéticas podem não representar comportamento real de churners | Monitorar distribuição dos scores em produção |

### Riscos atuais do projeto

| Risco | Impacto | Probabilidade | Mitigação |
|---|---|---|---|
| Dataset estático (Kaggle) | Drift não detectado em produção real | Alta | Retreino trimestral + monitoramento mensal de AUC |
| Threshold fixo no artefato | Threshold ótimo muda com o perfil da base | Média | Recalibrar na curva PR a cada retreino |
| Sem calibração de probabilidades | `predict_proba` pode ser sobre-confiante após SMOTE | Média | Aplicar `CalibratedClassifierCV(method='isotonic')` |
| Sem autenticação na API | Qualquer cliente pode consultar o modelo | Alta (produção) | Adicionar API key ou OAuth antes de deploy externo |
| Ausência de testes de drift | Degradação silenciosa | Alta | Implementar PSI / KS test nos logs de produção |

---

## Próximos Passos e Evoluções

- [ ] **Análise de custo de FP vs FN com stakeholder** — definir empiricamente o trade-off Precision/Recall baseado no custo real de uma ação de retenção vs. receita perdida por churn
- [ ] **Calibração de probabilidades** — `CalibratedClassifierCV(method='isotonic')` pós-SMOTE para scores mais confiáveis
- [ ] **Autenticação na API** — API key mínima antes de qualquer exposição externa
- [ ] **Auditoria de fairness** — métricas de Precision/Recall por subgrupo (`Senior Citizen`, faixa de `Monthly Charges`)
- [ ] **Monitoramento de drift** — PSI (Population Stability Index) e KS test nos inputs e scores
- [ ] **Containerização** — Dockerfile + docker-compose para deploy reprodutível
- [ ] **CI/CD** — pipeline de retreino automatizado com validação de métricas antes de promote
- [ ] **Features comportamentais** — histórico de chamadas ao suporte, variação mensal de uso, eventos de billing

---

## Documentação Adicional

| Documento | Conteúdo |
|---|---|
| `docs/ml_canvas.md` | Definição do problema, decisions, value proposition e KPIs |
| `docs/baseline_model_card.md` | Pipeline, métricas e interpretabilidade do baseline |
| `docs/champion_model_card.md` | Motivação da v2, comparação de candidatos, métricas de teste, threshold |
| `docs/api_contract.md` | Contratos detalhados de request/response |

---
