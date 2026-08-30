# Model Card — Campeão v2: Logistic Regression (Churn)

**Projeto:** Tech Challenge 1 — Pipeline Preditivo de Churn
**Autor:** Jonathan Costa — FIAP Pós-Graduação em Machine Learning
**Data:** Agosto/2026
**Arquivo do modelo:** `models/champion_v2_logreg.joblib`
**Baseline:** `models/baseline_logistic_regression.joblib` — [baseline_model_card.md](./baseline_model_card.md)
**Supersede:** `models/champion_gradientboosting.joblib` (v1 — depreciado, Precision 0,44 inaceitável)
**Revisão EDA aplicada:** ver [eda_and_model_revision.md](./eda_and_model_revision.md) — fix de imputação, colapso de categorias colineares, drop de features irrelevantes e feature `novo_cliente`.

---

## 1. Motivação para a v2

A versão anterior (GradientBoosting + threshold 0,159) atingiu Recall 0,928 mas **Precision de apenas 0,44** — 437 clientes contatados sem necessidade a cada ciclo. Auditoria identificou três falhas metodológicas:

1. Comparação injusta entre candidatos (uns com `class_weight='balanced'`, outros não).
2. Seleção por AUC-ROC com diferença estatisticamente irrelevante (0,001 vs. LogReg L1).
3. Recall "corrigido" via threshold artificial (0,159), colapsando a Precision.

A v2 corrige o pipeline e realinha o critério ao objetivo de negócio: **reduzir o volume de acionamentos desnecessários**.

---

## 2. Mudanças metodológicas

| Item | v1 | v2 |
|---|---|---|
| Balanceamento | `class_weight='balanced'` (inconsistente) | **SMOTE-NC** dentro do CV (`imblearn.Pipeline`) |
| Seleção do modelo | AUC-ROC (empate estatístico) | **F1** no CV (Precision × Recall equilibrado) |
| Threshold | 0,159 (Precision colapsa) | Maximiza Recall com **Precision ≥ 0,65** no CV |
| Comparação | 4 modelos, configs desiguais | 5 modelos, pipeline equivalente |
| Latência de predição | não medida | **benchmark 1 / 100 / batch** |

---

## 3. Comparação dos candidatos (Cross-Validation 5-fold, com SMOTE-NC)

| Modelo | AUC-ROC | PR-AUC | F1 | Recall | Precision |
|---|---|---|---|---|---|
| **LogReg** ★ | 0,8520 | 0,6659 | **0,6427** | 0,7505 | 0,5626 |
| GradientBoosting | **0,8575** | **0,6725** | 0,6386 | 0,6716 | 0,6090 |
| RandomForest | 0,8444 | 0,6218 | 0,6152 | 0,6274 | 0,6038 |
| HistGradBoost | 0,8436 | 0,6425 | 0,6082 | 0,6120 | 0,6047 |
| ExtraTrees | 0,8316 | 0,6025 | 0,6004 | 0,6227 | 0,5803 |

**LogReg** venceu por F1 e apresenta o melhor Recall entre os candidatos — importante para o objetivo posterior de calibrar threshold em favor de Precision.

> GradientBoosting tem AUC ligeiramente maior (+0,006), mas F1 pior — repete o padrão da v1. Escolha por AUC não se traduz em ganho operacional.

---

## 4. Benchmark de Latência (batch de 1.409 amostras)

| Modelo | 1 amostra (ms) | 100 amostras (ms) | Batch completo (ms) | ms/amostra | Throughput (rps) |
|---|---|---|---|---|---|
| **LogReg** | 4,6 | 5,0 | 12,6 | **0,009** | **112.068** |
| GradientBoosting | 6,4 | 6,6 | 22,0 | 0,016 | 63.975 |
| HistGradBoost | 8,7 | 9,5 | 23,9 | 0,017 | 59.058 |
| RandomForest | 38,2 | 38,4 | 54,3 | 0,039 | 25.942 |
| ExtraTrees | 39,9 | 50,9 | 70,8 | 0,050 | 19.896 |

LogReg é ~2× mais rápido que o mais rápido dos ensembles e ~4× mais rápido que RandomForest — vantagem operacional relevante para scoring em fluxo online.

---

## 5. Pipeline do Campeão

```
ImbPipeline (salvo em champion_v2_logreg.joblib)
│
├── impute (ColumnTransformer)
│   ├── num → SimpleImputer(strategy='median')       [6 features]
│   └── cat → SimpleImputer(strategy='most_frequent') [16 features]
│
├── smote → SMOTENC(categorical_features=<idx>, random_state=42, k_neighbors=5)
│           (aplicado SÓ em fit — nunca em predict)
│
├── encode (ColumnTransformer)
│   ├── num → StandardScaler()
│   └── cat → OneHotEncoder(handle_unknown='ignore', sparse_output=False)
│
└── clf → LogisticRegression(C=5.0, penalty='l2', max_iter=2000,
                              solver='lbfgs', random_state=42)
```

**Por que SMOTE-NC e não SMOTE puro?** O dataset é misto (numéricas + categóricas). SMOTE puro exigiria one-hot antes, gerando categorias fracionárias no espaço interpolado. `SMOTENC` sintetiza numéricas por interpolação e categóricas por voto majoritário dos vizinhos, preservando a semântica.

**Ordem do pipeline importa:** SMOTE ocorre **antes** do OHE — se rodasse depois, veria colunas binárias 0/1 já expandidas e perderia a estrutura categórica.

### Tuning
`RandomizedSearchCV(n_iter=30, cv=5, scoring='f1', random_state=42)`
Melhores parâmetros: `C=5.0`, `penalty='l2'` — F1 CV = 0,6444.

### Calibração de threshold
- Critério: maximizar Recall com **Precision ≥ 0,65** (curva PR sobre predições cross-val do treino).
- **Threshold = 0,678**
- Precision esperada (CV): 0,650 | Recall esperado (CV): 0,589

---

## 6. Resultados no Teste

### Comparação com Baseline e v1

| Métrica | Baseline v2 (LogReg L2) | v1 champion (GradientBoosting, thr=0,159) | **v2 champion (LogReg, thr=0,672)** |
|---|---|---|---|
| AUC-ROC | 0,8487 | 0,8546 | 0,8394 |
| PR-AUC | 0,6447 | — | 0,6515 |
| Precision (Churn) | 0,5113 | 0,4443 | **0,6160** |
| Recall (Churn) | 0,7834 | **0,9278** | 0,5749 |
| F1 (Churn) | 0,6188 | 0,6009 | 0,5947 |
| **Clientes acionados** | 573 | **784** | **349** |
| **Acionamentos corretos** | 293 (51%) | 347 (44%) | **215 (62%)** |
| **Falsos positivos** | 280 | 437 | **134** |

### Leitura de negócio

| Indicador | Baseline v2 | v1 champion | v2 champion | Δ v2 vs baseline |
|---|---|---|---|---|
| Clientes acionados por ciclo | 573 | 784 | **349** | **−39%** |
| Falsos positivos por ciclo | 280 | 437 | **134** | **−52%** |
| Churners capturados | 293 | 347 | 215 | −27% |
| Precisão do acionamento | 51,1% | 44,4% | **61,6%** | **+10,5 pp** |

> **Trade-off explícito:** a v2 troca capturar 78 churners a mais (baseline) por poupar 146 acionamentos desnecessários. Vale a pena se o custo unitário de uma ação de retenção (desconto, ligação, brinde) for alto em relação ao valor esperado do cliente que escapa. Se o custo por ação for baixo, o **baseline** ainda é competitivo.

### Matriz de Confusão (v2 champion)

```
                 Previsto: Não Churn   Previsto: Churn
Real: Não Churn       901  (TN)             134  (FP)
Real: Churn           159  (FN)             215  (TP)
```

---

## 7. Feature Engineering

Mantida da v1 (validada pela EDA):

| Feature | Descrição | Correlação c/ Churn |
|---|---|---|
| `qtd_addons` | Contagem de add-ons ativos (0–6) | −0,088 |
| `contrato_longo` | 1 se Two year ou One year | **−0,405** |
| `pagamento_automatico` | 1 se débito automático | −0,210 |
| `internet_fibra` | 1 se Fiber optic | +0,308 |

`Total Charges` removido (colinear com `Tenure Months`, ρ = 0,826).

**Features totais:** 22 (6 numéricas + 16 categóricas).

---

## 8. Uso em Produção

```python
import joblib

art = joblib.load('models/champion_v2_logreg.joblib')
pipeline  = art['pipeline']
threshold = art['threshold']

y_proba = pipeline.predict_proba(X_new)[:, 1]
y_pred  = (y_proba >= threshold).astype(int)
```

O artefato contém:
- `pipeline` — ImbPipeline completo (impute → SMOTE não roda no predict → encode → clf)
- `threshold` — 0,678
- `best_params` — hiperparâmetros do RandomizedSearchCV
- `feature_cols`, `num_features`, `cat_features` — schema de entrada
- `metrics_test` — AUC, PR-AUC, Precision, Recall, F1 no teste
- `inference_latency_ms_per_sample` — 0,009 ms/amostra (batch)
- `random_state` — 42

---

## 9. Reprodutibilidade

| Parâmetro | Valor |
|---|---|
| `random_state` (LogReg, SMOTENC, split, KFold, RandomizedSearch) | 42 |
| `test_size` | 0,20 |
| `stratify` | `Churn Value` |
| SMOTE | `SMOTENC(k_neighbors=5)` |
| `PRECISION_FLOOR` (calibração threshold) | 0,65 |
| Versão scikit-learn | 1.9.0 |
| Versão imbalanced-learn | 0.14.2 |

---

## 10. Limitações e Próximos Passos

| Limitação | Impacto | Ação sugerida |
|---|---|---|
| Recall no teste (0,567) abaixo da meta original de 0,70 | 162 churners não detectados por ciclo | Reduzir `PRECISION_FLOOR` para 0,55 se o custo de FP for baixo |
| Piso de Precision definido a priori (0,65) | Escolha metodológica, não empírica | Fazer análise de custo por FP e por FN com stakeholder de retenção |
| Sem calibração de probabilidades | `predict_proba` pode ser sobre-confiante após SMOTE | `CalibratedClassifierCV(method='isotonic')` se score for exposto ao usuário |
| Dataset estático (Kaggle/IBM) | Não reflete drift real | Retreinar trimestralmente quando dados reais estiverem disponíveis |
| SMOTE pode introduzir amostras sintéticas irrealistas em espaços densos | Ligeira perda de PR-AUC vs. baseline | Testar `BorderlineSMOTE` / `ADASYN` na próxima iteração |

---

## 11. Monitoramento em Produção

| Métrica | Valor | Alerta |
|---|---|---|
| Precision (Churn) | 0,606 | < 0,55 |
| Recall (Churn) | 0,567 | < 0,50 |
| AUC-ROC | 0,832 | < 0,78 |
| Taxa de acionamento (acionados/total) | 24,8% | > 40% ou < 15% |
| Latência p95 (ms/amostra, batch) | 0,009 | > 1,0 |

**Frequência de revisão:** mensal (métricas técnicas) + trimestral (retreino completo).
**Gatilho de retreino:** degradação de AUC > 5 pp **ou** Precision abaixo de 0,55.
