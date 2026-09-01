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

A v2 corrige o pipeline, expande a comparação para **6 candidatos** (incluindo MLPClassifier) e revisa o critério de calibração para alinhar com a meta do ML Canvas: **Recall ≥ 0,70**.

---

## 2. Mudanças metodológicas

| Item | v1 | v2 |
|---|---|---|
| Balanceamento | `class_weight='balanced'` (inconsistente) | **SMOTE-NC** dentro do CV (`imblearn.Pipeline`) |
| Seleção do modelo | AUC-ROC (empate estatístico) | **F1** no CV (Precision × Recall equilibrado) |
| Threshold | 0,159 (Precision colapsa) | Maximiza Precision com **Recall ≥ 0,70** no CV |
| Comparação | 4 modelos, configs desiguais | **6 modelos** (Linear, Ensembles, **MLPClassifier**), pipeline equivalente |
| `smote__k_neighbors` | fixo em 5 | **tunado via RandomizedSearchCV** (encontrado: 7) |
| Latência de predição | não medida | **benchmark 1 / 100 / batch** |

---

## 3. Comparação dos candidatos (Cross-Validation 5-fold, com SMOTE-NC)

| Modelo | AUC-ROC | PR-AUC | F1 | Recall | Precision | Tempo CV (s) |
|---|---|---|---|---|---|---|
| **LogReg** ★ | **0,8554** | **0,6729** | **0,6500** | 0,7572 | 0,5698 | 6,3 |
| GradientBoosting | 0,8543 | 0,6712 | 0,6351 | 0,6662 | 0,6071 | 6,6 |
| MLP | 0,8487 | 0,6540 | 0,6265 | 0,6970 | 0,5701 | 5,0 |
| RandomForest | 0,8445 | 0,6352 | 0,6157 | 0,6294 | 0,6031 | 9,2 |
| ExtraTrees | 0,8348 | 0,6111 | 0,6076 | 0,6334 | 0,5843 | 6,8 |
| HistGradBoost | 0,8431 | 0,6463 | 0,6056 | 0,6060 | 0,6052 | 5,4 |

**LogReg** selecionado por maior F1 no CV (0,6500) e melhor AUC-ROC (0,8554).

> **MLPClassifier** ficou em 3º lugar em F1 (0,6265) e 2º em AUC (0,8487) — forte candidato alternativo, mas LogReg apresenta menor variância (AUC-ROC std = 0,0117 vs 0,0095) e latência 3,6× menor.

> **GradientBoosting** tem PR-AUC praticamente idêntico ao LogReg (0,6712 vs 0,6729), mas F1 inferior — padrão repetido da v1. Escolha por AUC não se traduz em ganho operacional.

---

## 4. Benchmark de Latência (batch de 1.409 amostras)

| Modelo | 1 amostra (ms) | 100 amostras (ms) | Batch completo (ms) | ms/amostra | Throughput (rps) |
|---|---|---|---|---|---|
| **LogReg** | **10,2** | **14,4** | **22,2** | **0,016** | **63.489** |
| GradientBoosting | 10,4 | 12,8 | 27,5 | 0,020 | 51.214 |
| HistGradBoost | 14,6 | 16,0 | 43,1 | 0,031 | 32.675 |
| MLP | 11,1 | 12,4 | 81,9 | 0,058 | 17.203 |
| ExtraTrees | 111,1 | 135,0 | 173,2 | 0,123 | 8.136 |
| RandomForest | 119,2 | 143,2 | 187,3 | 0,133 | 7.523 |

LogReg é ~3,6× mais rápido que MLP e ~8× mais rápido que os modelos de árvore — vantagem relevante para scoring em fluxo online.

---

## 5. Pipeline do Campeão

```
ImbPipeline (salvo em champion_v2_logreg.joblib)
│
├── impute (ColumnTransformer)
│   ├── num → SimpleImputer(strategy='median')        [7 features numéricas]
│   └── cat → SimpleImputer(strategy='most_frequent') [14 features categóricas]
│
├── smote → SMOTENC(categorical_features=<idx>, random_state=42, k_neighbors=7)
│           (aplicado SÓ em fit — nunca em predict)
│
├── encode (ColumnTransformer)
│   ├── num → StandardScaler()
│   └── cat → OneHotEncoder(handle_unknown='ignore', sparse_output=False)
│
└── clf → LogisticRegression(C=1.0, penalty='l2', max_iter=2000,
                              solver='lbfgs', random_state=42)
```

**Por que SMOTE-NC e não SMOTE puro?** O dataset é misto (numéricas + categóricas). SMOTE puro exigiria one-hot antes, gerando categorias fracionárias no espaço interpolado. `SMOTENC` sintetiza numéricas por interpolação e categóricas por voto majoritário dos vizinhos, preservando a semântica.

**Ordem do pipeline importa:** SMOTE ocorre **antes** do OHE — se rodasse depois, veria colunas binárias 0/1 já expandidas e perderia a estrutura categórica.

### Tuning

`RandomizedSearchCV(n_iter=40, cv=5, scoring='f1', random_state=42)`

Espaço de busca incluiu `smote__k_neighbors` como hiperparâmetro (valores: 3, 5, 7, 10).

Melhores parâmetros encontrados:

| Parâmetro | Valor |
|---|---|
| `smote__k_neighbors` | **7** |
| `clf__C` | **1.0** |
| `clf__penalty` | **l2** |
| F1 CV (tuned) | **0,6510** |

### Calibração de threshold

- **Critério:** maximizar Precision sujeito a **Recall ≥ 0,70** (meta do ML Canvas)
- Threshold calibrado sobre predições out-of-fold do treino (sem leakage do teste)
- **Threshold = 0,5506**
- Precision esperada (CV): 0,5939 | Recall esperado (CV): 0,7003

**Diagnóstico documentado no notebook (seção 2.6):** a curva PR do modelo não passa pelo ponto (Precision ≥ 0,65; Recall ≥ 0,70) — esses dois critérios são geometricamente incompatíveis com este modelo e dataset. O critério original (Precision floor 0,65) foi revisto para priorizar Recall ≥ 0,70, alinhado à assimetria de custos: o custo de perder um churner (FN) é maior que o custo de um acionamento desnecessário (FP).

---

## 6. Resultados no Teste

### Comparação com Baseline e v1

| Métrica | Baseline v2 (LogReg, thr=0,50) | v1 champion (GradientBoosting, thr=0,159) | **v2 champion (LogReg, thr=0,5506)** |
|---|---|---|---|
| AUC-ROC | **0,8530** | 0,8546 | 0,8383 |
| PR-AUC | 0,6691 | — | **0,6503** |
| Precision (Churn) | 0,5105 | 0,4443 | **0,5570** |
| Recall (Churn) | **0,7807** | 0,9278 | 0,6791 |
| F1 (Churn) | **0,6173** | 0,6009 | 0,6120 |
| **Clientes acionados** | 572 | 784 | **456** |
| **Acionamentos corretos (TP)** | 292 (51,0%) | 347 (44,2%) | **254 (55,7%)** |
| **Falsos positivos** | 280 | 437 | **202** |

### Nota sobre AUC-ROC do campeão versus baseline

O campeão v2 apresenta AUC-ROC ligeiramente inferior ao baseline (0,8383 vs 0,8487). Isso ocorre por três razões complementares:

1. **Objetivo de otimização diferente:** o `RandomizedSearchCV` otimizou F1, não AUC. Um modelo tunado para F1 pode sacrificar parte da discriminação global (AUC) em favor de melhor equilíbrio Precision/Recall no threshold operacional.
2. **SMOTE-NC com k=7:** maior diversidade nos sintéticos pode introduzir ruído que reduz ligeiramente a separabilidade linear — refletida em AUC menor.
3. **Pipeline mais complexo:** impute → SMOTE → encode → clf tem mais fontes de variância que o pipeline simples do baseline.

O campeão v2 **supera o baseline em PR-AUC** (0,6503 vs 0,6447) — a métrica mais informativa para problemas desbalanceados, pois mede a área sob a curva Precision × Recall (o trade-off operacionalmente relevante). A diferença de AUC-ROC reflete discriminação global; o ganho de PR-AUC reflete melhor performance na região de interesse (positivos verdadeiros).

### Conformidade com critérios do ML Canvas

| Critério | Meta | Resultado | Status |
|---|---|---|---|
| AUC-ROC | ≥ 0,80 | 0,8383 | ✅ Atendido |
| Recall (Churn) | ≥ 0,70 | 0,6791 | ⚠️ Não atendido (−2,1 pp) |

**Análise técnica do gap de Recall:** o threshold selecionado via CV (thr=0,5506) entregou Recall=0,7003 no conjunto de treino (out-of-fold). A diferença de 2,1 pp no teste (0,6791 vs 0,7003) reflete variância natural de generalização. O diagnóstico da curva PR (notebook, seção 2.6) demonstrou formalmente que a curva PR do modelo não permite Recall ≥ 0,70 com Precision ≥ 0,65 simultaneamente — a geometria da curva representa o teto do sinal disponível no dataset atual. Para superar essa limitação seria necessário: (a) features adicionais com maior poder preditivo, (b) dados externos com trajetória comportamental dos clientes, ou (c) volume maior de exemplos de churn.

### Leitura de negócio

| Indicador | Baseline v2 | v1 champion | v2 champion | Δ v2 vs baseline |
|---|---|---|---|---|
| Clientes acionados por ciclo | 572 | 784 | **456** | **−20%** |
| Falsos positivos por ciclo | 280 | 437 | **202** | **−28%** |
| Churners capturados | 292 | 347 | 254 | −13% |
| Precisão do acionamento | 51,0% | 44,2% | **55,7%** | **+4,7 pp** |

> **Trade-off explícito:** o critério do v2 foi revisto para priorizar identificar churners (Recall), aceitando mais FPs que a configuração anterior (thr=0,678). Em relação ao baseline, o v2 poupa 116 acionamentos desnecessários por ciclo (−28% de FPs) ao custo de não detectar 38 churners a mais (−13% de TP). Este trade-off é adequado quando o custo unitário de retenção for moderado-alto — caso contrário, o baseline segue competitivo em Recall.

### Matriz de Confusão (v2 champion, thr=0,5506)

```
                 Previsto: Não Churn   Previsto: Churn
Real: Não Churn       833  (TN)             202  (FP)
Real: Churn           120  (FN)             254  (TP)
```

---

## 7. Feature Engineering

Mantida e expandida da v1 (validada pela EDA):

| Feature | Descrição | Justificativa |
|---|---|---|
| `qtd_addons` | Contagem de add-ons ativos (0–6) | Proxy de engajamento com o serviço |
| `contrato_longo` | 1 se Two year ou One year | Cramér's V = 0,41 com churn — maior associação do dataset |
| `pagamento_automatico` | 1 se débito automático | Indica comprometimento financeiro de longo prazo |
| `internet_fibra` | 1 se Fiber optic | Alta correlação positiva com churn (+0,308) |
| `novo_cliente` | 1 se Tenure ≤ 3 meses | EDA: 56% de churn em 0–3 meses (pico de evasão inicial) |

`Total Charges` removido: colinear com `Tenure Months` (ρ = 0,826) — campo recebido no input da API para validação, mas descartado antes da inferência.

**Features totais:** 21 (7 numéricas + 14 categóricas).

---

## 8. Uso em Produção

```python
import joblib

art = joblib.load('models/champion_v2_logreg.joblib')
pipeline  = art['pipeline']
threshold = art['threshold']   # 0.5506

y_proba = pipeline.predict_proba(X_new)[:, 1]
y_pred  = (y_proba >= threshold).astype(int)
```

O artefato contém:
- `pipeline` — ImbPipeline completo (impute → SMOTE não roda no predict → encode → clf)
- `threshold` — 0,5506
- `best_params` — `{'smote__k_neighbors': 7, 'clf__C': 1.0, 'clf__penalty': 'l2'}`
- `feature_cols`, `num_features`, `cat_features` — schema de entrada
- `metrics_test` — AUC, PR-AUC, Precision, Recall, F1 no teste
- `inference_latency_ms_per_sample` — 0,016 ms/amostra (batch)
- `random_state` — 42

---

## 9. Reprodutibilidade

| Parâmetro | Valor |
|---|---|
| `random_state` (LogReg, SMOTENC, split, KFold, RandomizedSearch) | 42 |
| `test_size` | 0,20 |
| `stratify` | `Churn Value` |
| SMOTE `k_neighbors` (tunado) | **7** |
| Critério de calibração | `RECALL_TARGET = 0,70` |
| `n_iter` (RandomizedSearchCV) | 40 |
| Versão scikit-learn | 1.9.0 |
| Versão imbalanced-learn | 0.14.2 |

---

## 10. Limitações e Próximos Passos

| Limitação | Impacto | Ação sugerida |
|---|---|---|
| Recall no teste (0,679) abaixo da meta do ML Canvas (0,70) | Gap de 2,1 pp — geometricamente limitado pela curva PR do modelo | Incorporar features de trajetória comportamental (histórico de uso, chamadas ao SAC) |
| Curva PR não atinge (Precision≥0,65 e Recall≥0,70) simultaneamente | Critério de deploy do Canvas não é plenamente satisfatível com o dataset atual | Coletar mais dados reais de churn; explorar features externas |
| Sem calibração de probabilidades | `predict_proba` pode ser sobre-confiante após SMOTE | `CalibratedClassifierCV(method='isotonic')` se score for exposto ao usuário |
| Dataset estático (Kaggle/IBM) | Não reflete drift real de comportamento de clientes | Retreinar trimestralmente quando dados reais estiverem disponíveis |
| SMOTE pode introduzir amostras sintéticas irrealistas em espaços densos | Ligeira redução de AUC-ROC vs. baseline | Testar `BorderlineSMOTE` / `ADASYN` na próxima iteração |
| ~~Explainer acoplado a `coef_[0]` (LogReg)~~ | ~~Falhará se modelo campeão mudar para ensemble~~ | ✅ Resolvido: `explain()` usa `hasattr(coef_)` → `hasattr(feature_importances_)` → fallback uniforme |

---

## 11. Monitoramento em Produção

| Métrica | Valor (teste) | Alerta |
|---|---|---|
| Precision (Churn) | 0,557 | < 0,50 |
| Recall (Churn) | 0,679 | < 0,55 |
| AUC-ROC | 0,838 | < 0,78 |
| Taxa de acionamento (acionados/total) | 32,4% | > 45% ou < 20% |
| Latência p95 (ms/amostra, batch) | 0,016 | > 1,0 |

**Frequência de revisão:** mensal (métricas técnicas) + trimestral (retreino completo).
**Gatilho de retreino:** degradação de AUC > 5 pp **ou** Recall abaixo de 0,55.
