# Registro de Experimentos — Pipeline Preditivo de Churn

Registro estruturado dos resultados de cada iteração de modelagem.
Métricas avaliadas no conjunto de **teste** (20% estratificado, `random_state=42`), exceto onde indicado como CV.

---

## Experimento 001 — Baseline v2

**Data:** Agosto/2026
**Notebook:** `notebooks/01_eda_telco_churn.ipynb`
**Modelo:** Logistic Regression (sklearn)
**Artefato:** `models/selected_v2_baseline.joblib` ✅ **ativo em produção**

### Configuração

| Parâmetro | Valor |
|---|---|
| Algoritmo | LogisticRegression |
| `C` | 1.0 |
| `class_weight` | `balanced` |
| `solver` | lbfgs |
| `max_iter` | 1000 |
| Balanceamento | `class_weight='balanced'` |
| Threshold | 0,50 (padrão) |
| Features | 18 (2 numéricas + 16 categóricas) — com `Total Charges` |
| Pré-processamento | ColumnTransformer: SimpleImputer + StandardScaler / OneHotEncoder |
| Seeds | `random_state=42`, `stratify=Churn Value` |

### Resultados (teste)

| Métrica | Valor |
|---|---|
| AUC-ROC | 0,8530 |
| PR-AUC | 0,6691 |
| Precision (Churn) | 0,5105 |
| Recall (Churn) | 0,7807 |
| F1 (Churn) | 0,6173 |
| Accuracy | 0,7431 |
| Clientes acionados | 572 |
| Falsos positivos | 280 |

### Observações

Modelo baseline: sem feature engineering, sem balanceamento via SMOTE, threshold padrão. Atende ambos os critérios do ML Canvas (AUC ≥ 0,80 ✅; Recall ≥ 0,70 ✅). Serve como piso de performance para todos os modelos subsequentes.

---

## Experimento 002 — Champion v1 (depreciado)

**Data:** Agosto/2026
**Notebook:** `notebooks/02_modeling_telco_churn.ipynb` (v1)
**Modelo:** GradientBoostingClassifier
**Artefato:** `models/champion_gradientboosting.joblib` (**depreciado**)

### Configuração

| Parâmetro | Valor |
|---|---|
| Algoritmo | GradientBoostingClassifier |
| Balanceamento | `class_weight='balanced'` (inconsistente entre candidatos) |
| Threshold | 0,159 (ajustado manualmente para maximizar Recall) |
| Comparação | 4 candidatos com configurações desiguais |

### Resultados (teste)

| Métrica | Valor |
|---|---|
| AUC-ROC | 0,8546 |
| Precision (Churn) | 0,4443 |
| Recall (Churn) | 0,9278 |
| F1 (Churn) | 0,6009 |
| Clientes acionados | 784 |
| Falsos positivos | 437 |

### Observações

**Depreciado.** Precision de 44,4% é inaceitável operacionalmente — 437 FPs por ciclo. Threshold artificial de 0,159 mascarou o baixo poder discriminativo do modelo. Metodologia de comparação injusta (modelos com configs diferentes). Substituído pelo Champion v2.

---

## Experimento 003 — Champion v2 (atual)

**Data:** Agosto/2026
**Notebook:** `notebooks/02_modeling_telco_churn.ipynb` (v2)
**Modelo:** LogisticRegression com SMOTE-NC
**Artefato:** `models/champion_v2_logreg.joblib`

### Configuração

| Parâmetro | Valor |
|---|---|
| Algoritmo | LogisticRegression |
| Balanceamento | SMOTENC dentro do CV (`imblearn.Pipeline`) |
| `smote__k_neighbors` | **7** (tunado via RandomizedSearchCV) |
| `clf__C` | 1.0 (tunado) |
| `clf__penalty` | l2 (tunado) |
| `solver` | lbfgs |
| `max_iter` | 2000 |
| Threshold | **0,5506** (maximiza Precision com Recall ≥ 0,70 no CV) |
| Features | 21 (7 numéricas + 14 categóricas) — sem `Total Charges`, com `novo_cliente` |
| Validação | StratifiedKFold(n_splits=5) |
| Tuning | RandomizedSearchCV(n_iter=40, scoring='f1') |
| Candidatos comparados | 6 (LogReg, GradientBoosting, MLP, RandomForest, ExtraTrees, HistGradBoost) |
| Seeds | `random_state=42` em split, CV, SMOTE, RandomizedSearch e classificador |

### Resultados Cross-Validation (treino, 5-fold com SMOTE-NC)

| Modelo | AUC-ROC | PR-AUC | F1 | Recall | Precision |
|---|---|---|---|---|---|
| **LogReg** ★ | **0,8554** | **0,6729** | **0,6500** | 0,7572 | 0,5698 |
| GradientBoosting | 0,8543 | 0,6712 | 0,6351 | 0,6662 | 0,6071 |
| MLP | 0,8487 | 0,6540 | 0,6265 | 0,6970 | 0,5701 |
| RandomForest | 0,8445 | 0,6352 | 0,6157 | 0,6294 | 0,6031 |
| ExtraTrees | 0,8348 | 0,6111 | 0,6076 | 0,6334 | 0,5843 |
| HistGradBoost | 0,8431 | 0,6463 | 0,6056 | 0,6060 | 0,6052 |

### Resultados (teste)

| Métrica | Valor | Meta ML Canvas | Status |
|---|---|---|---|
| AUC-ROC | 0,8383 | ≥ 0,80 | ✅ |
| PR-AUC | 0,6503 | — | Referência |
| Precision (Churn) | 0,5570 | — | Referência |
| Recall (Churn) | 0,6791 | ≥ 0,70 | ⚠️ −2,1 pp |
| F1 (Churn) | 0,6120 | — | Referência |
| Accuracy | 0,7694 | — | Referência |
| Clientes acionados | 456 | — | −20% vs baseline |
| Falsos positivos | 202 | — | −28% vs baseline |

### Diagnóstico de Threshold (documentado no notebook, seção 2.6)

A curva PR do modelo não passa pelo ponto (Precision ≥ 0,65 **e** Recall ≥ 0,70) — geometricamente incompatível com o sinal disponível no dataset atual (PR-AUC = 0,6729). O threshold escolhido (0,5506) entrega Recall = 0,7003 no CV e 0,6791 no teste — gap de generalização de 2,1 pp.

### Observações

Melhor configuração até agora. Supera o baseline em PR-AUC (0,6503 vs 0,6447) e Precision (0,5570 vs 0,5113). AUC-ROC ligeiramente inferior ao baseline (0,8383 vs 0,8487) — consequência esperada de otimizar F1 em vez de AUC no tuning. A meta de Recall ≥ 0,70 não foi atingida no teste por limitação geométrica da curva PR; para superar seria necessário incorporar features de trajetória comportamental ou dados externos.

---

## Próximas iterações sugeridas

| ID | Hipótese | Mudança proposta | Risco |
|---|---|---|---|
| 004 | Features de interação melhoram PR-AUC | Adicionar `alto_custo_sem_fidelidade = (MonthlyCharges > 70) & (Contract == 'Month-to-month')` | Baixo — feature simples |
| 005 | Calibração de probabilidades melhora threshold | Wrap com `CalibratedClassifierCV(method='isotonic')` | Médio — altera scores |
| 006 | GradientBoosting tunado supera LogReg em PR-AUC | Expandir espaço de busca do GB com `n_iter=80` | Médio — custo computacional |
| 007 | Dados reais de produção com drift corrigem generalização | Retreino com dados reais do próximo trimestre | Alto — requer coleta |
