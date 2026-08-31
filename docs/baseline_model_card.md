# Model Card — Baseline: Regressão Logística (Churn)

**Projeto:** Tech Challenge 1 — Pipeline Preditivo de Churn  
**Autor:** Jonathan Costa — FIAP Pós-Graduação em Machine Learning  
**Data:** Agosto/2026  
**Arquivo do modelo:** `models/baseline_logistic_regression.joblib`
**Revisão aplicada (v2):** ver [eda_and_model_revision.md](./eda_and_model_revision.md) — pipeline agora usa 17 features (removidos `Gender` e `Phone Service` por Cramér's V < 0,02), colapso de `"No internet service"`/`"No phone service"` → `"No"` (elimina 7 colunas OHE colineares) e imputação `Total Charges = 0` para clientes com `Tenure = 0`. Métricas mantiveram-se equivalentes (AUC 0,8487 · Recall 0,7834 · F1 0,6188), mas o modelo é mais parcimonioso e semanticamente correto.  

---

## 1. Propósito

Este modelo é o **baseline** do pipeline preditivo de churn da Telco. Ele estabelece o piso de performance que todo modelo subsequente precisa superar para justificar maior complexidade.

A tarefa é classificação binária supervisionada: dado o perfil de um cliente ativo, prever se ele vai cancelar o contrato (churn = 1) ou não (churn = 0) no próximo ciclo de faturamento.

---

## 2. Dataset

| Atributo | Valor |
|---|---|
| Fonte | Telco Customer Churn (Kaggle/IBM) |
| Total de registros | 7.043 |
| Features de entrada | 18 (2 numéricas + 16 categóricas) |
| Target | `Churn Value` (0 = Não Churn, 1 = Churn) |
| Distribuição do target | 73,5% Não Churn / 26,5% Churn (desbalanceado) |
| Split treino/teste | 80% / 20% estratificado por target |
| Treino | 5.634 amostras |
| Teste | 1.409 amostras |

---

## 3. Pipeline de Pré-processamento

O pré-processamento é encapsulado em um `ColumnTransformer` do scikit-learn, garantindo que **a mesma transformação** aplicada no treino seja replicada automaticamente no teste e em produção — eliminando risco de data leakage.

```
Pipeline
├── preprocessor (ColumnTransformer)
│   ├── num  → ['Tenure Months', 'Monthly Charges']
│   │   ├── SimpleImputer(strategy='median')
│   │   └── StandardScaler()
│   └── cat  → [16 features categóricas]
│       ├── SimpleImputer(strategy='most_frequent')
│       └── OneHotEncoder(handle_unknown='ignore', sparse_output=False)
└── classifier
    └── LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced',
                           solver='lbfgs', random_state=42)
```

> **Nota:** `Total Charges` é descartado pelo `FeatureEngineer` antes do treino e da inferência, por ser altamente colinear com `Tenure Months` (ρ = 0,826). O campo ainda é recebido no input da API para compatibilidade de schema, mas não alimenta o modelo.

### Colunas excluídas do treinamento

As colunas abaixo foram removidas antes da modelagem por serem identificadores, metadados geográficos ou conterem **leakage** do target:

| Coluna | Motivo |
|---|---|
| `CustomerID` | Identificador único — sem valor preditivo |
| `Count` | Constante (sempre 1) |
| `Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude` | Metadados geográficos sem relação causal |
| `Churn Label` | Representação textual do target — leakage direto |
| `Churn Score` | Score derivado do churn — leakage direto |
| `CLTV` | Valor de vida do cliente calculado pós-churn — leakage |
| `Churn Reason` | Motivo do cancelamento — existe somente após o evento |

### Padronização numérica (StandardScaler)

O `StandardScaler` transforma cada feature numérica para média zero e desvio padrão unitário:

$$z = \frac{x - \mu}{\sigma}$$

Parâmetros aprendidos no treino (aplicados sem refit no teste):

| Feature | Média (μ) | Desvio Padrão (σ) |
|---|---|---|
| Tenure Months | ~32,4 meses | ~24,6 meses |
| Monthly Charges | ~64,8 USD | ~30,1 USD |
| Total Charges | ~2.283 USD | ~2.267 USD |

> **Por que StandardScaler?** A Regressão Logística usa gradiente descendente e é sensível à escala das features. Sem normalização, features com magnitudes maiores (ex.: `Total Charges` em milhares) dominam os coeficientes e prejudicam a convergência.

### Encoding categórico (OneHotEncoder)

Cada categoria única é transformada em uma coluna binária (0/1). O parâmetro `handle_unknown='ignore'` garante que categorias não vistas no treino (dados de produção futuros) resultem em vetor de zeros em vez de erro.

Dimensão final após encoding: **3 numéricas + ~45 binárias = ~48 features totais**.

### Tratamento do desbalanceamento (class_weight='balanced')

Com 26,5% de churners, o modelo sem ajuste tenderia a prever sempre "Não Churn". O parâmetro `class_weight='balanced'` repondera automaticamente as amostras:

$$w_c = \frac{n\_amostras}{n\_classes \times n\_amostras\_da\_classe\_c}$$

Isso equivale a dar ~2,77x mais peso aos exemplos de churn no cálculo da função de perda.

---

## 4. Resultados no Conjunto de Teste

### Métricas Principais

| Métrica | Valor | Critério de Aprovação | Status |
|---|---|---|---|
| **AUC-ROC** | **0,8488** | ≥ 0,80 | ✅ Aprovado |
| **Recall (Churn)** | **0,7807** | ≥ 0,70 | ✅ Aprovado |
| **F1-Score (Churn)** | **0,6173** | — | Referência |

### Relatório Completo por Classe

| Classe | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Não Churn (0) | 0,9020 | 0,7295 | 0,8066 | 1.035 |
| **Churn (1)** | **0,5105** | **0,7807** | **0,6173** | **374** |
| **Macro avg** | 0,7063 | 0,7551 | 0,7120 | 1.409 |
| **Weighted avg** | 0,8018 | 0,7452 | 0,7614 | 1.409 |
| **Accuracy** | — | — | 0,7452 | 1.409 |

### Matriz de Confusão

```
                 Previsto: Não Churn   Previsto: Churn
Real: Não Churn       755  (TN)             280  (FP)
Real: Churn            82  (FN)             292  (TP)
```

| Indicador | Valor | Interpretação |
|---|---|---|
| True Positives (TP) | 292 | Churners corretamente identificados |
| True Negatives (TN) | 755 | Não-churners corretamente identificados |
| False Positives (FP) | 280 | Clientes abordados desnecessariamente |
| False Negatives (FN) | 82 | Churners que escaparam sem intervenção |

> **Leitura de negócio:** O modelo captura 78% dos clientes que realmente vão cancelar. A cada 100 churners reais, apenas 22 passam sem ser detectados. O custo de falso positivo (280 ações desnecessárias por ciclo) é aceitável dado o custo maior de perder receita recorrente.

---

## 5. Interpretabilidade dos Coeficientes

Os coeficientes da Regressão Logística indicam a contribuição relativa de cada feature na probabilidade de churn (após padronização). Valores **positivos** aumentam a probabilidade de churn; **negativos** a reduzem.

### Top 15 Features por Magnitude

| Coeficiente | Feature | Direção |
|---|---|---|
| −1,1930 | Tenure Months | Mais tempo de contrato → menos churn |
| −0,9307 | Dependents: Sim | Clientes com dependentes → muito menos churn |
| −0,7532 | Contract: Two year | Contrato longo → protetor forte |
| +0,7036 | Dependents: Não | Sem dependentes → risco elevado |
| +0,6597 | Contract: Month-to-month | Mensalidade → risco elevado |
| +0,5922 | Internet Service: Fiber optic | Fibra óptica → risco elevado |
| −0,5626 | Internet Service: DSL | DSL → protetor moderado |
| −0,5581 | Monthly Charges | Cobrança mensal (escalonada) → relação inversa |
| +0,4974 | Total Charges | Cobrança total (colinear com tenure) |
| −0,2739 | Paperless Billing: Não | Boleto físico → menos churn |

> **Nota sobre multicolinearidade:** `Tenure Months` e `Total Charges` têm correlação de 0,826. Em modelos lineares, isso distribui o peso entre ambos de forma instável. O coeficiente isolado de cada um não deve ser interpretado de forma independente — o efeito combinado é que importa.

---

## 6. Limitações do Baseline

| Limitação | Impacto | Próximo passo |
|---|---|---|
| Threshold fixo em 0,5 | Recall pode ser melhorado ajustando o threshold de classificação | Calibrar threshold na curva Precision-Recall |
| Multicolinearidade Tenure × Total Charges | Instabilidade dos coeficientes | Remover `Total Charges` ou aplicar regularização L1 |
| Sem engenharia de features | Features agregadas (qtd_addons, contrato_longo) podem melhorar performance | Testar na próxima iteração |
| Linearidade assumida | Relações não-lineares não capturadas | Avaliar Random Forest e Gradient Boosting |

---

## 7. Reprodutibilidade

```python
import joblib

pipeline = joblib.load('models/baseline_logistic_regression.joblib')

# Uso em produção (X deve conter as mesmas 19 colunas usadas no treino)
y_pred = pipeline.predict(X)
y_proba = pipeline.predict_proba(X)[:, 1]  # probabilidade de churn
```

| Parâmetro de reprodutibilidade | Valor |
|---|---|
| `random_state` (LogisticRegression) | 42 |
| `random_state` (train_test_split) | 42 |
| `test_size` | 0,20 |
| `stratify` | `Churn Value` |
| Versão scikit-learn | 1.9.0 |

---

## 8. Critérios de Deploy e Monitoramento

O modelo atende aos critérios definidos no ML Canvas para produção:

- ✅ AUC-ROC ≥ 0,80 → **0,8488**
- ✅ Recall (churn) ≥ 0,70 → **0,7807**

**Alertas de degradação:** caso o AUC-ROC em produção caia mais de 5 pontos percentuais (< 0,80), deve ser acionado retreino com dados atualizados.

**Frequência de retreino recomendada:** trimestral, ou sob demanda após alerta de degradação.
