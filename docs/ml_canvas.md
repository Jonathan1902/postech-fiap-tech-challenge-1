# OwnML Machine Learning Canvas
**Projeto:** Tech Challenge 1 - Pipeline Preditivo de Churn
**Designed for:** Telco - Telecomunicações (Fictício)
**Designed by:** Jonathan Costa, FIAP - Pós-Graduação em Machine Learning
**Date:** Agosto/2026

---

## CONTEXTO

A Telco é uma operadora de telecomunicações que está perdendo múltiplos clientes de maneira acelerada. Sabemos que o Custo de Aquisição de Clientes (CAC) é elevado, pois leva em consideração o investimento em marketing, vendas e infraestrutura para atrair novos clientes. Por conta disso, a empresa está buscando alternativas que possam reduzir a taxa de churn, ou seja, o número de clientes que cancelam seus serviços. A retenção de clientes é uma prioridade estratégica solicitada pela diretoria, por conta disso o case propõe a construção de um modelo preditivo de churn que permita identificar clientes em risco de cancelamento antes que isso aconteça, possibilitando ações preventivas da equipe de retenção.

### PREDICTION TASK
- Classificação binária supervisionada
- Entidade: cliente ativo da Telco - Telecomunicações
- Outcomes: Churn (1) ou Não-Churn (0)
- Observação do outcome: ao fim do ciclo de contrato/mês

### DECISIONS
- Acionar equipe de retenção de clientes com a lista de clientes em risco de churn
- Oferecer promoções personalizadas (desconto, upgrade de plano) para retenção de clientes
- Priorizar contato por score de risco (alto/médio/baixo)

### VALUE PROPOSITION
- **Beneficiário:** equipe de retenção e diretoria comercial da Telco - Telecomunicações
- **Dor:** perda de receita recorrente por cancelamentos não previstos (pedido da diretoria)
- **Integração:** previsões consumidas via API REST pelos times internos
- **Meta:** reduzir taxa de churn anual (avaliar o valor de referência com baseline inicial, para definir meta de melhoria)

### DATA COLLECTION
- Fonte inicial: dataset Telco Customer Churn (Kaggle/IBM) — ~7.000 registros, ~20 features

### DATA SOURCES
- Fonte inicial: dataset Telco Customer Churn (Kaggle/IBM) — ~7.000 registros, ~20 features
    - Estruturar o dataset em treino, validação e teste.

### IMPACT SIMULATION
- Custo de falso negativo: cliente churna sem intervenção (perda de receita)
- Custo de falso positivo: ação de retenção desnecessária (custo operacional)
- Critério de deploy: AUC-ROC ≥ 0,80 e Recall (churn) ≥ 0,70
- Restrição de equidade: verificar viés por perfil demográfico (SeniorCitizen, gender)

### MAKING PREDICTIONS
- Modo: batch (predição mensal para toda a base ativa)
- Frequência: mensal, com possibilidade de on-demand via API
- Latência aceitável: até 24h para batch; < 500ms para API

### BUILDING MODELS
- 1 modelo campeão escolhido entre: Logistic Regression, Random Forest, MLPClassifier
- Validação: cross-validation estratificada (5 folds)
- Atualização: retreino trimestral ou quando AUC cair mais do que 5% em monitoramento
- Recurso computacional: ambiente local

### FEATURES
- Numéricas: tenure, MonthlyCharges, TotalCharges → StandardScaler
- Categóricas: Contract, PaymentMethod, InternetService, serviços → OneHotEncoder
- Pipeline de pré-processamento com Scikit-Learn ColumnTransformer

### MONITORING
- Métricas técnicas: AUC-ROC, Recall, F1 — revisão mensal
- Métricas de negócio: taxa de churn mensal, taxa de conversão de retenção
- Alerta: degradação de AUC > 5% aciona revisão do modelo
- Revisão completa: trimestral
