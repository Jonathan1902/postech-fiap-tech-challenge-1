# Validation Report — Etapa 3

## Checklist

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | `make data` gera trusted (7043×18) e refined (7043×22) | ✅ | `trusted: (7043, 18) \| refined: (7043, 22)` |
| 2 | `make lint` → 0 findings | ✅ | `All checks passed!` |
| 3 | `make test-cov` → 39 testes verdes, cobertura 91% | ✅ | `39 passed, coverage 91.49%` |
| 4 | `/health` retorna `status=ok`, `threshold=0.6717` | ✅ | Teste `test_health_schema` |
| 5 | `/predict` com payload válido → 200, `churn_probability ∈ [0,1]` | ✅ | Teste `test_predict_valid_payload` |
| 6 | `/predict` com Contract inválido → 422 | ✅ | Teste `test_predict_invalid_contract` |
| 7 | `/predict` com Tenure negativo → 422 | ✅ | Teste `test_predict_negative_tenure` |
| 8 | Teste de regressão: perfil fixo (Tenure=1, Fibra, M2M) → prob > 0.5 | ✅ | Teste `test_regression_fixed_profile` |
| 9 | Página HTML servida em `/` | ✅ | `static/index.html` montado via StaticFiles |
| 10 | Log estruturado JSON com `request_id`, `latency_ms`, `input_hash` | ✅ | structlog configurado em middleware |
| 11 | Trusted não contém colunas de leakage | ✅ | Teste `test_trusted_no_leakage_columns` |
| 12 | Clientes com Tenure=0 têm Total Charges=0 na camada trusted | ✅ | Teste `test_trusted_tenure_zero_has_zero_total_charges` |

## Fixed Profile Regression Test

Profile used:
```json
{
  "Tenure Months": 1, "Monthly Charges": 99.65, "Total Charges": 99.65,
  "Internet Service": "Fiber optic", "Contract": "Month-to-month",
  "Payment Method": "Electronic check", "Paperless Billing": "Yes",
  "Senior Citizen": "No", "Partner": "No", "Dependents": "No",
  "Multiple Lines": "No", "Online Security": "No", "Online Backup": "No",
  "Device Protection": "No", "Tech Support": "No",
  "Streaming TV": "No", "Streaming Movies": "No"
}
```

Expected: `churn_probability > 0.5` (high-risk: new customer, fiber, month-to-month, electronic check).

## Dataset Verification

```
trusted: (7043, 18)  — 17 features + churn target
refined: (7043, 22)  — 21 features (5 derived, Total Charges dropped) + churn target
```

## Coverage Report

```
TOTAL  329 stmts  28 missed  91.49%  ≥ 85% required
```
