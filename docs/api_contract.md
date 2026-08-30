# API Contract — Churn Predictor

## Base URL
`http://localhost:8000`

---

## GET /health

**Response 200**
```json
{
  "status": "ok",
  "model_version": "a1b2c3d4",
  "threshold": 0.6717,
  "feature_contract_hash": "e5f6a7b8",
  "uptime_s": 12.3
}
```

---

## POST /predict

**Request body** (`application/json`):
```json
{
  "Tenure Months": 12,
  "Monthly Charges": 65.0,
  "Total Charges": 780.0,
  "Senior Citizen": "No",
  "Partner": "Yes",
  "Dependents": "No",
  "Multiple Lines": "No",
  "Internet Service": "DSL",
  "Online Security": "Yes",
  "Online Backup": "No",
  "Device Protection": "No",
  "Tech Support": "No",
  "Streaming TV": "No",
  "Streaming Movies": "No",
  "Contract": "Month-to-month",
  "Paperless Billing": "Yes",
  "Payment Method": "Electronic check"
}
```

**Response 200**
```json
{
  "customer_id": null,
  "churn_probability": 0.423,
  "churn_prediction": false,
  "threshold": 0.6717,
  "model_version": "a1b2c3d4",
  "decision_reason": "Baixa probabilidade de churn (42.3%). Cliente provavelmente retido.",
  "top_contributors": [
    {"feature": "cat__Contract_Month-to-month", "value": 1.0, "contribution": 0.812},
    {"feature": "cat__Internet Service_DSL", "value": 1.0, "contribution": -0.531}
  ]
}
```

**Error 422** — invalid field values or constraint violations:
```json
{"detail": [{"loc": ["body", "Contract"], "msg": "...", "type": "..."}]}
```

### Field Enums

| Field | Allowed values |
|---|---|
| Senior Citizen | `"Yes"`, `"No"` |
| Partner | `"Yes"`, `"No"` |
| Dependents | `"Yes"`, `"No"` |
| Multiple Lines | `"Yes"`, `"No"` |
| Internet Service | `"DSL"`, `"Fiber optic"`, `"No"` |
| Online Security | `"Yes"`, `"No"` |
| Online Backup | `"Yes"`, `"No"` |
| Device Protection | `"Yes"`, `"No"` |
| Tech Support | `"Yes"`, `"No"` |
| Streaming TV | `"Yes"`, `"No"` |
| Streaming Movies | `"Yes"`, `"No"` |
| Contract | `"Month-to-month"`, `"One year"`, `"Two year"` |
| Paperless Billing | `"Yes"`, `"No"` |
| Payment Method | `"Electronic check"`, `"Mailed check"`, `"Bank transfer (automatic)"`, `"Credit card (automatic)"` |

### Constraints
- `Tenure Months` ≥ 0
- `Monthly Charges` > 0
- `Total Charges` ≥ 0
- `Total Charges` must be 0 when `Tenure Months` is 0

---

## POST /predict/batch

Same payload as `/predict` but as a JSON array. Returns array of `PredictionResponse`.

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H 'Content-Type: application/json' \
  -d '[{...profile1...}, {...profile2...}]'
```

---

## curl examples

```bash
# Health
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "Tenure Months": 1,
    "Monthly Charges": 99.65,
    "Total Charges": 99.65,
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
