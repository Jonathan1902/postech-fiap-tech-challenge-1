VALID_PAYLOAD = {
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
    "Payment Method": "Electronic check",
}


def test_predict_valid_payload(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["churn_prediction"], bool)
    assert len(data["top_contributors"]) > 0


def test_predict_invalid_contract(client):
    payload = {**VALID_PAYLOAD, "Contract": "Weekly"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_negative_tenure(client):
    payload = {**VALID_PAYLOAD, "Tenure Months": -1}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_invalid_internet_service(client):
    payload = {**VALID_PAYLOAD, "Internet Service": "Satellite"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_tenure_zero_nonzero_total_charges(client):
    payload = {**VALID_PAYLOAD, "Tenure Months": 0, "Total Charges": 100.0}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post("/predict/batch", json=[VALID_PAYLOAD, VALID_PAYLOAD])
    assert resp.status_code == 200
    assert len(resp.json()) == 2
