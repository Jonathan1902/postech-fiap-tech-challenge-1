from tests.conftest import VALID_PAYLOAD


def test_predict_valid_payload(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["churn_prediction"], bool)
    assert len(data["top_contributors"]) > 0


def test_predict_invalid_contract(client):
    resp = client.post("/predict", json={**VALID_PAYLOAD, "Contract": "Weekly"})
    assert resp.status_code == 422


def test_predict_negative_tenure(client):
    resp = client.post("/predict", json={**VALID_PAYLOAD, "Tenure Months": -1})
    assert resp.status_code == 422


def test_predict_invalid_internet_service(client):
    resp = client.post("/predict", json={**VALID_PAYLOAD, "Internet Service": "Satellite"})
    assert resp.status_code == 422


def test_predict_tenure_zero_nonzero_total_charges(client):
    resp = client.post("/predict", json={**VALID_PAYLOAD, "Tenure Months": 0, "Total Charges": 100.0})
    assert resp.status_code == 422
