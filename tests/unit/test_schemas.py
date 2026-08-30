import pytest
from pydantic import ValidationError

from churn_predictor.domain.schemas import CustomerProfile


def _valid_payload(**overrides):
    base = {
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
    base.update(overrides)
    return base


def test_valid_profile():
    p = CustomerProfile.model_validate(_valid_payload())
    assert p.tenure_months == 12


def test_negative_tenure_rejected():
    with pytest.raises(ValidationError):
        CustomerProfile.model_validate(_valid_payload(**{"Tenure Months": -1}))


def test_invalid_contract_rejected():
    with pytest.raises(ValidationError):
        CustomerProfile.model_validate(_valid_payload(**{"Contract": "Weekly"}))


def test_invalid_internet_service_rejected():
    with pytest.raises(ValidationError):
        CustomerProfile.model_validate(_valid_payload(**{"Internet Service": "Satellite"}))


def test_total_charges_must_be_zero_when_tenure_zero():
    with pytest.raises(ValidationError):
        payload = _valid_payload(**{"Tenure Months": 0, "Total Charges": 100.0})
        CustomerProfile.model_validate(payload)


def test_total_charges_zero_when_tenure_zero_valid():
    p = CustomerProfile.model_validate(_valid_payload(**{"Tenure Months": 0, "Total Charges": 0.0}))
    assert p.tenure_months == 0
    assert p.total_charges == 0.0
