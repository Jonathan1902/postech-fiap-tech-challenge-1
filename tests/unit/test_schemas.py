import pytest
from pydantic import ValidationError

from churn_predictor.domain.schemas import CustomerProfile
from tests.conftest import VALID_PAYLOAD


def test_valid_profile():
    p = CustomerProfile.model_validate(VALID_PAYLOAD)
    assert p.tenure_months == 12


def test_negative_tenure_rejected():
    with pytest.raises(ValidationError):
        CustomerProfile.model_validate({**VALID_PAYLOAD, "Tenure Months": -1})


def test_invalid_contract_rejected():
    with pytest.raises(ValidationError):
        CustomerProfile.model_validate({**VALID_PAYLOAD, "Contract": "Weekly"})


def test_invalid_internet_service_rejected():
    with pytest.raises(ValidationError):
        CustomerProfile.model_validate({**VALID_PAYLOAD, "Internet Service": "Satellite"})


def test_total_charges_must_be_zero_when_tenure_zero():
    with pytest.raises(ValidationError):
        CustomerProfile.model_validate({**VALID_PAYLOAD, "Tenure Months": 0, "Total Charges": 100.0})


def test_total_charges_zero_when_tenure_zero_valid():
    p = CustomerProfile.model_validate({**VALID_PAYLOAD, "Tenure Months": 0, "Total Charges": 0.0})
    assert p.tenure_months == 0
    assert p.total_charges == 0.0


def test_internet_service_cable_accepted():
    p = CustomerProfile.model_validate({**VALID_PAYLOAD, "Internet Service": "Cable"})
    assert p.internet_service.value == "Cable"
