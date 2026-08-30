import pytest
from fastapi.testclient import TestClient

from churn_predictor.api.app import create_app
from churn_predictor.api.dependencies import get_predictor
from churn_predictor.domain.schemas import CustomerProfile
from churn_predictor.models.predictor import ChurnPredictor
from churn_predictor.utils.sample_generator import RandomCustomerGenerator


@pytest.fixture(scope="session")
def predictor() -> ChurnPredictor:
    return get_predictor()


@pytest.fixture(scope="session")
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_profile() -> CustomerProfile:
    return CustomerProfile.model_validate(
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
            "Payment Method": "Electronic check",
        }
    )


@pytest.fixture
def generator() -> RandomCustomerGenerator:
    return RandomCustomerGenerator(seed=42)
