import pandas as pd
import pytest

from churn_predictor.domain.schemas import Contributor
from churn_predictor.models.explainer import Explainer

_SAMPLE_DF = pd.DataFrame(
    [
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
    ]
)


@pytest.fixture
def explainer(predictor):
    return Explainer(
        predictor._pipeline,
        predictor._num_features,
        predictor._cat_features,
    )


@pytest.fixture
def engineered_df(predictor):
    from churn_predictor.features.engineering import FeatureEngineer
    return FeatureEngineer().transform(_SAMPLE_DF)


def test_explain_returns_list_of_contributors(explainer, engineered_df):
    result = explainer.explain(engineered_df)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(c, Contributor) for c in result)


def test_explain_top_n_respected(explainer, engineered_df):
    for n in [1, 3, 5]:
        result = explainer.explain(engineered_df, top_n=n)
        assert len(result) == n


def test_explain_contributions_are_floats(explainer, engineered_df):
    result = explainer.explain(engineered_df)
    for c in result:
        assert isinstance(c.contribution, float)


def test_explain_direction_values(explainer, engineered_df):
    valid = {"aumenta risco de churn", "reduz risco de churn"}
    result = explainer.explain(engineered_df)
    for c in result:
        assert c.direction in valid


def test_explain_feature_labels_non_empty(explainer, engineered_df):
    result = explainer.explain(engineered_df)
    for c in result:
        assert c.feature_label, f"Empty label for feature: {c.feature}"


def test_explain_high_risk_profile_has_positive_contributors(predictor):
    """High-risk profile (fibra, mensalista, novo cliente) deve ter ao menos um fator de risco no top-5."""
    from churn_predictor.domain.schemas import CustomerProfile
    from churn_predictor.features.engineering import FeatureEngineer

    profile = CustomerProfile.model_validate(
        {
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
            "Payment Method": "Electronic check",
        }
    )
    df = FeatureEngineer().transform(profile)
    explainer = Explainer(predictor._pipeline, predictor._num_features, predictor._cat_features)
    result = explainer.explain(df, top_n=5)

    # Perfil de alto risco: ao menos um dos top-5 contribuidores deve aumentar o risco
    increasing = [c for c in result if c.direction == "aumenta risco de churn"]
    assert len(increasing) > 0, "Nenhum fator de risco encontrado no top-5 para perfil de alto risco"
