import pytest

from churn_predictor.domain.schemas import Contributor
from churn_predictor.features.engineering import FeatureEngineer
from churn_predictor.models.explainer import Explainer
from tests.conftest import VALID_PAYLOAD
import pandas as pd


@pytest.fixture
def explainer(predictor):
    return Explainer(predictor._pipeline, predictor._num_features, predictor._cat_features)


@pytest.fixture
def engineered_df():
    return FeatureEngineer().transform(pd.DataFrame([VALID_PAYLOAD]))


def test_explain_returns_list_of_contributors(explainer, engineered_df):
    result = explainer.explain(engineered_df)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(c, Contributor) for c in result)


def test_explain_top_n_respected(explainer, engineered_df):
    for n in [1, 3, 5]:
        assert len(explainer.explain(engineered_df, top_n=n)) == n


def test_explain_direction_values(explainer, engineered_df):
    valid = {"aumenta risco de churn", "reduz risco de churn"}
    for c in explainer.explain(engineered_df):
        assert c.direction in valid


def test_explain_feature_labels_non_empty(explainer, engineered_df):
    for c in explainer.explain(engineered_df):
        assert c.feature_label, f"Empty label for feature: {c.feature}"


def test_explain_high_risk_profile_has_positive_contributors(predictor):
    """Perfil de alto risco deve ter ao menos um fator de aumento de risco no top-5."""
    from churn_predictor.domain.schemas import CustomerProfile
    from tests.conftest import VALID_PAYLOAD

    profile = CustomerProfile.model_validate({
        **VALID_PAYLOAD,
        "Tenure Months": 1,
        "Monthly Charges": 99.65,
        "Total Charges": 99.65,
        "Partner": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "No",
        "Tech Support": "No",
    })
    df = FeatureEngineer().transform(profile)
    explainer = Explainer(predictor._pipeline, predictor._num_features, predictor._cat_features)
    result = explainer.explain(df, top_n=5)
    assert any(c.direction == "aumenta risco de churn" for c in result)
