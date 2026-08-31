from churn_predictor.domain.schemas import CustomerProfile, PredictionResponse
from tests.conftest import VALID_PAYLOAD


def test_predict_label_consistent_with_threshold(predictor, sample_profile):
    result = predictor.predict(sample_profile)
    assert result.churn_prediction == (result.churn_probability >= predictor.threshold)


def test_predict_response_schema(predictor, sample_profile):
    result = predictor.predict(sample_profile)
    assert isinstance(result, PredictionResponse)
    assert len(result.top_contributors) > 0
    assert result.threshold == predictor.threshold


def test_regression_high_risk_profile(predictor):
    """Perfil de alto risco deve retornar probabilidade > 0.5."""
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
    result = predictor.predict(profile)
    assert result.churn_probability > 0.5
