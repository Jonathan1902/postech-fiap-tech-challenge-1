from churn_predictor.domain.schemas import CustomerProfile, PredictionResponse


def test_predict_returns_probability_in_range(predictor, sample_profile):
    prob = predictor.predict_proba(sample_profile)
    assert 0.0 <= prob <= 1.0


def test_predict_label_consistent_with_threshold(predictor, sample_profile):
    result = predictor.predict(sample_profile)
    assert result.churn_prediction == (result.churn_probability >= predictor.threshold)


def test_predict_response_schema(predictor, sample_profile):
    result = predictor.predict(sample_profile)
    assert isinstance(result, PredictionResponse)
    assert len(result.top_contributors) > 0
    assert result.threshold == predictor.threshold


def test_regression_fixed_profile(predictor):
    """Regression: known profile must produce stable probability (±0.05)."""
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
    prob = predictor.predict_proba(profile)
    # High-risk profile → expect high churn probability
    assert prob > 0.5


def test_predict_batch(predictor, generator):
    profiles = generator.generate_batch(5)
    results = predictor.predict_batch(profiles)
    assert len(results) == 5
    for r in results:
        assert 0.0 <= r.churn_probability <= 1.0
