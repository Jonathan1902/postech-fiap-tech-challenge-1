import pandas as pd
import pytest

from churn_predictor.features.engineering import EXPECTED_COLUMNS, FeatureEngineer


@pytest.fixture
def engineer():
    return FeatureEngineer()


def _profile_df(**overrides):
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
        "Online Backup": "Yes",
        "Device Protection": "No",
        "Tech Support": "Yes",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_output_columns(engineer):
    df = engineer.transform(_profile_df())
    assert list(df.columns) == EXPECTED_COLUMNS


def test_qtd_addons_counts_yes(engineer):
    # Online Security=Yes, Online Backup=Yes, Tech Support=Yes → 3
    df = engineer.transform(_profile_df())
    assert df["qtd_addons"].iloc[0] == 3


def test_qtd_addons_zero(engineer):
    df = engineer.transform(
        _profile_df(
            **{
                "Online Security": "No",
                "Online Backup": "No",
                "Device Protection": "No",
                "Tech Support": "No",
                "Streaming TV": "No",
                "Streaming Movies": "No",
            }
        )
    )
    assert df["qtd_addons"].iloc[0] == 0


def test_contrato_longo_month_to_month_is_zero(engineer):
    df = engineer.transform(_profile_df(**{"Contract": "Month-to-month"}))
    assert df["contrato_longo"].iloc[0] == 0


def test_contrato_longo_two_year_is_one(engineer):
    df = engineer.transform(_profile_df(**{"Contract": "Two year"}))
    assert df["contrato_longo"].iloc[0] == 1


def test_pagamento_automatico_electronic_check_is_zero(engineer):
    df = engineer.transform(_profile_df(**{"Payment Method": "Electronic check"}))
    assert df["pagamento_automatico"].iloc[0] == 0


def test_pagamento_automatico_bank_transfer_is_one(engineer):
    df = engineer.transform(_profile_df(**{"Payment Method": "Bank transfer (automatic)"}))
    assert df["pagamento_automatico"].iloc[0] == 1


def test_internet_fibra(engineer):
    df = engineer.transform(_profile_df(**{"Internet Service": "Fiber optic"}))
    assert df["internet_fibra"].iloc[0] == 1
    df2 = engineer.transform(_profile_df(**{"Internet Service": "DSL"}))
    assert df2["internet_fibra"].iloc[0] == 0


def test_novo_cliente_tenure_le_3(engineer):
    df = engineer.transform(_profile_df(**{"Tenure Months": 3, "Total Charges": 195.0}))
    assert df["novo_cliente"].iloc[0] == 1
    df2 = engineer.transform(_profile_df(**{"Tenure Months": 10}))
    assert df2["novo_cliente"].iloc[0] == 0


def test_total_charges_dropped(engineer):
    df = engineer.transform(_profile_df())
    assert "Total Charges" not in df.columns
