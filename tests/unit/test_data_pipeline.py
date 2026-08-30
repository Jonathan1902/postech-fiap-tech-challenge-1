import pytest

from churn_predictor.config import settings
from churn_predictor.data.pipeline import EXCLUDE_COLS, DataPipeline

RAW_PATH = settings.raw_data_path


@pytest.fixture(scope="module")
def pipeline():
    return DataPipeline()


@pytest.fixture(scope="module")
def trusted(pipeline):
    return pipeline.build_trusted(RAW_PATH)


@pytest.fixture(scope="module")
def refined(pipeline, trusted):
    return pipeline.build_refined(trusted)


def test_trusted_no_leakage_columns(trusted):
    for col in EXCLUDE_COLS:
        assert col not in trusted.columns, f"Leakage column {col!r} found in trusted"


def test_trusted_has_churn_column(trusted):
    assert "churn" in trusted.columns


def test_trusted_total_charges_no_nan(trusted):
    assert trusted["Total Charges"].isna().sum() == 0


def test_trusted_tenure_zero_has_zero_total_charges(trusted):
    mask = trusted["Tenure Months"] == 0
    assert (trusted.loc[mask, "Total Charges"] == 0).all()


def test_refined_has_derived_features(refined):
    derived = [
        "qtd_addons", "contrato_longo", "pagamento_automatico", "internet_fibra", "novo_cliente"
    ]
    for col in derived:
        assert col in refined.columns


def test_refined_no_total_charges(refined):
    assert "Total Charges" not in refined.columns


def test_refined_row_count(trusted, refined):
    assert len(refined) == len(trusted)
