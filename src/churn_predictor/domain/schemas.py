from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from churn_predictor.domain.enums import (
    ContractType,
    InternetService,
    PaymentMethod,
    SeniorCitizen,
    YesNo,
)


class CustomerProfile(BaseModel):
    """Raw customer profile after basic cleaning (trusted schema)."""

    customer_id: str | None = None

    # Numerics
    tenure_months: int = Field(..., ge=0, alias="Tenure Months")
    monthly_charges: float = Field(..., gt=0, alias="Monthly Charges")
    total_charges: float = Field(..., ge=0, alias="Total Charges")

    # Demographics
    senior_citizen: SeniorCitizen = Field(..., alias="Senior Citizen")
    partner: YesNo = Field(..., alias="Partner")
    dependents: YesNo = Field(..., alias="Dependents")

    # Services
    multiple_lines: YesNo = Field(..., alias="Multiple Lines")
    internet_service: InternetService = Field(..., alias="Internet Service")
    online_security: YesNo = Field(..., alias="Online Security")
    online_backup: YesNo = Field(..., alias="Online Backup")
    device_protection: YesNo = Field(..., alias="Device Protection")
    tech_support: YesNo = Field(..., alias="Tech Support")
    streaming_tv: YesNo = Field(..., alias="Streaming TV")
    streaming_movies: YesNo = Field(..., alias="Streaming Movies")

    # Billing
    contract: ContractType = Field(..., alias="Contract")
    paperless_billing: YesNo = Field(..., alias="Paperless Billing")
    payment_method: PaymentMethod = Field(..., alias="Payment Method")

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def check_total_charges_consistency(self) -> CustomerProfile:
        if self.tenure_months == 0 and self.total_charges != 0:
            raise ValueError("Total Charges must be 0 when Tenure Months is 0")
        return self


class Contributor(BaseModel):
    feature: str
    value: float
    contribution: float


class PredictionResponse(BaseModel):
    customer_id: str | None = None
    churn_probability: float
    churn_prediction: bool
    threshold: float
    model_version: str
    decision_reason: str
    top_contributors: list[Contributor]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    threshold: float
    feature_contract_hash: str
    uptime_s: float
