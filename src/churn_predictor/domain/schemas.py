from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from churn_predictor.domain.enums import (
    ContractType,
    InternetService,
    PaymentMethod,
    SeniorCitizen,
    YesNo,
)

_FIELD_ALIASES: dict[str, str] = {
    "tenure_months": "Tenure Months",
    "monthly_charges": "Monthly Charges",
    "total_charges": "Total Charges",
    "senior_citizen": "Senior Citizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "multiple_lines": "Multiple Lines",
    "internet_service": "Internet Service",
    "online_security": "Online Security",
    "online_backup": "Online Backup",
    "device_protection": "Device Protection",
    "tech_support": "Tech Support",
    "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming Movies",
    "contract": "Contract",
    "paperless_billing": "Paperless Billing",
    "payment_method": "Payment Method",
}


class CustomerProfile(BaseModel):
    """Raw customer profile after basic cleaning (trusted schema)."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=lambda field_name: _FIELD_ALIASES.get(field_name, field_name),
    )

    customer_id: str | None = None

    # Numerics
    tenure_months: Annotated[int, Field(ge=0)]
    monthly_charges: Annotated[float, Field(gt=0)]
    total_charges: Annotated[float, Field(ge=0)]

    # Demographics
    senior_citizen: SeniorCitizen
    partner: YesNo
    dependents: YesNo

    # Services
    multiple_lines: YesNo
    internet_service: InternetService
    online_security: YesNo
    online_backup: YesNo
    device_protection: YesNo
    tech_support: YesNo
    streaming_tv: YesNo
    streaming_movies: YesNo

    # Billing
    contract: ContractType
    paperless_billing: YesNo
    payment_method: PaymentMethod

    @model_validator(mode="after")
    def check_total_charges_consistency(self) -> "CustomerProfile":
        if self.tenure_months == 0 and self.total_charges != 0:
            raise ValueError("Total Charges must be 0 when Tenure Months is 0")
        return self


class Contributor(BaseModel):
    feature: str
    feature_label: str
    contribution: float
    direction: str


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
