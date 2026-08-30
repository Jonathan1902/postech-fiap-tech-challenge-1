from __future__ import annotations

import pandas as pd

from churn_predictor.domain.schemas import CustomerProfile

ADDON_COLS = [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]

AUTO_PAYMENTS = {"Bank transfer (automatic)", "Credit card (automatic)"}

EXPECTED_COLUMNS = [
    "Tenure Months",
    "Monthly Charges",
    "qtd_addons",
    "contrato_longo",
    "pagamento_automatico",
    "internet_fibra",
    "novo_cliente",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]


class FeatureEngineer:
    """Applies business feature engineering: trusted DataFrame → model-ready DataFrame."""

    def transform(self, data: pd.DataFrame | CustomerProfile) -> pd.DataFrame:
        if isinstance(data, CustomerProfile):
            df = self._profile_to_df(data)
        else:
            df = data.copy()

        df["qtd_addons"] = sum((df[c] == "Yes").astype(int) for c in ADDON_COLS)
        df["contrato_longo"] = (df["Contract"] != "Month-to-month").astype(int)
        df["pagamento_automatico"] = df["Payment Method"].isin(AUTO_PAYMENTS).astype(int)
        df["internet_fibra"] = (df["Internet Service"] == "Fiber optic").astype(int)
        df["novo_cliente"] = (df["Tenure Months"] <= 3).astype(int)

        df = df.drop(columns=["Total Charges"], errors="ignore")

        return df[EXPECTED_COLUMNS]

    @staticmethod
    def _profile_to_df(profile: CustomerProfile) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Tenure Months": profile.tenure_months,
                    "Monthly Charges": profile.monthly_charges,
                    "Total Charges": profile.total_charges,
                    "Senior Citizen": profile.senior_citizen.value,
                    "Partner": profile.partner.value,
                    "Dependents": profile.dependents.value,
                    "Multiple Lines": profile.multiple_lines.value,
                    "Internet Service": profile.internet_service.value,
                    "Online Security": profile.online_security.value,
                    "Online Backup": profile.online_backup.value,
                    "Device Protection": profile.device_protection.value,
                    "Tech Support": profile.tech_support.value,
                    "Streaming TV": profile.streaming_tv.value,
                    "Streaming Movies": profile.streaming_movies.value,
                    "Contract": profile.contract.value,
                    "Paperless Billing": profile.paperless_billing.value,
                    "Payment Method": profile.payment_method.value,
                }
            ]
        )
