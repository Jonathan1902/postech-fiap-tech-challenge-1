from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn_predictor.features.engineering import ADDON_COLS, FeatureEngineer

EXCLUDE_COLS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Label",
    "Churn Score",
    "CLTV",
    "Churn Reason",
]

DROP_LOW_ASSOC = ["Gender", "Phone Service"]
SHEET_NAME = "Telco_Churn"
TARGET_COL = "Churn Value"


class DataPipeline:
    def __init__(self, engineer: FeatureEngineer | None = None):
        self._engineer = engineer or FeatureEngineer()

    def build_trusted(self, src_xlsx: Path) -> pd.DataFrame:
        df = pd.read_excel(src_xlsx, sheet_name=SHEET_NAME)
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

        mask_new = (df["Tenure Months"] == 0) & (df["Total Charges"].isna())
        df.loc[mask_new, "Total Charges"] = 0.0

        # Fill any remaining NaN (non-numeric strings at non-zero tenure) with column median
        df["Total Charges"] = df["Total Charges"].fillna(df["Total Charges"].median())

        for c in ADDON_COLS:
            df[c] = df[c].replace("No internet service", "No")
        df["Multiple Lines"] = df["Multiple Lines"].replace("No phone service", "No")

        df = df.drop(columns=[c for c in DROP_LOW_ASSOC if c in df.columns])
        df = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns])

        df = df.rename(columns={TARGET_COL: "churn"})

        return df

    def build_refined(self, trusted_df: pd.DataFrame) -> pd.DataFrame:
        non_feature = {"churn"}
        feature_cols = [c for c in trusted_df.columns if c not in non_feature]
        features = self._engineer.transform(trusted_df[feature_cols])
        features["churn"] = trusted_df["churn"].values
        return features

    def run(self, src_xlsx: Path, trusted_path: Path, refined_path: Path) -> None:
        trusted_path.parent.mkdir(parents=True, exist_ok=True)
        refined_path.parent.mkdir(parents=True, exist_ok=True)

        trusted = self.build_trusted(src_xlsx)
        trusted.to_csv(trusted_path, index=False)

        refined = self.build_refined(trusted)
        refined.to_csv(refined_path, index=False)
