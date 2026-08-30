from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    model_path: Path = Path("models/champion_v2_logreg.joblib")
    log_level: str = "INFO"
    prometheus_enabled: bool = False

    raw_data_path: Path = Path("data/raw/telco_customer_churn.xlsx")
    trusted_data_path: Path = Path("data/trusted/telco_churn_trusted.csv")
    refined_data_path: Path = Path("data/refined/telco_churn_refined.csv")


settings = Settings()
