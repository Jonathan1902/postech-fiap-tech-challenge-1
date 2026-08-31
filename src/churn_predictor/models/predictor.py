from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import joblib

from churn_predictor.domain.schemas import CustomerProfile, PredictionResponse
from churn_predictor.features.engineering import FeatureEngineer
from churn_predictor.models.explainer import Explainer


class ChurnPredictor:
    def __init__(self, model_path: Path | str):
        model_path = Path(model_path)
        artifact: dict[str, Any] = joblib.load(model_path)

        self._pipeline = artifact["pipeline"]
        self._threshold: float = artifact["threshold"]
        self._feature_cols: list[str] = artifact["feature_cols"]
        self._num_features: list[str] = artifact["num_features"]
        self._cat_features: list[str] = artifact["cat_features"]

        raw = model_path.read_bytes()
        self._model_version = hashlib.sha256(raw).hexdigest()[:8]
        self._feature_contract_hash = hashlib.sha256(
            str(self._feature_cols).encode()
        ).hexdigest()[:8]

        self._engineer = FeatureEngineer()
        self._explainer = Explainer(self._pipeline, self._num_features, self._cat_features)

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def feature_cols(self) -> list[str]:
        return self._feature_cols

    @property
    def feature_contract_hash(self) -> str:
        return self._feature_contract_hash

    def metadata(self) -> dict[str, Any]:
        return {
            "model_version": self._model_version,
            "threshold": self._threshold,
            "feature_cols": self._feature_cols,
        }

    def predict_proba(self, profile: CustomerProfile) -> float:
        df = self._engineer.transform(profile)
        proba = self._pipeline.predict_proba(df)[0, 1]
        return float(proba)

    def predict(self, profile: CustomerProfile) -> PredictionResponse:
        df = self._engineer.transform(profile)
        prob = float(self._pipeline.predict_proba(df)[0, 1])
        label = prob >= self._threshold

        contributors = self._explainer.explain(df)

        top_label = contributors[0].feature_label if contributors else "n/a"
        if label:
            decision = f"Alta probabilidade de churn ({prob:.1%}). Principal fator: {top_label}."
        else:
            decision = f"Baixa probabilidade de churn ({prob:.1%}). Principal fator de retenção: {top_label}."

        return PredictionResponse(
            customer_id=profile.customer_id,
            churn_probability=prob,
            churn_prediction=label,
            threshold=self._threshold,
            model_version=self._model_version,
            decision_reason=decision,
            top_contributors=contributors,
        )

