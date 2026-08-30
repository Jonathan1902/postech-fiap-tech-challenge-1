from __future__ import annotations

import numpy as np
import pandas as pd

from churn_predictor.domain.schemas import Contributor


class Explainer:
    """Computes feature contributions for LogisticRegression using coef × scaled_value."""

    def __init__(self, pipeline, num_features: list[str], cat_features: list[str]):
        self._pipeline = pipeline
        self._num_features = num_features
        self._cat_features = cat_features

    def explain(self, df: pd.DataFrame, top_n: int = 5) -> list[Contributor]:
        encode_step = self._pipeline.named_steps["encode"]
        clf_step = self._pipeline.named_steps["clf"]

        # Transform through impute only (no SMOTE in predict)
        imputed = self._pipeline.named_steps["impute"].transform(df)
        encoded = encode_step.transform(imputed)

        coefs = clf_step.coef_[0]
        contribs = coefs * encoded[0]

        # Build feature names
        names = self._build_feature_names(encode_step)

        top_idx = np.argsort(np.abs(contribs))[::-1][:top_n]
        return [
            Contributor(
                feature=names[i] if i < len(names) else f"feature_{i}",
                value=float(encoded[0][i]),
                contribution=float(contribs[i]),
            )
            for i in top_idx
        ]

    def _build_feature_names(self, encode_step) -> list[str]:
        names: list[str] = []
        for name, transformer, _ in encode_step.transformers_:
            if name == "num":
                if hasattr(transformer, "feature_names_in_"):
                    names.extend(transformer.feature_names_in_.tolist())
                else:
                    names.extend(self._num_features)
            elif name == "cat":
                if hasattr(transformer, "get_feature_names_out"):
                    names.extend(transformer.get_feature_names_out().tolist())
                else:
                    names.extend(self._cat_features)
            elif name == "remainder":
                pass
        return names
