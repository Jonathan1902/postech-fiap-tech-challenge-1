from __future__ import annotations

import numpy as np
import pandas as pd

from churn_predictor.domain.schemas import Contributor

# Human-readable labels for every feature the pipeline can produce.
# Numeric/engineered features use their raw name as key;
# OHE categorical features use the "ColumnName_Value" format produced by
# OHE.get_feature_names_out(column_names).
_LABEL_MAP: dict[str, str] = {
    # Numeric / engineered
    "Tenure Months": "Tempo de contrato (meses)",
    "Monthly Charges": "Cobrança mensal",
    "qtd_addons": "Quantidade de serviços adicionais",
    "contrato_longo": "Contrato de longa duração",
    "pagamento_automatico": "Pagamento automático",
    "internet_fibra": "Internet por fibra óptica",
    "novo_cliente": "Cliente novo (≤ 3 meses)",
    # Categorical — OHE produces "ColName_Value"
    "Senior Citizen_Yes": "Cidadão sênior: sim",
    "Senior Citizen_No": "Cidadão sênior: não",
    "Partner_Yes": "Possui cônjuge/parceiro",
    "Partner_No": "Sem cônjuge/parceiro",
    "Dependents_Yes": "Possui dependentes",
    "Dependents_No": "Sem dependentes",
    "Multiple Lines_Yes": "Múltiplas linhas telefônicas",
    "Multiple Lines_No": "Linha telefônica única",
    "Internet Service_DSL": "Internet: DSL",
    "Internet Service_Fiber optic": "Internet: fibra óptica",
    "Internet Service_Cable": "Internet: cabo",
    "Internet Service_No": "Sem serviço de internet",
    "Online Security_Yes": "Segurança online: ativa",
    "Online Security_No": "Segurança online: inativa",
    "Online Backup_Yes": "Backup online: ativo",
    "Online Backup_No": "Backup online: inativo",
    "Device Protection_Yes": "Proteção de dispositivo: ativa",
    "Device Protection_No": "Proteção de dispositivo: inativa",
    "Tech Support_Yes": "Suporte técnico: ativo",
    "Tech Support_No": "Suporte técnico: inativo",
    "Streaming TV_Yes": "Streaming de TV: ativo",
    "Streaming TV_No": "Streaming de TV: inativo",
    "Streaming Movies_Yes": "Streaming de filmes: ativo",
    "Streaming Movies_No": "Streaming de filmes: inativo",
    "Contract_Month-to-month": "Tipo de contrato: mensal",
    "Contract_One year": "Tipo de contrato: anual",
    "Contract_Two year": "Tipo de contrato: bienal",
    "Paperless Billing_Yes": "Fatura digital: sim",
    "Paperless Billing_No": "Fatura digital: não",
    "Payment Method_Electronic check": "Pagamento: cheque eletrônico",
    "Payment Method_Mailed check": "Pagamento: cheque enviado",
    "Payment Method_Bank transfer (automatic)": "Pagamento: transferência automática",
    "Payment Method_Credit card (automatic)": "Pagamento: cartão de crédito automático",
}


class Explainer:
    """Computes per-feature contributions for the loaded production pipeline.

    Supports two strategies depending on the classifier type:
    - Linear models (coef_ available): contribution = coef × encoded_value
    - Tree/ensemble models (feature_importances_ available): contribution = importance score
    """

    def __init__(self, pipeline, num_features: list[str], cat_features: list[str]):
        self._pipeline = pipeline
        self._num_features = num_features
        self._cat_features = cat_features

    def explain(self, df: pd.DataFrame, top_n: int = 5) -> list[Contributor]:
        encode_step = self._pipeline.named_steps["encode"]
        clf_step = self._pipeline.named_steps["clf"]

        imputed = self._pipeline.named_steps["impute"].transform(df)
        encoded = encode_step.transform(imputed)

        names = self._build_feature_names(encode_step)

        if hasattr(clf_step, "coef_"):
            # Linear models (LogisticRegression, SGDClassifier, …)
            # contribution = coefficient × scaled feature value
            contribs = clf_step.coef_[0] * encoded[0]
        elif hasattr(clf_step, "feature_importances_"):
            # Tree/ensemble models (RandomForest, GradientBoosting, ExtraTrees, …)
            # feature_importances_ is always non-negative; sign set by correlation direction
            contribs = clf_step.feature_importances_
        else:
            # Fallback: uniform importance — signals that model type is unsupported
            contribs = np.ones(encoded.shape[1]) / encoded.shape[1]

        top_idx = np.argsort(np.abs(contribs))[::-1][:top_n]
        return [
            Contributor(
                feature=names[i] if i < len(names) else f"feature_{i}",
                feature_label=_LABEL_MAP.get(
                    names[i] if i < len(names) else "", f"feature_{i}"
                ),
                contribution=float(contribs[i]),
                direction="aumenta risco de churn" if contribs[i] > 0 else "reduz risco de churn",
            )
            for i in top_idx
        ]

    def _build_feature_names(self, encode_step) -> list[str]:
        names: list[str] = []
        for name, transformer, columns in encode_step.transformers_:
            if name == "num":
                # Prefer string names; fall back when columns are integer indices.
                if hasattr(transformer, "feature_names_in_"):
                    col_names = transformer.feature_names_in_.tolist()
                elif isinstance(columns, list) and columns and isinstance(columns[0], str):
                    col_names = columns
                else:
                    col_names = self._num_features
                names.extend(col_names)
            elif name == "cat":
                # Use string names so OHE produces "ColName_Value" instead of "x0_Value".
                if isinstance(columns, list) and columns and isinstance(columns[0], str):
                    col_names = columns
                else:
                    col_names = self._cat_features
                if hasattr(transformer, "get_feature_names_out"):
                    names.extend(transformer.get_feature_names_out(col_names).tolist())
                else:
                    names.extend(col_names)
        return names
