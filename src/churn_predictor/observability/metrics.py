from __future__ import annotations

try:
    from prometheus_client import Counter, Histogram

    PREDICTION_COUNTER = Counter(
        "churn_predictions_total",
        "Total predictions by label",
        ["label"],
    )
    PREDICTION_LATENCY = Histogram(
        "churn_prediction_latency_seconds",
        "Prediction endpoint latency",
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    )
    ERROR_COUNTER = Counter(
        "churn_errors_total",
        "Errors by route",
        ["route"],
    )
    PROMETHEUS_AVAILABLE = True
except Exception:
    PROMETHEUS_AVAILABLE = False
