import hashlib
import time

import structlog
from fastapi import APIRouter, Depends

from churn_predictor.api.dependencies import get_predictor
from churn_predictor.domain.schemas import CustomerProfile, PredictionResponse
from churn_predictor.models.predictor import ChurnPredictor
from churn_predictor.observability.metrics import (
    ERROR_COUNTER,
    PREDICTION_COUNTER,
    PREDICTION_LATENCY,
    PROMETHEUS_AVAILABLE,
)

router = APIRouter()
logger = structlog.get_logger()


@router.post("/predict", response_model=PredictionResponse)
def predict(
    profile: CustomerProfile,
    predictor: ChurnPredictor = Depends(get_predictor),
) -> PredictionResponse:
    payload_hash = hashlib.sha256(profile.model_dump_json().encode()).hexdigest()[:12]

    t0 = time.perf_counter()
    try:
        result = predictor.predict(profile)
    except Exception:
        if PROMETHEUS_AVAILABLE:
            ERROR_COUNTER.labels(route="/predict").inc()
        raise
    finally:
        if PROMETHEUS_AVAILABLE:
            PREDICTION_LATENCY.observe(time.perf_counter() - t0)

    if PROMETHEUS_AVAILABLE:
        PREDICTION_COUNTER.labels(label=str(result.churn_prediction)).inc()

    top = result.top_contributors[0].feature if result.top_contributors else "n/a"
    logger.info(
        "prediction",
        input_hash=payload_hash,
        probability=round(result.churn_probability, 4),
        label=result.churn_prediction,
        top_contributor=top,
        model_version=result.model_version,
    )

    return result


@router.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(
    profiles: list[CustomerProfile],
    predictor: ChurnPredictor = Depends(get_predictor),
) -> list[PredictionResponse]:
    return predictor.predict_batch(profiles)
