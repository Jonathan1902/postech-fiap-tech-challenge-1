import hashlib
import time

import structlog
from fastapi import APIRouter, Depends

from churn_predictor.api.dependencies import get_predictor
from churn_predictor.domain.schemas import CustomerProfile, PredictionResponse
from churn_predictor.models.predictor import ChurnPredictor

router = APIRouter()
logger = structlog.get_logger()


@router.post("/predict", response_model=PredictionResponse)
def predict(
    profile: CustomerProfile,
    predictor: ChurnPredictor = Depends(get_predictor),
) -> PredictionResponse:
    payload_hash = hashlib.sha256(profile.model_dump_json().encode()).hexdigest()[:12]

    t0 = time.perf_counter()
    result = predictor.predict(profile)

    top = result.top_contributors[0].feature if result.top_contributors else "n/a"
    logger.info(
        "prediction",
        input_hash=payload_hash,
        latency=round(time.perf_counter() - t0, 4),
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
