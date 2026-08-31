import logging

from fastapi import APIRouter, Depends

from churn_predictor.api.dependencies import get_predictor
from churn_predictor.domain.schemas import CustomerProfile, PredictionResponse
from churn_predictor.models.predictor import ChurnPredictor

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/predict", response_model=PredictionResponse)
def predict(
    profile: CustomerProfile,
    predictor: ChurnPredictor = Depends(get_predictor),
) -> PredictionResponse:
    result = predictor.predict(profile)
    logger.info(
        "prediction probability=%.4f label=%s",
        result.churn_probability,
        result.churn_prediction,
    )
    return result
