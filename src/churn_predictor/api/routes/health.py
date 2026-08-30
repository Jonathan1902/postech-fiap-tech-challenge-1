import time

from fastapi import APIRouter, Depends

from churn_predictor.api.dependencies import get_predictor
from churn_predictor.domain.schemas import HealthResponse
from churn_predictor.models.predictor import ChurnPredictor

router = APIRouter()
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
def health(predictor: ChurnPredictor = Depends(get_predictor)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_version=predictor.model_version,
        threshold=predictor.threshold,
        feature_contract_hash=predictor.feature_contract_hash,
        uptime_s=round(time.time() - _start_time, 1),
    )
