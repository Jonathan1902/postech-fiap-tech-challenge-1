from functools import lru_cache

from churn_predictor.config import settings
from churn_predictor.models.predictor import ChurnPredictor


@lru_cache(maxsize=1)
def get_predictor() -> ChurnPredictor:
    return ChurnPredictor(settings.model_path)
