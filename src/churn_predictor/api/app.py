from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from churn_predictor.api.dependencies import get_predictor
from churn_predictor.api.middleware import RequestLoggingMiddleware
from churn_predictor.api.routes import health as health_router
from churn_predictor.api.routes import predict as predict_router
from churn_predictor.config import settings
from churn_predictor.observability.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    get_predictor()  # fail-fast: load model at startup
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Churn Predictor API",
        description="Predição de churn de clientes de telecomunicações.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(RequestLoggingMiddleware)

    app.include_router(health_router.router)
    app.include_router(predict_router.router)

    static_dir = Path("static")
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


app = create_app()
