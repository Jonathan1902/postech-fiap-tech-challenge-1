from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from churn_predictor.api.dependencies import get_predictor
from churn_predictor.api.middleware import RequestLoggingMiddleware
from churn_predictor.api.routes import health as health_router
from churn_predictor.api.routes import predict as predict_router
from churn_predictor.config import settings
from churn_predictor.observability.logging import configure_logging

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    try:
        get_predictor()  # fail-fast: load model at startup
    except FileNotFoundError as exc:
        logger.error("model_not_found", path=str(settings.model_path), error=str(exc))
        raise
    except Exception as exc:
        logger.error("model_load_failed", error=str(exc))
        raise
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Churn Predictor API",
        description="Predição de churn de clientes de telecomunicações.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(RequestLoggingMiddleware)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "unhandled_exception",
            error=type(exc).__name__,
            detail=str(exc),
            path=str(request.url.path),
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Check logs for details."},
        )

    app.include_router(health_router.router)
    app.include_router(predict_router.router)

    static_dir = Path("static")
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


app = create_app()
