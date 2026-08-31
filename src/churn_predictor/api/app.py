import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from churn_predictor.api.dependencies import get_predictor
from churn_predictor.api.routes import health as health_router
from churn_predictor.api.routes import predict as predict_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_predictor()
    except FileNotFoundError as exc:
        logger.error("model_not_found: %s", exc)
        raise
    except Exception as exc:
        logger.error("model_load_failed: %s", exc)
        raise
    yield


def create_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO)

    app = FastAPI(
        title="Churn Predictor API",
        description="Predição de churn de clientes de telecomunicações.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception %s: %s", request.url.path, exc)
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
