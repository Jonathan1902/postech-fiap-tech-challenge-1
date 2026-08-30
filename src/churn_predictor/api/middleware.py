import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = structlog.get_logger()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id, route=request.url.path)

        response = await call_next(request)

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info("request", latency_ms=round(latency_ms, 2), status=response.status_code)

        response.headers["X-Request-Id"] = request_id
        return response
