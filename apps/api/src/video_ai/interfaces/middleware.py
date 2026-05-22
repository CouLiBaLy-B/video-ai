"""HTTP middleware."""

from __future__ import annotations

import logging
import time
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("video_ai.http")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Attach request ids and emit structured request timing logs."""

    async def dispatch(self, request: Request, call_next: object) -> Response:
        """Process request and log duration."""
        request_id = request.headers.get("x-request-id", str(uuid4()))
        start = time.perf_counter()
        response: Response
        try:
            response = await call_next(request)  # type: ignore[operator]
        except Exception:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.exception(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                },
            )
            raise
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["x-request-id"] = request_id
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response
