"""HTTP middleware."""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger("video_ai.http")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Attach request ids and emit structured request timing logs."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request and log duration."""
        request_id = request.headers.get("x-request-id", str(uuid4()))
        start = time.perf_counter()
        response: Response
        try:
            response = await call_next(request)
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


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add baseline browser security headers to every response."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Apply security headers."""
        response = await call_next(request)
        response.headers.setdefault("x-content-type-options", "nosniff")
        response.headers.setdefault("x-frame-options", "DENY")
        response.headers.setdefault("referrer-policy", "no-referrer")
        response.headers.setdefault(
            "permissions-policy", "camera=(), microphone=(), geolocation=()"
        )
        return response


@dataclass(frozen=True)
class RateLimitConfig:
    """In-memory fixed-window-ish rate limit configuration."""

    enabled: bool
    requests: int
    window_seconds: int


class InMemoryRateLimitMiddleware(BaseHTTPMiddleware):
    """Simple per-client in-memory sliding-window rate limiter.

    This is suitable for local/dev and single-process deployments. Production
    multi-process deployments should use a shared store such as Redis or an API
    gateway rate limiter.
    """

    def __init__(self, app: object, config: RateLimitConfig) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._config = config
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Reject requests exceeding the configured per-client limit."""
        if not self._config.enabled or request.url.path == "/health":
            return await call_next(request)

        client_key = self._client_key(request)
        now = time.monotonic()
        bucket = self._requests[client_key]
        window_start = now - self._config.window_seconds
        while bucket and bucket[0] < window_start:
            bucket.popleft()

        if len(bucket) >= self._config.requests:
            retry_after = max(1, int(self._config.window_seconds - (now - bucket[0])))
            return JSONResponse(
                {"detail": "Rate limit exceeded"},
                status_code=429,
                headers={"retry-after": str(retry_after)},
            )

        bucket.append(now)
        response = await call_next(request)
        response.headers["x-ratelimit-limit"] = str(self._config.requests)
        response.headers["x-ratelimit-remaining"] = str(max(0, self._config.requests - len(bucket)))
        return response

    @staticmethod
    def _client_key(request: Request) -> str:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",", maxsplit=1)[0].strip()
        if request.client is None:
            return "unknown"
        return request.client.host
