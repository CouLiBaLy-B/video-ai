"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from video_ai.application.metrics import MetricsService
from video_ai.config.settings import get_settings
from video_ai.infrastructure.logging import configure_logging
from video_ai.infrastructure.prometheus import GLOBAL_PROMETHEUS_METRICS, render_prometheus
from video_ai.infrastructure.sentry import configure_sentry
from video_ai.interfaces.api import router
from video_ai.interfaces.middleware import (
    InMemoryRateLimitMiddleware,
    RateLimitConfig,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
)


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""
    configure_logging()
    settings = get_settings()
    configure_sentry(
        dsn=settings.sentry_dsn,
        traces_sample_rate=settings.sentry_traces_sample_rate,
    )
    app = FastAPI(title="Video AI", version="0.1.0")
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        InMemoryRateLimitMiddleware,
        config=RateLimitConfig(
            enabled=settings.rate_limit_enabled,
            requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        from video_ai.interfaces.dependencies import get_job_repository

        job_metrics = await MetricsService(get_job_repository()).collect_job_metrics()
        return render_prometheus(GLOBAL_PROMETHEUS_METRICS, job_metrics)

    return app


app = create_app()
