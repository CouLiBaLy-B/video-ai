"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from video_ai.infrastructure.logging import configure_logging
from video_ai.interfaces.api import router
from video_ai.interfaces.middleware import RequestLoggingMiddleware


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""
    configure_logging()
    app = FastAPI(title="Video AI", version="0.1.0")
    app.add_middleware(RequestLoggingMiddleware)
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

    return app


app = create_app()
