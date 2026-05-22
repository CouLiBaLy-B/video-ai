"""FastAPI dependencies."""

from functools import lru_cache
from pathlib import Path

from video_ai.application.jobs import JobApplicationService
from video_ai.application.orchestrator import VideoGenerationOrchestrator
from video_ai.config.settings import Settings, get_settings
from video_ai.domain.ports import JobRepository
from video_ai.infrastructure.factories import (
    create_model_router,
    create_prompt_enhancer,
    create_video_generator,
    create_vision_analyzer,
    create_workflow_planner,
)
from video_ai.infrastructure.video import HeuristicQualityReviewer
from video_ai.storage.local import LocalStorageService
from video_ai.storage.memory import InMemoryJobRepository
from video_ai.storage.sqlite import SQLiteJobRepository


@lru_cache(maxsize=1)
def get_memory_job_repository() -> InMemoryJobRepository:
    """Return the process-local in-memory job repository."""
    return InMemoryJobRepository()


@lru_cache(maxsize=1)
def get_sqlite_job_repository(database_path: str) -> SQLiteJobRepository:
    """Return a SQLite-backed job repository."""
    return SQLiteJobRepository(Path(database_path))


def get_job_repository() -> JobRepository:
    """Return the configured job repository."""
    settings = get_settings()
    if settings.job_repository_backend == "sqlite":
        return get_sqlite_job_repository(str(settings.sqlite_database_path))
    return get_memory_job_repository()


def get_upload_storage() -> LocalStorageService:
    """Return storage for uploaded input assets."""
    settings: Settings = get_settings()
    return LocalStorageService(settings.storage_root / "uploads")


def get_output_storage() -> LocalStorageService:
    """Return storage for generated video assets."""
    settings: Settings = get_settings()
    return LocalStorageService(settings.storage_root / "outputs")


def get_job_service() -> JobApplicationService:
    """Build the job application service."""
    return JobApplicationService(get_job_repository(), get_upload_storage())


def get_orchestrator() -> VideoGenerationOrchestrator:
    """Build the configured orchestrator."""
    settings = get_settings()
    return VideoGenerationOrchestrator(
        repository=get_job_repository(),
        workflow_planner=create_workflow_planner(settings),
        vision_analyzer=create_vision_analyzer(settings),
        prompt_enhancer=create_prompt_enhancer(settings),
        model_router=create_model_router(settings),
        video_generator=create_video_generator(settings, get_output_storage()),
        quality_reviewer=HeuristicQualityReviewer(),
    )
