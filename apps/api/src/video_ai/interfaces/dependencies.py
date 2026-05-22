"""FastAPI dependencies."""

from functools import lru_cache

from video_ai.application.jobs import JobApplicationService
from video_ai.config.settings import Settings, get_settings
from video_ai.storage.local import LocalStorageService
from video_ai.storage.memory import InMemoryJobRepository


@lru_cache(maxsize=1)
def get_job_repository() -> InMemoryJobRepository:
    """Return the process-local job repository."""
    return InMemoryJobRepository()


def get_job_service() -> JobApplicationService:
    """Build the job application service."""
    settings: Settings = get_settings()
    storage = LocalStorageService(settings.storage_root / "uploads")
    return JobApplicationService(get_job_repository(), storage)
