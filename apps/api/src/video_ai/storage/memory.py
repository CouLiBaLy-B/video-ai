"""In-memory repositories for development and tests."""

from __future__ import annotations

from uuid import UUID

from video_ai.domain.models import VideoGenerationJob


class InMemoryJobRepository:
    """Non-persistent job repository."""

    def __init__(self) -> None:
        self._jobs: dict[UUID, VideoGenerationJob] = {}

    async def save(self, job: VideoGenerationJob) -> None:
        """Save or replace a job."""
        self._jobs[job.id] = job

    async def get(self, job_id: UUID) -> VideoGenerationJob | None:
        """Retrieve a job by id."""
        return self._jobs.get(job_id)

    async def list_all(self) -> list[VideoGenerationJob]:
        """Return all jobs."""
        return list(self._jobs.values())
