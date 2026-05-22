"""HTTP interface schemas."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from video_ai.domain.enums import JobStatus
from video_ai.domain.models import VideoGenerationJob


class JobResponse(BaseModel):
    """Public job representation."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: JobStatus
    status_reason: str | None
    prompt: str
    created_at: datetime
    updated_at: datetime
    video_url: str | None = None

    @classmethod
    def from_job(cls, job: VideoGenerationJob) -> JobResponse:
        """Map a domain job to an API response."""
        video_url = f"/api/generations/{job.id}/video" if job.video else None
        return cls(
            id=job.id,
            status=job.status,
            status_reason=job.status_reason,
            prompt=job.request.prompt,
            created_at=job.created_at,
            updated_at=job.updated_at,
            video_url=video_url,
        )


class ErrorResponse(BaseModel):
    """Error payload."""

    detail: str
