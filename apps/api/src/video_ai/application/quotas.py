"""User quota and generation limit enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from video_ai.domain.enums import JobStatus
from video_ai.domain.ports import JobRepository


class QuotaExceededError(ValueError):
    """Raised when a user exceeds a configured quota or generation limit."""


@dataclass(frozen=True)
class QuotaLimits:
    """Configurable quota limits."""

    max_active_jobs_per_user: int
    max_daily_jobs_per_user: int
    max_generation_width: int
    max_generation_height: int
    max_generation_frames: int


class QuotaService:
    """Enforce per-user job quotas and generation parameter bounds."""

    _ACTIVE_STATUSES = {
        JobStatus.QUEUED,
        JobStatus.ANALYZING,
        JobStatus.PLANNING,
        JobStatus.WAITING_FOR_APPROVAL,
        JobStatus.GENERATING,
        JobStatus.REVIEWING,
    }

    def __init__(self, repository: JobRepository, limits: QuotaLimits) -> None:
        self._repository = repository
        self._limits = limits

    async def validate_create(
        self,
        *,
        user_id: str | None,
        width: int | None,
        height: int | None,
        num_frames: int | None,
    ) -> None:
        """Validate quotas and requested generation bounds before job creation."""
        self._validate_parameter_bounds(width=width, height=height, num_frames=num_frames)
        if user_id is None:
            return
        jobs = [job for job in await self._repository.list_all() if job.request.user_id == user_id]
        active_count = sum(1 for job in jobs if job.status in self._ACTIVE_STATUSES)
        if active_count >= self._limits.max_active_jobs_per_user:
            raise QuotaExceededError("Maximum active jobs per user exceeded")

        since = datetime.now(UTC) - timedelta(days=1)
        daily_count = sum(1 for job in jobs if job.created_at >= since)
        if daily_count >= self._limits.max_daily_jobs_per_user:
            raise QuotaExceededError("Maximum daily jobs per user exceeded")

    def _validate_parameter_bounds(
        self,
        *,
        width: int | None,
        height: int | None,
        num_frames: int | None,
    ) -> None:
        if width is not None and width > self._limits.max_generation_width:
            raise QuotaExceededError("Requested width exceeds configured maximum")
        if height is not None and height > self._limits.max_generation_height:
            raise QuotaExceededError("Requested height exceeds configured maximum")
        if num_frames is not None and num_frames > self._limits.max_generation_frames:
            raise QuotaExceededError("Requested frame count exceeds configured maximum")
