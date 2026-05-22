"""Application metrics services."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict

from video_ai.domain.enums import JobStatus
from video_ai.domain.ports import JobRepository


class JobMetrics(BaseModel):
    """Aggregated generation job metrics."""

    model_config = ConfigDict(frozen=True)

    total: int
    by_status: dict[JobStatus, int]
    queued: int = 0
    planning: int = 0
    analyzing: int = 0
    waiting_for_approval: int = 0
    generating: int = 0
    reviewing: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0


class MetricsService:
    """Compute application metrics from repositories."""

    def __init__(self, repository: JobRepository) -> None:
        self._repository = repository

    async def collect_job_metrics(self) -> JobMetrics:
        """Collect job counts by status."""
        jobs = await self._repository.list_all()
        counter = Counter(job.status for job in jobs)
        return JobMetrics(
            total=len(jobs),
            by_status={status: counter.get(status, 0) for status in JobStatus},
            queued=counter.get(JobStatus.QUEUED, 0),
            planning=counter.get(JobStatus.PLANNING, 0),
            analyzing=counter.get(JobStatus.ANALYZING, 0),
            waiting_for_approval=counter.get(JobStatus.WAITING_FOR_APPROVAL, 0),
            generating=counter.get(JobStatus.GENERATING, 0),
            reviewing=counter.get(JobStatus.REVIEWING, 0),
            completed=counter.get(JobStatus.COMPLETED, 0),
            failed=counter.get(JobStatus.FAILED, 0),
            cancelled=counter.get(JobStatus.CANCELLED, 0),
        )
