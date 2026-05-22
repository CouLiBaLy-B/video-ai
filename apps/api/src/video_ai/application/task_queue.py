"""Task queue abstractions for background workflow execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol
from uuid import UUID

from fastapi import BackgroundTasks

from video_ai.application.orchestrator import VideoGenerationOrchestrator
from video_ai.domain.enums import JobStatus
from video_ai.domain.ports import JobRepository


class TaskQueue(Protocol):
    """Queue capable of scheduling async job tasks."""

    def enqueue(self, task: Callable[[], Awaitable[None]]) -> None: ...


class FastApiBackgroundTaskQueue:
    """TaskQueue adapter backed by FastAPI BackgroundTasks."""

    def __init__(self, background_tasks: BackgroundTasks) -> None:
        self._background_tasks = background_tasks

    def enqueue(self, task: Callable[[], Awaitable[None]]) -> None:
        """Schedule a coroutine task after the response is sent."""
        self._background_tasks.add_task(_await_task, task)


async def run_orchestrator_job(
    *,
    orchestrator: VideoGenerationOrchestrator,
    repository: JobRepository,
    job_id: UUID,
) -> None:
    """Run a newly created job if it was not cancelled before execution."""
    job = await repository.get(job_id)
    if job is None or job.status == JobStatus.CANCELLED:
        return
    try:
        await orchestrator.run(job)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        latest = await repository.get(job_id) or job
        if latest.status != JobStatus.CANCELLED:
            failed = latest.transition(JobStatus.FAILED, f"Generation failed: {exc}")
            await repository.save(failed)


async def run_approved_orchestrator_job(
    *,
    orchestrator: VideoGenerationOrchestrator,
    repository: JobRepository,
    job_id: UUID,
) -> None:
    """Run an approved job if it was not cancelled before execution."""
    job = await repository.get(job_id)
    if job is None or job.status == JobStatus.CANCELLED:
        return
    try:
        await orchestrator.run_approved(job)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        latest = await repository.get(job_id) or job
        if latest.status != JobStatus.CANCELLED:
            failed = latest.transition(JobStatus.FAILED, f"Approved generation failed: {exc}")
            await repository.save(failed)


async def _await_task(task: Callable[[], Awaitable[None]]) -> None:
    """Await a queued coroutine factory."""
    await task()
