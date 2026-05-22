"""Task queue abstractions for background workflow execution."""

from __future__ import annotations

import asyncio
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


class JobTaskQueue(Protocol):
    """Queue capable of scheduling generation jobs by id."""

    def enqueue_generation(self, job_id: UUID) -> str: ...

    def enqueue_approved_generation(self, job_id: UUID) -> str: ...


class FastApiBackgroundTaskQueue:
    """TaskQueue adapter backed by FastAPI BackgroundTasks."""

    def __init__(self, background_tasks: BackgroundTasks) -> None:
        self._background_tasks = background_tasks

    def enqueue(self, task: Callable[[], Awaitable[None]]) -> None:
        """Schedule a coroutine task after the response is sent."""
        self._background_tasks.add_task(_await_task, task)


class FastApiJobTaskQueue:
    """JobTaskQueue using FastAPI BackgroundTasks for local/dev execution."""

    def __init__(
        self,
        *,
        background_tasks: BackgroundTasks,
        orchestrator: VideoGenerationOrchestrator,
        repository: JobRepository,
    ) -> None:
        self._queue = FastApiBackgroundTaskQueue(background_tasks)
        self._orchestrator = orchestrator
        self._repository = repository

    def enqueue_generation(self, job_id: UUID) -> str:
        """Schedule a newly created generation job."""
        self._queue.enqueue(
            lambda: run_orchestrator_job(
                orchestrator=self._orchestrator,
                repository=self._repository,
                job_id=job_id,
            )
        )
        return f"fastapi:{job_id}"

    def enqueue_approved_generation(self, job_id: UUID) -> str:
        """Schedule an approved generation job."""
        self._queue.enqueue(
            lambda: run_approved_orchestrator_job(
                orchestrator=self._orchestrator,
                repository=self._repository,
                job_id=job_id,
            )
        )
        return f"fastapi-approved:{job_id}"


class RedisRqJobTaskQueue:
    """JobTaskQueue adapter backed by Redis Queue (RQ)."""

    def __init__(self, *, redis_url: str, queue_name: str) -> None:
        try:
            from redis import Redis
            from rq import Queue
        except ImportError as exc:  # pragma: no cover - optional worker dependency
            raise RuntimeError(
                "Install the 'worker' extra to use TASK_QUEUE_BACKEND=redis-rq"
            ) from exc
        self._queue = Queue(queue_name, connection=Redis.from_url(redis_url))

    def enqueue_generation(self, job_id: UUID) -> str:
        """Schedule a newly created generation job in RQ."""
        job = self._queue.enqueue("video_ai.workers.jobs.run_job_task", str(job_id))
        return str(job.id)

    def enqueue_approved_generation(self, job_id: UUID) -> str:
        """Schedule an approved generation job in RQ."""
        job = self._queue.enqueue("video_ai.workers.jobs.run_approved_job_task", str(job_id))
        return str(job.id)


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


def run_async(coro: Awaitable[None]) -> None:
    """Run a coroutine from a synchronous worker function."""
    asyncio.run(coro)


async def _await_task(task: Callable[[], Awaitable[None]]) -> None:
    """Await a queued coroutine factory."""
    await task()
