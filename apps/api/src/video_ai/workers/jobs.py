"""RQ worker entrypoints for video generation jobs."""

from __future__ import annotations

from uuid import UUID

from video_ai.application.task_queue import (
    run_approved_orchestrator_job,
    run_async,
    run_orchestrator_job,
)
from video_ai.interfaces.dependencies import get_job_repository, get_orchestrator


def run_job_task(job_id: str) -> None:
    """RQ entrypoint for newly created jobs."""
    run_async(
        run_orchestrator_job(
            orchestrator=get_orchestrator(),
            repository=get_job_repository(),
            job_id=UUID(job_id),
        )
    )


def run_approved_job_task(job_id: str) -> None:
    """RQ entrypoint for approved GPU jobs."""
    run_async(
        run_approved_orchestrator_job(
            orchestrator=get_orchestrator(),
            repository=get_job_repository(),
            job_id=UUID(job_id),
        )
    )
