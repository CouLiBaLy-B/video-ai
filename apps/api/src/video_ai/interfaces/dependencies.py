"""FastAPI dependencies."""

from functools import lru_cache
from pathlib import Path

from fastapi import BackgroundTasks

from video_ai.application.jobs import JobApplicationService
from video_ai.application.orchestrator import VideoGenerationOrchestrator
from video_ai.application.quotas import QuotaLimits, QuotaService
from video_ai.application.task_queue import (
    FastApiJobTaskQueue,
    JobTaskQueue,
    RedisRqJobTaskQueue,
)
from video_ai.config.settings import Settings, get_settings
from video_ai.domain.ports import JobRepository, StorageService
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
from video_ai.storage.s3 import S3StorageService
from video_ai.storage.sqlalchemy_repository import SqlAlchemyJobRepository
from video_ai.storage.sqlite import SQLiteJobRepository


@lru_cache(maxsize=1)
def get_memory_job_repository() -> InMemoryJobRepository:
    """Return the process-local in-memory job repository."""
    return InMemoryJobRepository()


@lru_cache(maxsize=1)
def get_sqlite_job_repository(database_path: str) -> SQLiteJobRepository:
    """Return a SQLite-backed job repository."""
    return SQLiteJobRepository(Path(database_path))


@lru_cache(maxsize=1)
def get_postgres_job_repository(database_url: str) -> SqlAlchemyJobRepository:
    """Return a SQLAlchemy/Postgres-backed job repository."""
    return SqlAlchemyJobRepository(database_url)


def get_job_repository() -> JobRepository:
    """Return the configured job repository."""
    settings = get_settings()
    if settings.job_repository_backend == "postgres":
        return get_postgres_job_repository(settings.database_url)
    if settings.job_repository_backend == "sqlite":
        return get_sqlite_job_repository(str(settings.sqlite_database_path))
    return get_memory_job_repository()


def get_upload_storage() -> StorageService:
    """Return storage for uploaded input assets."""
    settings: Settings = get_settings()
    return _create_storage_service(settings, "uploads")


def get_output_storage() -> StorageService:
    """Return storage for generated video assets."""
    settings: Settings = get_settings()
    return _create_storage_service(settings, "outputs")


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
        runtime_profile=_runtime_profile(settings),
    )


def _runtime_profile(settings: Settings) -> str:
    """Return a compact runtime profile for event logs."""
    fallback = "fallback=mock" if settings.vllm_fallback_to_mock else "fallback=disabled"
    return (
        f"planner={settings.agent_planner_provider};"
        f"ai={settings.ai_provider};"
        f"{fallback};"
        f"video={settings.video_generator_backend}"
    )


def create_job_task_queue(
    *,
    background_tasks: BackgroundTasks,
    orchestrator: VideoGenerationOrchestrator,
    repository: JobRepository,
) -> JobTaskQueue:
    """Create the configured job task queue adapter."""
    settings = get_settings()
    if settings.task_queue_backend == "redis-rq":
        retry_intervals = [
            int(item.strip())
            for item in settings.rq_retry_intervals_seconds.split(",")
            if item.strip()
        ]
        return RedisRqJobTaskQueue(
            redis_url=settings.redis_url,
            queue_name=settings.rq_queue_name,
            job_timeout_seconds=settings.rq_job_timeout_seconds,
            result_ttl_seconds=settings.rq_result_ttl_seconds,
            failure_ttl_seconds=settings.rq_failure_ttl_seconds,
            retry_max=settings.rq_retry_max,
            retry_intervals_seconds=retry_intervals,
        )
    return FastApiJobTaskQueue(
        background_tasks=background_tasks,
        orchestrator=orchestrator,
        repository=repository,
    )


def _create_storage_service(settings: Settings, namespace: str) -> StorageService:
    """Create configured storage service for a logical namespace."""
    if settings.storage_backend == "s3":
        return S3StorageService(
            bucket=settings.s3_bucket,
            endpoint_url=settings.s3_endpoint_url,
            region_name=settings.s3_region,
            access_key_id=settings.s3_access_key_id,
            secret_access_key=settings.s3_secret_access_key,
            public_base_url=(
                f"{settings.s3_public_base_url.rstrip('/')}/{namespace}"
                if settings.s3_public_base_url
                else None
            ),
            presigned_url_expires_seconds=settings.s3_presigned_url_expires_seconds,
        )
    return LocalStorageService(settings.storage_root / namespace)


def get_quota_service() -> QuotaService:
    """Build quota service from runtime settings."""
    settings = get_settings()
    return QuotaService(
        get_job_repository(),
        QuotaLimits(
            max_active_jobs_per_user=settings.max_active_jobs_per_user,
            max_daily_jobs_per_user=settings.max_daily_jobs_per_user,
            max_generation_width=settings.max_generation_width,
            max_generation_height=settings.max_generation_height,
            max_generation_frames=settings.max_generation_frames,
        ),
    )
