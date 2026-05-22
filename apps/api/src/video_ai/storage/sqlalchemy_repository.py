"""SQLAlchemy-backed job repository for production databases."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from video_ai.domain.models import VideoGenerationJob


class SqlAlchemyJobRepository:
    """Persist jobs through SQLAlchemy using a portable JSON-document table."""

    def __init__(self, database_url: str) -> None:
        try:
            from sqlalchemy import create_engine
        except ImportError as exc:  # pragma: no cover - optional db dependency
            raise RuntimeError(
                "Install the 'db' extra to use JOB_REPOSITORY_BACKEND=postgres"
            ) from exc
        self._database_url = database_url
        self._engine = create_engine(database_url, future=True)
        self._initialize()

    async def save(self, job: VideoGenerationJob) -> None:
        """Insert or update a job."""
        from sqlalchemy import text

        with self._engine.begin() as connection:
            connection.execute(
                text("DELETE FROM generation_jobs WHERE id = :id"), {"id": str(job.id)}
            )
            connection.execute(
                text(
                    """
                    INSERT INTO generation_jobs (id, status, payload, created_at, updated_at)
                    VALUES (:id, :status, :payload, :created_at, :updated_at)
                    """
                ),
                {
                    "id": str(job.id),
                    "status": job.status.value,
                    "payload": job.model_dump_json(),
                    "created_at": job.created_at.isoformat(),
                    "updated_at": job.updated_at.isoformat(),
                },
            )

    async def get(self, job_id: UUID) -> VideoGenerationJob | None:
        """Retrieve a job by id."""
        from sqlalchemy import text

        with self._engine.begin() as connection:
            row = connection.execute(
                text("SELECT payload FROM generation_jobs WHERE id = :id"),
                {"id": str(job_id)},
            ).fetchone()
        if row is None:
            return None
        return VideoGenerationJob.model_validate_json(str(row[0]))

    async def list_all(self) -> list[VideoGenerationJob]:
        """Return all persisted jobs ordered by creation time descending."""
        from sqlalchemy import text

        with self._engine.begin() as connection:
            rows = connection.execute(
                text("SELECT payload FROM generation_jobs ORDER BY created_at DESC")
            ).fetchall()
        return [VideoGenerationJob.model_validate_json(str(row[0])) for row in rows]

    def _initialize(self) -> None:
        from sqlalchemy import text

        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS generation_jobs (
                        id VARCHAR(36) PRIMARY KEY,
                        status VARCHAR(64) NOT NULL,
                        payload TEXT NOT NULL,
                        created_at VARCHAR(64) NOT NULL,
                        updated_at VARCHAR(64) NOT NULL
                    )
                    """
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_generation_jobs_status "
                    "ON generation_jobs(status)"
                )
            )

    @property
    def engine(self) -> Any:
        """Expose engine for infrastructure-level tests and migrations."""
        return self._engine
