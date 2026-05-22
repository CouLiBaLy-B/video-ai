"""SQLite job repository."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import UUID

from video_ai.domain.models import VideoGenerationJob


class SQLiteJobRepository:
    """Persist jobs as JSON documents in SQLite.

    This repository intentionally stores the aggregate as a JSON document while
    the product model is still evolving. It gives durable local persistence
    without prematurely coupling the domain to a relational schema.
    """

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    async def save(self, job: VideoGenerationJob) -> None:
        """Insert or update a job."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO generation_jobs (id, status, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (
                    str(job.id),
                    job.status.value,
                    job.model_dump_json(),
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                ),
            )
            connection.commit()

    async def get(self, job_id: UUID) -> VideoGenerationJob | None:
        """Retrieve a job by id."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM generation_jobs WHERE id = ?",
                (str(job_id),),
            ).fetchone()
        if row is None:
            return None
        return VideoGenerationJob.model_validate_json(str(row[0]))

    async def list_all(self) -> list[VideoGenerationJob]:
        """Return all persisted jobs ordered by creation time descending."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM generation_jobs ORDER BY created_at DESC",
            ).fetchall()
        return [VideoGenerationJob.model_validate_json(str(row[0])) for row in rows]

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS generation_jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_generation_jobs_status ON generation_jobs(status)"
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._database_path)
