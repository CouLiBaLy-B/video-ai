from pathlib import Path

from video_ai.domain.enums import JobStatus
from video_ai.domain.models import GenerationRequest, ImageAsset, VideoGenerationJob
from video_ai.storage.sqlite import SQLiteJobRepository


def make_job(tmp_path: Path) -> VideoGenerationJob:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"image")
    return VideoGenerationJob(
        request=GenerationRequest(
            prompt="cinematic cat",
            image=ImageAsset(path=image_path, mime_type="image/png", size_bytes=5),
        )
    )


async def test_sqlite_repository_saves_and_loads_job(tmp_path: Path) -> None:
    repository = SQLiteJobRepository(tmp_path / "jobs.sqlite3")
    job = make_job(tmp_path)
    updated = job.transition(JobStatus.ANALYZING, "started")

    await repository.save(updated)
    loaded = await repository.get(job.id)

    assert loaded is not None
    assert loaded.id == job.id
    assert loaded.status == JobStatus.ANALYZING
    assert loaded.status_reason == "started"


async def test_sqlite_repository_persists_across_instances(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    first = SQLiteJobRepository(database)
    job = make_job(tmp_path)
    await first.save(job)

    second = SQLiteJobRepository(database)
    loaded = await second.get(job.id)

    assert loaded is not None
    assert loaded.request.prompt == "cinematic cat"


async def test_sqlite_repository_lists_jobs(tmp_path: Path) -> None:
    repository = SQLiteJobRepository(tmp_path / "jobs.sqlite3")
    await repository.save(make_job(tmp_path))
    await repository.save(make_job(tmp_path))

    jobs = await repository.list_all()

    assert len(jobs) == 2
