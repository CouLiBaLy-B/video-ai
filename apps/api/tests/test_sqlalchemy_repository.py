from pathlib import Path

from video_ai.domain.enums import JobStatus
from video_ai.domain.models import GenerationRequest, ImageAsset, VideoGenerationJob
from video_ai.storage.sqlalchemy_repository import SqlAlchemyJobRepository


def make_job(tmp_path: Path) -> VideoGenerationJob:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"image")
    return VideoGenerationJob(
        request=GenerationRequest(
            prompt="cinematic cat",
            image=ImageAsset(path=image_path, mime_type="image/png", size_bytes=5),
        )
    )


async def test_sqlalchemy_repository_saves_and_loads_job(tmp_path: Path) -> None:
    repository = SqlAlchemyJobRepository(f"sqlite+pysqlite:///{tmp_path / 'jobs.db'}")
    job = make_job(tmp_path).transition(JobStatus.PLANNING, "planning")

    await repository.save(job)
    loaded = await repository.get(job.id)

    assert loaded is not None
    assert loaded.id == job.id
    assert loaded.status == JobStatus.PLANNING
    assert loaded.status_reason == "planning"


async def test_sqlalchemy_repository_lists_jobs(tmp_path: Path) -> None:
    repository = SqlAlchemyJobRepository(f"sqlite+pysqlite:///{tmp_path / 'jobs.db'}")
    await repository.save(make_job(tmp_path))
    await repository.save(make_job(tmp_path))

    jobs = await repository.list_all()

    assert len(jobs) == 2


def test_sqlalchemy_repository_exposes_engine(tmp_path: Path) -> None:
    repository = SqlAlchemyJobRepository(f"sqlite+pysqlite:///{tmp_path / 'jobs.db'}")

    assert repository.engine is not None
