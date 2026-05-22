from pathlib import Path

from video_ai.application.metrics import MetricsService
from video_ai.domain.enums import JobStatus
from video_ai.domain.models import GenerationRequest, ImageAsset, VideoGenerationJob
from video_ai.storage.memory import InMemoryJobRepository


def make_job(tmp_path: Path) -> VideoGenerationJob:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"image")
    return VideoGenerationJob(
        request=GenerationRequest(
            prompt="cinematic cat",
            image=ImageAsset(path=image_path, mime_type="image/png", size_bytes=5),
        )
    )


async def test_metrics_service_counts_jobs_by_status(tmp_path: Path) -> None:
    repository = InMemoryJobRepository()
    await repository.save(make_job(tmp_path).transition(JobStatus.COMPLETED, "done"))
    await repository.save(make_job(tmp_path).transition(JobStatus.FAILED, "failed"))
    await repository.save(make_job(tmp_path).transition(JobStatus.FAILED, "failed"))

    metrics = await MetricsService(repository).collect_job_metrics()

    assert metrics.total == 3
    assert metrics.completed == 1
    assert metrics.failed == 2
    assert metrics.by_status[JobStatus.FAILED] == 2
