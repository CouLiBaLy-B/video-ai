from pathlib import Path

from video_ai.domain.enums import JobStatus
from video_ai.domain.models import GenerationRequest, ImageAsset, VideoGenerationJob
from video_ai.interfaces.schemas import JobResponse


def test_job_transition_appends_events(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"image")
    job = VideoGenerationJob(
        request=GenerationRequest(
            prompt="cinematic cat",
            image=ImageAsset(path=image_path, mime_type="image/png", size_bytes=5),
        )
    )

    updated = job.transition(JobStatus.PLANNING, "planning started")

    assert job.events == []
    assert len(updated.events) == 1
    assert updated.events[0].status == JobStatus.PLANNING
    assert updated.events[0].message == "planning started"


def test_job_response_includes_events(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"image")
    job = VideoGenerationJob(
        request=GenerationRequest(
            prompt="cinematic cat",
            image=ImageAsset(path=image_path, mime_type="image/png", size_bytes=5),
        )
    ).transition(JobStatus.ANALYZING, "analysis")

    response = JobResponse.from_job(job)

    assert response.events[0].status == JobStatus.ANALYZING
    assert response.events[0].message == "analysis"
