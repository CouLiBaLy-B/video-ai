from pathlib import Path

from video_ai.agents.planners import SimpleWorkflowPlanner
from video_ai.application.orchestrator import VideoGenerationOrchestrator
from video_ai.application.task_queue import run_orchestrator_job
from video_ai.domain.enums import JobStatus
from video_ai.domain.models import GenerationRequest, ImageAsset, VideoGenerationJob
from video_ai.infrastructure.simple_ai import SimplePromptEnhancer, SimpleVisionAnalyzer
from video_ai.infrastructure.video import (
    HeuristicQualityReviewer,
    MockVideoGenerator,
    StaticVideoModelRouter,
)
from video_ai.storage.local import LocalStorageService
from video_ai.storage.memory import InMemoryJobRepository


def make_orchestrator(
    repository: InMemoryJobRepository, tmp_path: Path
) -> VideoGenerationOrchestrator:
    return VideoGenerationOrchestrator(
        repository=repository,
        workflow_planner=SimpleWorkflowPlanner(),
        vision_analyzer=SimpleVisionAnalyzer(),
        prompt_enhancer=SimplePromptEnhancer(),
        model_router=StaticVideoModelRouter(),
        video_generator=MockVideoGenerator(LocalStorageService(tmp_path / "storage")),
        quality_reviewer=HeuristicQualityReviewer(),
    )


def make_job(tmp_path: Path) -> VideoGenerationJob:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"image")
    return VideoGenerationJob(
        request=GenerationRequest(
            prompt="cinematic cat",
            image=ImageAsset(path=image_path, mime_type="image/png", size_bytes=5),
        )
    )


async def test_queued_task_does_not_run_cancelled_job(tmp_path: Path) -> None:
    repository = InMemoryJobRepository()
    job = make_job(tmp_path).transition(JobStatus.CANCELLED, "cancelled")
    await repository.save(job)

    await run_orchestrator_job(
        orchestrator=make_orchestrator(repository, tmp_path), repository=repository, job_id=job.id
    )

    loaded = await repository.get(job.id)
    assert loaded is not None
    assert loaded.status == JobStatus.CANCELLED
