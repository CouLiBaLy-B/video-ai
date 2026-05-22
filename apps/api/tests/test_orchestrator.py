from pathlib import Path

from video_ai.agents.planners import SimpleWorkflowPlanner
from video_ai.application.orchestrator import VideoGenerationOrchestrator
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


def make_job(tmp_path: Path) -> VideoGenerationJob:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"image")
    image = ImageAsset(path=image_path, mime_type="image/png", size_bytes=5)
    request = GenerationRequest(prompt="cinematic cat", image=image)
    return VideoGenerationJob(request=request)


async def test_orchestrator_runs_full_mock_workflow(tmp_path: Path) -> None:
    repository = InMemoryJobRepository()
    storage = LocalStorageService(tmp_path / "storage")
    job = make_job(tmp_path)
    await repository.save(job)
    orchestrator = VideoGenerationOrchestrator(
        repository=repository,
        workflow_planner=SimpleWorkflowPlanner(),
        vision_analyzer=SimpleVisionAnalyzer(),
        prompt_enhancer=SimplePromptEnhancer(),
        model_router=StaticVideoModelRouter(),
        video_generator=MockVideoGenerator(storage),
        quality_reviewer=HeuristicQualityReviewer(),
        runtime_profile="ai=test;video=mock",
    )

    completed = await orchestrator.run(job)

    assert completed.status == JobStatus.COMPLETED
    assert completed.events[0].message.endswith("(ai=test;video=mock)")
    assert completed.plan is not None
    assert completed.plan.steps
    assert completed.video is not None
    assert completed.quality_report is not None
    assert completed.quality_report.accepted is True
    persisted = await repository.get(job.id)
    assert persisted is not None
    assert persisted.status == JobStatus.COMPLETED
