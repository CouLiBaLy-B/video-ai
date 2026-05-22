from pathlib import Path

from video_ai.agents.planners import SimpleWorkflowPlanner
from video_ai.application.orchestrator import VideoGenerationOrchestrator
from video_ai.domain.enums import VideoBackend
from video_ai.domain.models import GenerationRequest, ImageAsset, VideoGenerationJob
from video_ai.infrastructure.composite_video import CompositeVideoGenerator
from video_ai.infrastructure.routing import RequestAwareVideoModelRouter
from video_ai.infrastructure.simple_ai import SimplePromptEnhancer, SimpleVisionAnalyzer
from video_ai.infrastructure.video import HeuristicQualityReviewer, MockVideoGenerator
from video_ai.storage.local import LocalStorageService
from video_ai.storage.memory import InMemoryJobRepository


def make_job(tmp_path: Path) -> VideoGenerationJob:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"image")
    request = GenerationRequest(
        prompt="cinematic cat",
        image=ImageAsset(path=image_path, mime_type="image/png", size_bytes=5),
        preferences={"requested_backend": "mock", "width": 320, "height": 240, "fps": 12},
    )
    return VideoGenerationJob(request=request)


async def test_orchestrator_applies_request_preferences(tmp_path: Path) -> None:
    repository = InMemoryJobRepository()
    storage = LocalStorageService(tmp_path / "storage")
    job = make_job(tmp_path)
    router = RequestAwareVideoModelRouter(
        default_backend=VideoBackend.MOCK,
        profiles={VideoBackend.MOCK: await __import__(
            "video_ai.infrastructure.video", fromlist=["StaticVideoModelRouter"]
        ).StaticVideoModelRouter().select_model(job.request)},
    )
    orchestrator = VideoGenerationOrchestrator(
        repository=repository,
        workflow_planner=SimpleWorkflowPlanner(),
        vision_analyzer=SimpleVisionAnalyzer(),
        prompt_enhancer=SimplePromptEnhancer(),
        model_router=router,
        video_generator=CompositeVideoGenerator({VideoBackend.MOCK: MockVideoGenerator(storage)}),
        quality_reviewer=HeuristicQualityReviewer(),
    )

    completed = await orchestrator.run(job)

    assert completed.video is not None
    assert completed.video.width == 320
    assert completed.video.height == 240
    assert completed.video.fps == 12
