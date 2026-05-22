from pathlib import Path

import httpx

from video_ai.agents.planners import SimpleWorkflowPlanner
from video_ai.application.orchestrator import VideoGenerationOrchestrator
from video_ai.domain.enums import VideoBackend
from video_ai.domain.models import GenerationRequest, ImageAsset, VideoGenerationJob
from video_ai.infrastructure.composite_video import CompositeVideoGenerator
from video_ai.infrastructure.routing import RequestAwareVideoModelRouter
from video_ai.infrastructure.video import (
    HeuristicQualityReviewer,
    MockVideoGenerator,
    StaticVideoModelRouter,
)
from video_ai.infrastructure.vllm import VllmChatGateway, VllmPromptEnhancer, VllmVisionAnalyzer
from video_ai.storage.local import LocalStorageService
from video_ai.storage.memory import InMemoryJobRepository


def make_transport(content: str) -> httpx.MockTransport:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return httpx.MockTransport(handler)


def make_job(tmp_path: Path) -> VideoGenerationJob:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"image")
    return VideoGenerationJob(
        request=GenerationRequest(
            prompt="cinematic cat",
            image=ImageAsset(path=image_path, mime_type="image/png", size_bytes=5),
        )
    )


async def test_vllm_adapters_complete_generation_workflow(tmp_path: Path) -> None:
    vision_client = httpx.AsyncClient(
        transport=make_transport(
            '{"subject":"cat","scene":"studio","style":"cinematic",'
            '"motion_suggestions":["slow dolly"],"constraints":["stable identity"]}'
        )
    )
    text_client = httpx.AsyncClient(
        transport=make_transport(
            '{"positive":"cinematic cat in studio, slow dolly",'
            '"negative":"blur, flicker", "camera_motion":"slow dolly", "style":"cinematic"}'
        )
    )
    vision_gateway = VllmChatGateway(
        base_url="http://vision.test/v1",
        api_key="test",
        model="vision",
        client=vision_client,
    )
    text_gateway = VllmChatGateway(
        base_url="http://text.test/v1",
        api_key="test",
        model="text",
        client=text_client,
    )
    job = make_job(tmp_path)
    mock_profile = await StaticVideoModelRouter().select_model(job.request)
    orchestrator = VideoGenerationOrchestrator(
        repository=InMemoryJobRepository(),
        workflow_planner=SimpleWorkflowPlanner(),
        vision_analyzer=VllmVisionAnalyzer(vision_gateway),
        prompt_enhancer=VllmPromptEnhancer(text_gateway),
        model_router=RequestAwareVideoModelRouter(
            default_backend=VideoBackend.MOCK,
            profiles={VideoBackend.MOCK: mock_profile},
        ),
        video_generator=CompositeVideoGenerator(
            {VideoBackend.MOCK: MockVideoGenerator(LocalStorageService(tmp_path / "storage"))}
        ),
        quality_reviewer=HeuristicQualityReviewer(),
    )

    completed = await orchestrator.run(job)

    assert completed.video is not None
    assert completed.quality_report is not None
    assert completed.quality_report.accepted is True
    assert completed.status == "completed"
    await vision_client.aclose()
    await text_client.aclose()
