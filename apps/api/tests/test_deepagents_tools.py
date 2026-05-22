from pathlib import Path

from video_ai.agents.tools import VideoAgentToolbelt, build_deepagents_tools
from video_ai.domain.enums import VideoBackend
from video_ai.domain.models import GenerationRequest, ImageAsset, VideoGenerationJob
from video_ai.infrastructure.routing import RequestAwareVideoModelRouter
from video_ai.infrastructure.simple_ai import SimplePromptEnhancer, SimpleVisionAnalyzer
from video_ai.infrastructure.video import StaticVideoModelRouter


def make_job(tmp_path: Path) -> VideoGenerationJob:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"image")
    return VideoGenerationJob(
        request=GenerationRequest(
            prompt="cinematic cat",
            image=ImageAsset(path=image_path, mime_type="image/png", size_bytes=5),
        )
    )


async def test_deepagents_tools_execute_analysis_prompt_and_routing(tmp_path: Path) -> None:
    job = make_job(tmp_path)
    profile = await StaticVideoModelRouter().select_model(job.request)
    toolbelt = VideoAgentToolbelt(
        vision_analyzer=SimpleVisionAnalyzer(),
        prompt_enhancer=SimplePromptEnhancer(),
        model_router=RequestAwareVideoModelRouter(
            default_backend=VideoBackend.MOCK,
            profiles={VideoBackend.MOCK: profile},
        ),
    )
    tools = {tool.__name__: tool for tool in build_deepagents_tools(toolbelt, job)}

    analysis_json = await tools["analyze_input_image"]()
    prompt_json = await tools["enhance_video_prompt"](analysis_json)
    model_json = await tools["select_video_model"]()
    validation_json = tools["validate_ltx_parameters"](768, 512, 49, 24, 20, 3.5)

    assert "main subject" in analysis_json
    assert "cinematic cat" in prompt_json
    assert "mock-video-v1" in model_json
    assert '"valid": true' in validation_json
