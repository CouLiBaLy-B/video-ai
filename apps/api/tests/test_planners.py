from pathlib import Path

from video_ai.agents.planners import SimpleWorkflowPlanner
from video_ai.domain.models import GenerationRequest, ImageAsset, VideoGenerationJob


async def test_simple_workflow_planner_returns_specialized_steps(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"image")
    job = VideoGenerationJob(
        request=GenerationRequest(
            prompt="cinematic cat",
            image=ImageAsset(path=image_path, mime_type="image/png", size_bytes=5),
        )
    )

    plan = await SimpleWorkflowPlanner().plan(job)

    assert plan.requires_human_approval is False
    assert {step.agent for step in plan.steps} >= {
        "vision-analysis-agent",
        "cinematic-prompt-agent",
        "model-routing-agent",
        "quality-review-agent",
    }
