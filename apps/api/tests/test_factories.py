from pathlib import Path

import pytest

from video_ai.agents.planners import DeepAgentsWorkflowPlanner, SimpleWorkflowPlanner
from video_ai.config.settings import Settings
from video_ai.domain.enums import VideoBackend
from video_ai.infrastructure.factories import (
    create_model_router,
    create_prompt_enhancer,
    create_video_generator,
    create_vision_analyzer,
    create_workflow_planner,
)
from video_ai.infrastructure.ltx_video import LtxVideoGenerator
from video_ai.infrastructure.simple_ai import SimplePromptEnhancer, SimpleVisionAnalyzer
from video_ai.infrastructure.video import MockVideoGenerator
from video_ai.infrastructure.vllm import VllmPromptEnhancer, VllmVisionAnalyzer
from video_ai.storage.local import LocalStorageService


def test_planner_factory_returns_simple_by_default() -> None:
    assert isinstance(create_workflow_planner(Settings()), SimpleWorkflowPlanner)


def test_planner_factory_can_return_deepagents_planner() -> None:
    planner = create_workflow_planner(Settings(agent_planner_provider="deepagents"))

    assert isinstance(planner, DeepAgentsWorkflowPlanner)


def test_mock_ai_factories_return_simple_adapters() -> None:
    settings = Settings(ai_provider="mock")

    assert isinstance(create_vision_analyzer(settings), SimpleVisionAnalyzer)
    assert isinstance(create_prompt_enhancer(settings), SimplePromptEnhancer)


async def test_router_uses_ltx_profile_when_configured() -> None:
    settings = Settings(video_generator_backend="ltx-video", ltx_video_model_id="custom/ltx")
    router = create_model_router(settings)

    profile = await router.select_model(request=object())  # type: ignore[arg-type]

    assert profile.backend == VideoBackend.LTX_VIDEO
    assert profile.id == "custom/ltx"


def test_vllm_factories_return_vllm_adapters() -> None:
    settings = Settings(ai_provider="vllm")

    assert isinstance(create_vision_analyzer(settings), VllmVisionAnalyzer)
    assert isinstance(create_prompt_enhancer(settings), VllmPromptEnhancer)


def test_video_generator_factory_returns_mock_by_default(tmp_path: Path) -> None:
    storage = LocalStorageService(tmp_path)

    generator = create_video_generator(Settings(video_generator_backend="mock"), storage)

    assert isinstance(generator, MockVideoGenerator)


def test_video_generator_factory_returns_ltx(tmp_path: Path) -> None:
    storage = LocalStorageService(tmp_path)

    generator = create_video_generator(Settings(video_generator_backend="ltx-video"), storage)

    assert isinstance(generator, LtxVideoGenerator)


def test_video_generator_factory_fails_fast_for_wan(tmp_path: Path) -> None:
    storage = LocalStorageService(tmp_path)

    with pytest.raises(NotImplementedError):
        create_video_generator(Settings(video_generator_backend="wan-i2v"), storage)
