"""Infrastructure factories for runtime-configurable adapters."""

from __future__ import annotations

from video_ai.agents.deepagents_factory import DeepAgentConfig
from video_ai.agents.planners import DeepAgentsWorkflowPlanner, SimpleWorkflowPlanner
from video_ai.config.settings import Settings
from video_ai.domain.enums import VideoBackend
from video_ai.domain.models import ModelProfile
from video_ai.domain.ports import PromptEnhancer, StorageService, VideoGenerator, VideoModelRouter, VisionAnalyzer, WorkflowPlanner
from video_ai.infrastructure.ltx_video import LTX_VIDEO_2B_DISTILLED_PROFILE, LtxVideoGenerator
from video_ai.infrastructure.simple_ai import SimplePromptEnhancer, SimpleVisionAnalyzer
from video_ai.infrastructure.video import MockVideoGenerator, StaticVideoModelRouter
from video_ai.infrastructure.vllm import VllmChatGateway, VllmPromptEnhancer, VllmVisionAnalyzer


WAN_I2V_A14B_PROFILE = ModelProfile(
    id="Wan-AI/Wan2.2-I2V-A14B",
    backend=VideoBackend.WAN_I2V,
    display_name="Wan2.2 I2V A14B",
    supports_image_to_video=True,
    supports_text_to_video=False,
    min_vram_gb=80,
    default_width=832,
    default_height=480,
    default_fps=16,
)


def create_workflow_planner(settings: Settings) -> WorkflowPlanner:
    """Create the configured agentic workflow planner."""
    if settings.agent_planner_provider == "deepagents":
        return DeepAgentsWorkflowPlanner(
            DeepAgentConfig(model=settings.deepagents_model, skills_path="skills")
        )
    return SimpleWorkflowPlanner()


def create_vision_analyzer(settings: Settings) -> VisionAnalyzer:
    """Create the configured vision analyzer."""
    if settings.ai_provider == "vllm":
        gateway = VllmChatGateway(
            base_url=settings.vllm_vision_base_url,
            api_key=settings.vllm_api_key,
            model=settings.vllm_vision_model,
            timeout_seconds=settings.vllm_timeout_seconds,
        )
        return VllmVisionAnalyzer(gateway)
    return SimpleVisionAnalyzer()


def create_prompt_enhancer(settings: Settings) -> PromptEnhancer:
    """Create the configured prompt enhancer."""
    if settings.ai_provider == "vllm":
        gateway = VllmChatGateway(
            base_url=settings.vllm_text_base_url,
            api_key=settings.vllm_api_key,
            model=settings.vllm_text_model,
            timeout_seconds=settings.vllm_timeout_seconds,
        )
        return VllmPromptEnhancer(gateway)
    return SimplePromptEnhancer()


def create_model_router(settings: Settings) -> VideoModelRouter:
    """Create a router profile aligned with the configured video backend."""
    return StaticVideoModelRouter(default_profile=_profile_for_backend(settings))


def create_video_generator(settings: Settings, storage: StorageService) -> VideoGenerator:
    """Create the configured video generator."""
    if settings.video_generator_backend == "ltx-video":
        return LtxVideoGenerator(
            storage=storage,
            model_id=settings.ltx_video_model_id,
            torch_dtype=settings.ltx_video_torch_dtype,
            device=settings.ltx_video_device,
        )
    if settings.video_generator_backend == "wan-i2v":
        # A dedicated Wan adapter will be implemented once runtime requirements
        # are finalized. We fail fast instead of silently using the wrong model.
        raise NotImplementedError("Wan I2V generation adapter is not implemented yet")
    return MockVideoGenerator(storage)


def _profile_for_backend(settings: Settings) -> ModelProfile:
    if settings.video_generator_backend == "ltx-video":
        return LTX_VIDEO_2B_DISTILLED_PROFILE.model_copy(
            update={"id": settings.ltx_video_model_id}
        )
    if settings.video_generator_backend == "wan-i2v":
        return WAN_I2V_A14B_PROFILE
    return ModelProfile(
        id="mock-video-v1",
        backend=VideoBackend.MOCK,
        display_name="Mock Video Generator",
        supports_image_to_video=True,
        supports_text_to_video=False,
        min_vram_gb=0,
        default_width=512,
        default_height=512,
        default_fps=24,
    )
