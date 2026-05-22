"""DeepAgents tool adapters for the video workflow."""

from __future__ import annotations

import json
from collections.abc import Callable, Coroutine
from typing import Any

from video_ai.domain.models import (
    EnhancedPrompt,
    GenerationParameters,
    ImageAnalysis,
    VideoGenerationJob,
)
from video_ai.domain.ports import PromptEnhancer, VideoModelRouter, VisionAnalyzer
from video_ai.infrastructure.ltx_video import LtxVideoParameterValidator

AgentTool = Callable[..., Coroutine[Any, Any, str]] | Callable[..., str]


class VideoAgentToolbelt:
    """Typed toolbelt used by DeepAgents runtime wrappers.

    The toolbelt intentionally exposes planning/analysis/routing/validation tools.
    The actual GPU generation remains controlled by the application orchestrator
    so approvals, quotas, cancellation and worker scheduling remain authoritative.
    """

    def __init__(
        self,
        *,
        vision_analyzer: VisionAnalyzer,
        prompt_enhancer: PromptEnhancer,
        model_router: VideoModelRouter,
    ) -> None:
        self._vision_analyzer = vision_analyzer
        self._prompt_enhancer = prompt_enhancer
        self._model_router = model_router
        self._ltx_validator = LtxVideoParameterValidator()

    async def analyze_input_image(self, job: VideoGenerationJob) -> ImageAnalysis:
        """Analyze the job image and prompt."""
        return await self._vision_analyzer.analyze(job.request.image, job.request.prompt)

    async def enhance_video_prompt(
        self, job: VideoGenerationJob, analysis: ImageAnalysis
    ) -> EnhancedPrompt:
        """Enhance the user prompt for video generation."""
        return await self._prompt_enhancer.enhance(job.request.prompt, analysis)

    async def select_video_model(self, job: VideoGenerationJob) -> dict[str, Any]:
        """Select the video model profile for the job."""
        model = await self._model_router.select_model(job.request)
        return model.model_dump(mode="json")

    def validate_ltx_parameters(
        self,
        *,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        inference_steps: int,
        guidance_scale: float,
    ) -> dict[str, Any]:
        """Validate LTX-Video parameters and return JSON-serializable data."""
        result = self._ltx_validator.validate(
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
        )
        return {
            "valid": result.valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "recommended": result.recommended,
        }


def build_deepagents_tools(
    toolbelt: VideoAgentToolbelt, job: VideoGenerationJob
) -> list[AgentTool]:
    """Build DeepAgents-compatible closure tools for a single generation job."""

    async def analyze_input_image() -> str:
        """Analyze the uploaded image and return structured visual context as JSON."""
        analysis = await toolbelt.analyze_input_image(job)
        return analysis.model_dump_json()

    async def enhance_video_prompt(analysis_json: str) -> str:
        """Enhance prompt from ImageAnalysis JSON and return EnhancedPrompt JSON."""
        analysis = ImageAnalysis.model_validate_json(analysis_json)
        prompt = await toolbelt.enhance_video_prompt(job, analysis)
        return prompt.model_dump_json()

    async def select_video_model() -> str:
        """Select an image-to-video model profile and return it as JSON."""
        return json.dumps(await toolbelt.select_video_model(job))

    def validate_ltx_parameters(
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        inference_steps: int,
        guidance_scale: float,
    ) -> str:
        """Validate LTX-Video parameters before expensive GPU generation."""
        return json.dumps(
            toolbelt.validate_ltx_parameters(
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                inference_steps=inference_steps,
                guidance_scale=guidance_scale,
            )
        )

    return [
        analyze_input_image,
        enhance_video_prompt,
        select_video_model,
        validate_ltx_parameters,
    ]


def build_generation_parameters_summary(parameters: GenerationParameters) -> str:
    """Serialize generation parameters for agent context without large objects."""
    payload = {
        "backend": parameters.model.backend.value,
        "model_id": parameters.model.id,
        "width": parameters.width,
        "height": parameters.height,
        "num_frames": parameters.num_frames,
        "fps": parameters.fps,
        "seed": parameters.seed,
        "guidance_scale": parameters.guidance_scale,
        "inference_steps": parameters.inference_steps,
    }
    return json.dumps(payload, sort_keys=True)
