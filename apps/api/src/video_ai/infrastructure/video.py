"""Video generation infrastructure adapters."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from video_ai.domain.enums import VideoBackend
from video_ai.domain.models import (
    GeneratedVideo,
    GenerationParameters,
    GenerationRequest,
    ModelProfile,
    QualityReport,
)
from video_ai.domain.ports import StorageService


class StaticVideoModelRouter:
    """Route requests to a configured default model profile."""

    def __init__(self, default_profile: ModelProfile | None = None) -> None:
        self._default_profile = default_profile or ModelProfile(
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

    async def select_model(self, request: GenerationRequest) -> ModelProfile:
        """Return the default model profile.

        The request is accepted to satisfy the router port and to allow future
        model-selection policies based on prompt, image and infrastructure.
        """
        _ = request
        return self._default_profile


class MockVideoGenerator:
    """GPU-free generator used for tests and end-to-end development."""

    def __init__(self, storage: StorageService) -> None:
        self._storage = storage

    async def generate(self, parameters: GenerationParameters) -> GeneratedVideo:
        """Generate a deterministic mock video artifact."""
        payload = {
            "kind": "mock-video",
            "created_at": datetime.now(UTC).isoformat(),
            "model": parameters.model.id,
            "prompt": parameters.prompt.positive,
            "negative_prompt": parameters.prompt.negative,
            "width": parameters.width,
            "height": parameters.height,
            "num_frames": parameters.num_frames,
            "fps": parameters.fps,
            "seed": parameters.seed,
        }
        content = b"MOCK_MP4\n" + json.dumps(payload, sort_keys=True).encode("utf-8")
        storage = await self._storage.save_bytes(content, "mock-video.mp4", "video/mp4")
        return GeneratedVideo(
            storage=storage,
            width=parameters.width,
            height=parameters.height,
            fps=parameters.fps,
            num_frames=parameters.num_frames,
            duration_seconds=parameters.num_frames / parameters.fps,
            seed=parameters.seed,
            model_id=parameters.model.id,
        )


class HeuristicQualityReviewer:
    """Simple deterministic quality reviewer for non-agentic tests."""

    async def review(
        self,
        request: GenerationRequest,
        video: GeneratedVideo,
        parameters: GenerationParameters,
    ) -> QualityReport:
        """Return a basic acceptance report."""
        issues: list[str] = []
        if video.duration_seconds < 1.0:
            issues.append("Video is very short")
        if request.prompt.lower() not in parameters.prompt.positive.lower():
            issues.append("Enhanced prompt may have drifted from user prompt")
        score = 0.9 if not issues else 0.65
        return QualityReport(
            score=score,
            summary="Mock quality review completed.",
            issues=issues,
            suggested_improvements=[] if not issues else ["Regenerate with a clearer prompt"],
            accepted=score >= 0.7,
        )
