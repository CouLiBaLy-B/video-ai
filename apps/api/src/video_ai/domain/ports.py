"""Ports used by application services and infrastructure adapters."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from video_ai.domain.models import (
    EnhancedPrompt,
    GeneratedVideo,
    GenerationParameters,
    GenerationRequest,
    ImageAnalysis,
    ImageAsset,
    ModelProfile,
    QualityReport,
    StorageRef,
    VideoGenerationJob,
)


class VisionAnalyzer(Protocol):
    """Analyze an image and prompt into structured visual context."""

    async def analyze(self, image: ImageAsset, prompt: str) -> ImageAnalysis: ...


class PromptEnhancer(Protocol):
    """Transform user prompts into video-model-ready prompts."""

    async def enhance(self, prompt: str, analysis: ImageAnalysis) -> EnhancedPrompt: ...


class VideoModelRouter(Protocol):
    """Select the best video model for a request."""

    async def select_model(self, request: GenerationRequest) -> ModelProfile: ...


class VideoGenerator(Protocol):
    """Generate a video from normalized parameters."""

    async def generate(self, parameters: GenerationParameters) -> GeneratedVideo: ...


class VideoQualityReviewer(Protocol):
    """Review generated videos for quality and prompt adherence."""

    async def review(
        self,
        request: GenerationRequest,
        video: GeneratedVideo,
        parameters: GenerationParameters,
    ) -> QualityReport: ...


class StorageService(Protocol):
    """Store binary assets and return stable references."""

    async def save_bytes(self, content: bytes, filename: str, mime_type: str) -> StorageRef: ...


class JobRepository(Protocol):
    """Persist and retrieve video generation jobs."""

    async def save(self, job: VideoGenerationJob) -> None: ...

    async def get(self, job_id: UUID) -> VideoGenerationJob | None: ...
