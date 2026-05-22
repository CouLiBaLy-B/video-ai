"""Composite video generator routing by model backend."""

from __future__ import annotations

from video_ai.domain.enums import VideoBackend
from video_ai.domain.models import GeneratedVideo, GenerationParameters
from video_ai.domain.ports import VideoGenerator


class VideoGeneratorUnavailableError(RuntimeError):
    """Raised when no generator is configured for a requested backend."""


class CompositeVideoGenerator:
    """Route generation to the adapter matching `parameters.model.backend`."""

    def __init__(self, generators: dict[VideoBackend, VideoGenerator]) -> None:
        self._generators = generators

    async def generate(self, parameters: GenerationParameters) -> GeneratedVideo:
        """Generate using the backend selected in the model profile."""
        generator = self._generators.get(parameters.model.backend)
        if generator is None:
            raise VideoGeneratorUnavailableError(
                f"No generator configured for backend {parameters.model.backend.value}"
            )
        return await generator.generate(parameters)
