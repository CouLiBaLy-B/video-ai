"""Fallback AI adapters for resilient vLLM operation."""

from __future__ import annotations

from video_ai.domain.models import EnhancedPrompt, ImageAnalysis, ImageAsset
from video_ai.domain.ports import PromptEnhancer, VisionAnalyzer


class FallbackVisionAnalyzer:
    """Try a primary vision analyzer and fall back to another implementation."""

    def __init__(self, primary: VisionAnalyzer, fallback: VisionAnalyzer) -> None:
        self._primary = primary
        self._fallback = fallback
        self.last_used_provider = "primary"

    async def analyze(self, image: ImageAsset, prompt: str) -> ImageAnalysis:
        """Analyze with primary provider, falling back on runtime errors."""
        try:
            result = await self._primary.analyze(image, prompt)
            self.last_used_provider = "primary"
            return result
        except Exception:  # pragma: no cover - exact provider errors vary
            self.last_used_provider = "fallback"
            return await self._fallback.analyze(image, prompt)


class FallbackPromptEnhancer:
    """Try a primary prompt enhancer and fall back to another implementation."""

    def __init__(self, primary: PromptEnhancer, fallback: PromptEnhancer) -> None:
        self._primary = primary
        self._fallback = fallback
        self.last_used_provider = "primary"

    async def enhance(self, prompt: str, analysis: ImageAnalysis) -> EnhancedPrompt:
        """Enhance with primary provider, falling back on runtime errors."""
        try:
            result = await self._primary.enhance(prompt, analysis)
            self.last_used_provider = "primary"
            return result
        except Exception:  # pragma: no cover - exact provider errors vary
            self.last_used_provider = "fallback"
            return await self._fallback.enhance(prompt, analysis)
