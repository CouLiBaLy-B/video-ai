from pathlib import Path

import pytest

from video_ai.domain.models import EnhancedPrompt, ImageAnalysis, ImageAsset
from video_ai.infrastructure.fallback_ai import FallbackPromptEnhancer, FallbackVisionAnalyzer
from video_ai.infrastructure.simple_ai import SimplePromptEnhancer, SimpleVisionAnalyzer


class FailingVisionAnalyzer:
    async def analyze(self, image: ImageAsset, prompt: str) -> ImageAnalysis:
        _ = image, prompt
        raise RuntimeError("vision unavailable")


class FailingPromptEnhancer:
    async def enhance(self, prompt: str, analysis: ImageAnalysis) -> EnhancedPrompt:
        _ = prompt, analysis
        raise RuntimeError("prompt unavailable")


async def test_fallback_vision_analyzer_uses_fallback(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"image")
    analyzer = FallbackVisionAnalyzer(FailingVisionAnalyzer(), SimpleVisionAnalyzer())

    analysis = await analyzer.analyze(
        ImageAsset(path=image_path, mime_type="image/png", size_bytes=5), "cat"
    )

    assert analysis.subject == "main subject from input image"
    assert analyzer.last_used_provider == "fallback"


async def test_fallback_prompt_enhancer_uses_fallback() -> None:
    enhancer = FallbackPromptEnhancer(FailingPromptEnhancer(), SimplePromptEnhancer())

    prompt = await enhancer.enhance("cat", ImageAnalysis(subject="cat", scene="room"))

    assert "cat" in prompt.positive
    assert enhancer.last_used_provider == "fallback"


async def test_fallback_can_be_disabled_by_using_primary_directly(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"image")

    with pytest.raises(RuntimeError):
        await FailingVisionAnalyzer().analyze(
            ImageAsset(path=image_path, mime_type="image/png", size_bytes=5), "cat"
        )
