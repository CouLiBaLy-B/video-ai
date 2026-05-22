"""Deterministic AI adapters used for local mock workflows."""

from video_ai.domain.models import EnhancedPrompt, ImageAnalysis, ImageAsset


class SimpleVisionAnalyzer:
    """Minimal vision analyzer for offline development."""

    async def analyze(self, image: ImageAsset, prompt: str) -> ImageAnalysis:
        """Return a generic image analysis without calling a model."""
        _ = image
        return ImageAnalysis(
            subject="main subject from input image",
            scene="uploaded image scene",
            style="cinematic",
            lighting="natural",
            composition="centered composition",
            motion_suggestions=["slow cinematic camera movement"],
            constraints=["preserve the identity and structure of the input image"],
        )


class SimplePromptEnhancer:
    """Minimal prompt enhancer for offline development."""

    async def enhance(self, prompt: str, analysis: ImageAnalysis) -> EnhancedPrompt:
        """Create a deterministic cinematic prompt."""
        return EnhancedPrompt(
            positive=(
                f"{prompt}. {analysis.subject} in {analysis.scene}, cinematic lighting, "
                "smooth realistic motion, stable identity, high detail"
            ),
            negative="low quality, blurry, distorted anatomy, flicker, warped subject",
            camera_motion="slow dolly in",
            style=analysis.style,
        )
