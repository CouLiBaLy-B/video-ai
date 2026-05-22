"""Smoke test real vLLM text and vision endpoints.

This script intentionally fails fast when the configured endpoints are not
reachable. It is meant to be run on an environment where vLLM servers are
already running.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps/api/src"))
from tempfile import NamedTemporaryFile

from PIL import Image

from video_ai.config.settings import Settings
from video_ai.domain.models import ImageAnalysis, ImageAsset
from video_ai.infrastructure.vllm import (
    VllmChatGateway,
    VllmHealthChecker,
    VllmPromptEnhancer,
    VllmVisionAnalyzer,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Smoke test configured vLLM endpoints")
    parser.add_argument("--image", type=Path, default=None, help="Optional image for VLM test")
    parser.add_argument("--prompt", default="A cinematic slow camera move around the subject")
    parser.add_argument("--skip-vision", action="store_true", help="Only test text endpoint")
    return parser.parse_args()


async def main() -> int:
    """Run the vLLM smoke test."""
    args = parse_args()
    settings = Settings(ai_provider="vllm", vllm_fallback_to_mock=False)

    text_ok = await VllmHealthChecker(
        base_url=settings.vllm_text_base_url,
        api_key=settings.vllm_api_key,
    ).check()
    if not text_ok:
        raise RuntimeError(f"Text vLLM endpoint unavailable: {settings.vllm_text_base_url}")

    text_gateway = VllmChatGateway(
        base_url=settings.vllm_text_base_url,
        api_key=settings.vllm_api_key,
        model=settings.vllm_text_model,
        timeout_seconds=settings.vllm_timeout_seconds,
    )
    enhanced = await VllmPromptEnhancer(text_gateway).enhance(
        args.prompt,
        ImageAnalysis(subject="test subject", scene="test scene"),
    )
    print("TEXT_OK", enhanced.model_dump_json())

    if args.skip_vision:
        return 0

    vision_ok = await VllmHealthChecker(
        base_url=settings.vllm_vision_base_url,
        api_key=settings.vllm_api_key,
    ).check()
    if not vision_ok:
        raise RuntimeError(f"Vision vLLM endpoint unavailable: {settings.vllm_vision_base_url}")

    image_path = args.image or _create_temp_image()
    image = ImageAsset(
        path=image_path,
        mime_type="image/png",
        size_bytes=image_path.stat().st_size,
    )
    vision_gateway = VllmChatGateway(
        base_url=settings.vllm_vision_base_url,
        api_key=settings.vllm_api_key,
        model=settings.vllm_vision_model,
        timeout_seconds=settings.vllm_timeout_seconds,
    )
    analysis = await VllmVisionAnalyzer(vision_gateway).analyze(image, args.prompt)
    print("VISION_OK", analysis.model_dump_json())
    return 0


def _create_temp_image() -> Path:
    image = Image.new("RGB", (64, 64), color=(64, 96, 160))
    handle = NamedTemporaryFile(suffix=".png", delete=False)
    image.save(handle.name, format="PNG")
    return Path(handle.name)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
