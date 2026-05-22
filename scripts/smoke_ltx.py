"""Smoke test LTX-Video parameter validation and optional real GPU generation."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps/api/src"))
from tempfile import NamedTemporaryFile

from PIL import Image

from video_ai.config.settings import Settings
from video_ai.domain.models import EnhancedPrompt, GenerationParameters, ImageAsset
from video_ai.infrastructure.ltx_video import (
    LTX_VIDEO_2B_DISTILLED_PROFILE,
    LtxVideoGenerator,
    LtxVideoParameterValidator,
)
from video_ai.storage.local import LocalStorageService


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Smoke test LTX-Video")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--prompt", default="cinematic slow camera move")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path(".data/smoke-ltx"))
    parser.add_argument("--run", action="store_true", help="Actually load model and generate video")
    return parser.parse_args()


async def main() -> int:
    """Run validation and optionally real LTX generation."""
    args = parse_args()
    settings = Settings(video_generator_backend="ltx-video")
    validation = LtxVideoParameterValidator().validate(
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        fps=args.fps,
        inference_steps=args.steps,
        guidance_scale=args.guidance,
    )
    print("VALIDATION", validation)
    if not validation.valid:
        return 2
    if not args.run:
        print("DRY_RUN_OK pass --run to execute real GPU generation")
        return 0

    image_path = args.image or _create_temp_image()
    image = ImageAsset(
        path=image_path,
        mime_type="image/png",
        size_bytes=image_path.stat().st_size,
    )
    params = GenerationParameters(
        prompt=EnhancedPrompt(positive=args.prompt),
        image=image,
        model=LTX_VIDEO_2B_DISTILLED_PROFILE.model_copy(
            update={"id": settings.ltx_video_model_id}
        ),
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        fps=args.fps,
        seed=args.seed,
        guidance_scale=args.guidance,
        inference_steps=args.steps,
    )
    generator = LtxVideoGenerator(
        storage=LocalStorageService(args.output_dir),
        model_id=settings.ltx_video_model_id,
        torch_dtype=settings.ltx_video_torch_dtype,
        device=settings.ltx_video_device,
    )
    video = await generator.generate(params)
    print("VIDEO_OK", video.model_dump_json())
    return 0


def _create_temp_image() -> Path:
    image = Image.new("RGB", (256, 256), color=(64, 96, 160))
    handle = NamedTemporaryFile(suffix=".png", delete=False)
    image.save(handle.name, format="PNG")
    return Path(handle.name)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
