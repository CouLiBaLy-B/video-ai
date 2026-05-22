from pathlib import Path

import pytest

from video_ai.domain.enums import VideoBackend
from video_ai.domain.models import EnhancedPrompt, GenerationParameters, ImageAsset
from video_ai.infrastructure.ltx_video import (
    LTX_VIDEO_2B_DISTILLED_PROFILE,
    LtxValidationResult,
    LtxVideoGenerator,
    LtxVideoParameterValidator,
)
from video_ai.storage.local import LocalStorageService


def test_ltx_profile_is_image_to_video() -> None:
    assert LTX_VIDEO_2B_DISTILLED_PROFILE.backend == VideoBackend.LTX_VIDEO
    assert LTX_VIDEO_2B_DISTILLED_PROFILE.supports_image_to_video is True
    assert LTX_VIDEO_2B_DISTILLED_PROFILE.min_vram_gb >= 1


def test_ltx_generator_is_lazy(tmp_path: Path) -> None:
    generator = LtxVideoGenerator(storage=LocalStorageService(tmp_path), device="cpu")

    assert generator is not None


def test_ltx_validator_accepts_safe_defaults() -> None:
    result = LtxVideoParameterValidator().validate(
        width=768,
        height=512,
        num_frames=121,
        fps=24,
        inference_steps=30,
        guidance_scale=3.5,
    )

    assert result == LtxValidationResult(valid=True)


def test_ltx_validator_recommends_aligned_dimensions_and_frames() -> None:
    result = LtxVideoParameterValidator().validate(
        width=777,
        height=513,
        num_frames=120,
        fps=60,
        inference_steps=2,
        guidance_scale=12,
    )

    assert result.valid is False
    assert "width" in result.recommended
    assert "height" in result.recommended
    assert "num_frames" in result.recommended
    assert result.warnings


def test_ltx_generator_maps_parameters_without_loading_pipeline(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"image")
    image = ImageAsset(path=image_path, mime_type="image/png", size_bytes=5)
    generator = LtxVideoGenerator(storage=LocalStorageService(tmp_path), device="cpu")
    parameters = GenerationParameters(
        prompt=EnhancedPrompt(positive="cat", negative="blur"),
        image=image,
        model=LTX_VIDEO_2B_DISTILLED_PROFILE,
        width=768,
        height=512,
        num_frames=121,
        fps=24,
        seed=None,
        guidance_scale=4.0,
        inference_steps=20,
        extra={"decode_timestep": 0.03, "unsafe": "ignored"},
    )

    kwargs = generator._build_inference_kwargs(parameters, image="loaded-image")  # noqa: SLF001

    assert kwargs["image"] == "loaded-image"
    assert kwargs["prompt"] == "cat"
    assert kwargs["negative_prompt"] == "blur"
    assert kwargs["num_inference_steps"] == 20
    assert kwargs["decode_timestep"] == 0.03
    assert "unsafe" not in kwargs
    assert "generator" not in kwargs


def test_ltx_generator_rejects_invalid_parameters_before_loading_pipeline(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"image")
    image = ImageAsset(path=image_path, mime_type="image/png", size_bytes=5)
    parameters = GenerationParameters(
        prompt=EnhancedPrompt(positive="cat"),
        image=image,
        model=LTX_VIDEO_2B_DISTILLED_PROFILE,
        width=777,
        height=512,
    )
    generator = LtxVideoGenerator(storage=LocalStorageService(tmp_path), device="cpu")

    with pytest.raises(ValueError, match="Invalid LTX-Video parameters"):
        import asyncio

        asyncio.run(generator.generate(parameters))
