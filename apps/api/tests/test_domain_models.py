from pathlib import Path

import pytest
from pydantic import ValidationError

from video_ai.domain.enums import JobStatus, VideoBackend
from video_ai.domain.models import (
    EnhancedPrompt,
    GenerationParameters,
    GenerationRequest,
    ImageAsset,
    ModelProfile,
)


def make_image() -> ImageAsset:
    return ImageAsset(path=Path("input.png"), mime_type="image/png", size_bytes=10)


def test_generation_request_normalizes_prompt() -> None:
    request = GenerationRequest(prompt="  cinematic   cat   shot  ", image=make_image())

    assert request.prompt == "cinematic cat shot"


def test_image_asset_rejects_invalid_mime_type() -> None:
    with pytest.raises(ValidationError):
        ImageAsset(path=Path("bad.gif"), mime_type="image/gif", size_bytes=10)


def test_job_transition_is_immutable() -> None:
    request = GenerationRequest(prompt="cinematic cat shot", image=make_image())
    job = __import__("video_ai.domain.models", fromlist=["VideoGenerationJob"]).VideoGenerationJob(
        request=request
    )

    updated = job.transition(JobStatus.ANALYZING, "vision analysis started")

    assert job.status == JobStatus.QUEUED
    assert updated.status == JobStatus.ANALYZING
    assert updated.status_reason == "vision analysis started"


def test_generation_parameters_accept_valid_profile() -> None:
    image = make_image()
    profile = ModelProfile(
        id="mock",
        backend=VideoBackend.MOCK,
        display_name="Mock Generator",
        min_vram_gb=0,
        default_width=512,
        default_height=512,
    )

    params = GenerationParameters(
        prompt=EnhancedPrompt(positive="a cinematic cat"),
        image=image,
        model=profile,
        width=512,
        height=512,
    )

    assert params.model.backend == VideoBackend.MOCK
    assert params.num_frames == 121
