from pathlib import Path

from video_ai.domain.enums import VideoBackend
from video_ai.domain.models import EnhancedPrompt, GenerationParameters, GenerationRequest, ImageAsset
from video_ai.infrastructure.video import (
    HeuristicQualityReviewer,
    MockVideoGenerator,
    StaticVideoModelRouter,
)
from video_ai.storage.local import LocalStorageService


def make_image(tmp_path: Path) -> ImageAsset:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"image")
    return ImageAsset(path=image_path, mime_type="image/png", size_bytes=5)


async def test_static_router_returns_mock_profile(tmp_path: Path) -> None:
    router = StaticVideoModelRouter()
    request = GenerationRequest(prompt="cinematic cat", image=make_image(tmp_path))

    profile = await router.select_model(request)

    assert profile.backend == VideoBackend.MOCK
    assert profile.min_vram_gb == 0


async def test_mock_video_generator_saves_artifact(tmp_path: Path) -> None:
    storage = LocalStorageService(tmp_path / "storage")
    router = StaticVideoModelRouter()
    request = GenerationRequest(prompt="cinematic cat", image=make_image(tmp_path))
    model = await router.select_model(request)
    params = GenerationParameters(
        prompt=EnhancedPrompt(positive="cinematic cat"),
        image=request.image,
        model=model,
        width=320,
        height=240,
        num_frames=24,
        fps=24,
        seed=123,
    )
    generator = MockVideoGenerator(storage)

    video = await generator.generate(params)

    assert video.duration_seconds == 1.0
    assert video.storage.path is not None
    assert video.storage.path.exists()
    assert video.storage.path.read_bytes().startswith(b"MOCK_MP4")


async def test_heuristic_quality_reviewer_accepts_good_mock(tmp_path: Path) -> None:
    storage = LocalStorageService(tmp_path / "storage")
    request = GenerationRequest(prompt="cinematic cat", image=make_image(tmp_path))
    model = await StaticVideoModelRouter().select_model(request)
    params = GenerationParameters(
        prompt=EnhancedPrompt(positive="cinematic cat, slow dolly"),
        image=request.image,
        model=model,
        width=320,
        height=240,
        num_frames=48,
        fps=24,
    )
    video = await MockVideoGenerator(storage).generate(params)

    report = await HeuristicQualityReviewer().review(request, video, params)

    assert report.accepted is True
    assert report.score == 0.9
