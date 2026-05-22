from video_ai.domain.enums import VideoBackend
from video_ai.infrastructure.ltx_video import LTX_VIDEO_2B_DISTILLED_PROFILE, LtxVideoGenerator
from video_ai.storage.local import LocalStorageService


def test_ltx_profile_is_image_to_video() -> None:
    assert LTX_VIDEO_2B_DISTILLED_PROFILE.backend == VideoBackend.LTX_VIDEO
    assert LTX_VIDEO_2B_DISTILLED_PROFILE.supports_image_to_video is True
    assert LTX_VIDEO_2B_DISTILLED_PROFILE.min_vram_gb >= 1


def test_ltx_generator_is_lazy(tmp_path) -> None:  # type: ignore[no-untyped-def]
    generator = LtxVideoGenerator(storage=LocalStorageService(tmp_path), device="cpu")

    assert generator is not None
