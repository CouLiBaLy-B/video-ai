from video_ai.config.settings import Settings
from video_ai.domain.models import StorageRef
from video_ai.storage.urls import resolve_download_url


def test_resolve_download_url_returns_public_http_uri() -> None:
    ref = StorageRef(uri="https://cdn.test/video.mp4", mime_type="video/mp4", size_bytes=10)

    assert resolve_download_url(ref, Settings()) == "https://cdn.test/video.mp4"
