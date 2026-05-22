from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from pytest import MonkeyPatch

from image_helpers import png_bytes
from video_ai.application.quotas import QuotaExceededError, QuotaLimits, QuotaService
from video_ai.domain.enums import JobStatus
from video_ai.domain.models import GenerationRequest, ImageAsset, VideoGenerationJob
from video_ai.interfaces.dependencies import get_memory_job_repository
from video_ai.main import create_app
from video_ai.storage.memory import InMemoryJobRepository


def make_job(tmp_path: Path, user_id: str = "alice") -> VideoGenerationJob:
    image_path = tmp_path / f"{user_id}.png"
    image_path.write_bytes(b"image")
    return VideoGenerationJob(
        request=GenerationRequest(
            prompt="cinematic cat",
            image=ImageAsset(path=image_path, mime_type="image/png", size_bytes=5),
            user_id=user_id,
        )
    )


def limits(active: int = 1, daily: int = 10) -> QuotaLimits:
    return QuotaLimits(
        max_active_jobs_per_user=active,
        max_daily_jobs_per_user=daily,
        max_generation_width=512,
        max_generation_height=512,
        max_generation_frames=49,
    )


async def test_quota_service_rejects_too_many_active_jobs(tmp_path: Path) -> None:
    repository = InMemoryJobRepository()
    active_job = make_job(tmp_path).transition(JobStatus.QUEUED, "queued")
    await repository.save(active_job)

    with pytest.raises(QuotaExceededError, match="active jobs"):
        await QuotaService(repository, limits(active=1)).validate_create(
            user_id="alice", width=512, height=512, num_frames=49
        )


async def test_quota_service_rejects_generation_bounds() -> None:
    repository = InMemoryJobRepository()
    service = QuotaService(repository, limits())

    with pytest.raises(QuotaExceededError, match="width"):
        await service.validate_create(user_id="alice", width=1024, height=512, num_frames=49)

    with pytest.raises(QuotaExceededError, match="frame"):
        await service.validate_create(user_id="alice", width=512, height=512, num_frames=121)


async def test_api_rejects_generation_above_configured_bounds(monkeypatch: MonkeyPatch) -> None:
    get_memory_job_repository.cache_clear()
    monkeypatch.setenv("MAX_GENERATION_WIDTH", "256")
    app = create_app()
    files = {"image": ("input.png", png_bytes(), "image/png")}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/generations",
            data={"prompt": "too wide", "width": "512"},
            files=files,
        )

    assert response.status_code == 403
    assert "width" in response.json()["detail"]


async def test_api_rejects_user_active_quota(monkeypatch: MonkeyPatch) -> None:
    get_memory_job_repository.cache_clear()
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("API_KEYS", "alice:alice-key")
    monkeypatch.setenv("MAX_ACTIVE_JOBS_PER_USER", "1")
    app = create_app()
    files = {"image": ("input.png", png_bytes(), "image/png")}
    data = {"prompt": "gpu approval", "requested_backend": "ltx-video"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        first = await client.post(
            "/api/generations", data=data, files=files, headers={"X-API-Key": "alice-key"}
        )
        second = await client.post(
            "/api/generations", data=data, files=files, headers={"X-API-Key": "alice-key"}
        )

    assert first.status_code == 202
    assert second.status_code == 403
    assert "active jobs" in second.json()["detail"]
