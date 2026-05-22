from httpx import ASGITransport, AsyncClient

from image_helpers import png_bytes
from video_ai.main import create_app


async def test_health_endpoint() -> None:
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_system_capabilities_endpoint() -> None:
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/system/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert "mock" in payload["available_video_backends"]
    assert "ltx-video" in payload["available_video_backends"]


async def test_system_health_endpoint() -> None:
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/system/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["api"]["status"] == "ok"
    assert payload["storage"]["status"] == "ok"
    assert payload["job_repository"]["status"] == "ok"
    assert payload["vllm_text"]["status"] in {"ok", "unavailable"}


async def test_ltx_validation_endpoint() -> None:
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/system/ltx/validate",
            json={"width": 777, "height": 512, "num_frames": 120},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is False
    assert "width" in payload["recommended"]


async def test_create_get_and_download_generation_job() -> None:
    app = create_app()
    files = {"image": ("input.png", png_bytes(), "image/png")}
    data = {"prompt": "Make this image cinematic"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        create_response = await client.post("/api/generations", data=data, files=files)
        assert create_response.status_code == 202
        payload = create_response.json()

        get_response = await client.get(f"/api/generations/{payload['id']}")
        assert get_response.status_code == 200
        fetched = get_response.json()
        assert fetched["id"] == payload["id"]
        assert fetched["status"] == "completed"
        assert fetched["video_url"] is not None

        list_response = await client.get("/api/generations")
        assert list_response.status_code == 200
        assert any(item["id"] == payload["id"] for item in list_response.json())

        video_response = await client.get(f"/api/generations/{payload['id']}/video")
        assert video_response.status_code == 200
        assert video_response.content.startswith(b"MOCK_MP4")


async def test_create_generation_rejects_unsupported_image() -> None:
    app = create_app()
    files = {"image": ("input.gif", b"gif", "image/gif")}
    data = {"prompt": "Make this image cinematic"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/generations", data=data, files=files)

    assert response.status_code == 415


async def test_create_generation_accepts_generation_parameters() -> None:
    app = create_app()
    files = {"image": ("input.png", png_bytes(), "image/png")}
    data = {
        "prompt": "Make this image cinematic",
        "requested_backend": "mock",
        "width": "320",
        "height": "240",
        "num_frames": "24",
        "fps": "12",
        "seed": "42",
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        create_response = await client.post("/api/generations", data=data, files=files)
        assert create_response.status_code == 202
        payload = create_response.json()
        video_response = await client.get(f"/api/generations/{payload['id']}/video")

    assert b'"width": 320' in video_response.content
    assert b'"height": 240' in video_response.content
    assert b'"fps": 12' in video_response.content


async def test_ltx_generation_waits_for_human_approval_and_can_be_rejected() -> None:
    app = create_app()
    files = {"image": ("input.png", png_bytes(), "image/png")}
    data = {
        "prompt": "Make this image cinematic",
        "requested_backend": "ltx-video",
        "width": "768",
        "height": "512",
        "num_frames": "121",
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        create_response = await client.post("/api/generations", data=data, files=files)
        assert create_response.status_code == 202
        payload = create_response.json()
        get_response = await client.get(f"/api/generations/{payload['id']}")
        payload = get_response.json()
        assert payload["status"] == "waiting_for_approval"
        assert payload["video_url"] is None

        reject_response = await client.post(f"/api/generations/{payload['id']}/reject")
        assert reject_response.status_code == 200
        assert reject_response.json()["status"] == "cancelled"


async def test_approve_generation_rejects_jobs_not_waiting_for_approval() -> None:
    app = create_app()
    files = {"image": ("input.png", png_bytes(), "image/png")}
    data = {"prompt": "Make this image cinematic", "requested_backend": "mock"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        create_response = await client.post("/api/generations", data=data, files=files)
        payload = create_response.json()
        approve_response = await client.post(f"/api/generations/{payload['id']}/approve")

    assert approve_response.status_code == 409
