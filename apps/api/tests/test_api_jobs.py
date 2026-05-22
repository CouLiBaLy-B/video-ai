from httpx import ASGITransport, AsyncClient

from image_helpers import png_bytes
from video_ai.main import create_app


async def test_health_endpoint() -> None:
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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
