from httpx import ASGITransport, AsyncClient

from video_ai.main import create_app


async def test_health_endpoint() -> None:
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_create_and_get_generation_job() -> None:
    app = create_app()
    files = {"image": ("input.png", b"fake-png-bytes", "image/png")}
    data = {"prompt": "Make this image cinematic"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        create_response = await client.post("/api/generations", data=data, files=files)
        assert create_response.status_code == 202
        payload = create_response.json()
        assert payload["status"] == "queued"

        get_response = await client.get(f"/api/generations/{payload['id']}")
        assert get_response.status_code == 200
        assert get_response.json()["id"] == payload["id"]


async def test_create_generation_rejects_unsupported_image() -> None:
    app = create_app()
    files = {"image": ("input.gif", b"gif", "image/gif")}
    data = {"prompt": "Make this image cinematic"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/generations", data=data, files=files)

    assert response.status_code == 415
