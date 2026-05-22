from httpx import ASGITransport, AsyncClient
from pytest import MonkeyPatch

from image_helpers import png_bytes
from video_ai.interfaces.dependencies import get_memory_job_repository
from video_ai.main import create_app


async def test_auth_enabled_requires_api_key(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("API_KEYS", "alice:alice-key")
    app = create_app()
    files = {"image": ("input.png", png_bytes(), "image/png")}
    data = {"prompt": "Make this image cinematic"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/generations", data=data, files=files)

    assert response.status_code == 401


async def test_generation_history_is_filtered_by_authenticated_user(
    monkeypatch: MonkeyPatch,
) -> None:
    get_memory_job_repository.cache_clear()
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("API_KEYS", "alice:alice-key,bob:bob-key")
    app = create_app()
    files_a = {"image": ("a.png", png_bytes(), "image/png")}
    files_b = {"image": ("b.png", png_bytes(), "image/png")}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        alice_create = await client.post(
            "/api/generations",
            data={"prompt": "Alice private cinematic image"},
            files=files_a,
            headers={"X-API-Key": "alice-key"},
        )
        bob_create = await client.post(
            "/api/generations",
            data={"prompt": "Bob private cinematic image"},
            files=files_b,
            headers={"X-API-Key": "bob-key"},
        )
        alice_list = await client.get("/api/generations", headers={"X-API-Key": "alice-key"})
        bob_get_alice = await client.get(
            f"/api/generations/{alice_create.json()['id']}",
            headers={"X-API-Key": "bob-key"},
        )

    assert alice_create.status_code == 202
    assert bob_create.status_code == 202
    assert alice_list.status_code == 200
    assert [job["prompt"] for job in alice_list.json()] == ["Alice private cinematic image"]
    assert bob_get_alice.status_code == 404


async def test_owner_can_access_own_job(monkeypatch: MonkeyPatch) -> None:
    get_memory_job_repository.cache_clear()
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("API_KEYS", "alice:alice-key")
    app = create_app()
    files = {"image": ("input.png", png_bytes(), "image/png")}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        created = await client.post(
            "/api/generations",
            data={"prompt": "Alice owned job"},
            files=files,
            headers={"X-API-Key": "alice-key"},
        )
        fetched = await client.get(
            f"/api/generations/{created.json()['id']}",
            headers={"X-API-Key": "alice-key"},
        )

    assert fetched.status_code == 200
    assert fetched.json()["prompt"] == "Alice owned job"
