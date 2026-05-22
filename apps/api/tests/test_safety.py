from httpx import ASGITransport, AsyncClient
from pytest import MonkeyPatch

from image_helpers import png_bytes
from video_ai.application.safety import SafetyDecision, SafetyPolicy, SafetyService, parse_terms
from video_ai.main import create_app


def test_safety_service_allows_normal_prompt() -> None:
    service = SafetyService(
        SafetyPolicy(enabled=True, blocked_terms=("blocked",), review_terms=("review",))
    )

    result = service.evaluate_prompt("a cinematic cat")

    assert result.decision == SafetyDecision.ALLOWED


def test_safety_service_blocks_configured_term() -> None:
    service = SafetyService(
        SafetyPolicy(enabled=True, blocked_terms=("bomb making",), review_terms=())
    )

    result = service.evaluate_prompt("teach bomb making")

    assert result.decision == SafetyDecision.BLOCKED
    assert result.reasons


def test_safety_service_flags_review_term() -> None:
    service = SafetyService(SafetyPolicy(enabled=True, blocked_terms=(), review_terms=("weapon",)))

    result = service.evaluate_prompt("cinematic weapon on a table")

    assert result.decision == SafetyDecision.NEEDS_REVIEW


def test_parse_terms_normalizes_terms() -> None:
    assert parse_terms(" Weapon ,  Blood ") == ("weapon", "blood")


async def test_api_blocks_unsafe_prompt(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("SAFETY_ENABLED", "true")
    monkeypatch.setenv("SAFETY_BLOCKED_TERMS", "forbidden phrase")
    app = create_app()
    files = {"image": ("input.png", png_bytes(), "image/png")}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/generations",
            data={"prompt": "this contains forbidden phrase"},
            files=files,
        )

    assert response.status_code == 403
    assert "blocked term" in response.json()["detail"]


async def test_api_flags_review_prompt(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("SAFETY_ENABLED", "true")
    monkeypatch.setenv("SAFETY_REVIEW_TERMS", "reviewme")
    app = create_app()
    files = {"image": ("input.png", png_bytes(), "image/png")}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/generations",
            data={"prompt": "this contains reviewme"},
            files=files,
        )

    assert response.status_code == 409
    assert "manual review" in response.json()["detail"]
