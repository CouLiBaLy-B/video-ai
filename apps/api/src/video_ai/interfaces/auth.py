"""Authentication and ownership helpers."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Header, HTTPException, status

from video_ai.config.settings import get_settings
from video_ai.domain.models import VideoGenerationJob


@dataclass(frozen=True)
class UserContext:
    """Authenticated or anonymous user context."""

    user_id: str | None
    authenticated: bool
    auth_enabled: bool


def parse_api_keys(raw: str) -> dict[str, str]:
    """Parse `user_id:api_key,user_id_2:api_key_2` config."""
    result: dict[str, str] = {}
    for item in raw.split(","):
        if not item.strip() or ":" not in item:
            continue
        user_id, api_key = item.split(":", maxsplit=1)
        user_id = user_id.strip()
        api_key = api_key.strip()
        if user_id and api_key:
            result[api_key] = user_id
    return result


async def get_current_user(x_api_key: str | None = Header(default=None)) -> UserContext:
    """Resolve current user from X-API-Key when auth is enabled."""
    settings = get_settings()
    if not settings.auth_enabled:
        return UserContext(user_id=None, authenticated=False, auth_enabled=False)

    if not x_api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing X-API-Key")

    user_id = parse_api_keys(settings.api_keys).get(x_api_key)
    if user_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")
    return UserContext(user_id=user_id, authenticated=True, auth_enabled=True)


def ensure_job_access(job: VideoGenerationJob, user: UserContext) -> None:
    """Raise when the current user cannot access a job."""
    if not user.auth_enabled:
        return
    if job.request.user_id != user.user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Generation job not found")
