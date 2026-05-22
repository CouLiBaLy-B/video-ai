"""Request-aware video model routing."""

from __future__ import annotations

from video_ai.domain.enums import VideoBackend
from video_ai.domain.models import GenerationRequest, ModelProfile


class RequestAwareVideoModelRouter:
    """Select a model profile from request preferences and configured defaults."""

    def __init__(
        self, *, default_backend: VideoBackend, profiles: dict[VideoBackend, ModelProfile]
    ) -> None:
        self._default_backend = default_backend
        self._profiles = profiles

    async def select_model(self, request: GenerationRequest) -> ModelProfile:
        """Select the requested backend when available, otherwise default."""
        backend = request.preferences.requested_backend or self._default_backend
        if backend not in self._profiles:
            backend = self._default_backend
        return self._profiles[backend]
