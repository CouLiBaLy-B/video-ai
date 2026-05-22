"""Application service for job lifecycle management."""

from __future__ import annotations

from pathlib import Path

from video_ai.domain.enums import VideoBackend
from video_ai.domain.models import (
    GenerationPreferences,
    GenerationRequest,
    ImageAsset,
    VideoGenerationJob,
)
from video_ai.domain.ports import JobRepository, StorageService


class JobApplicationService:
    """Create and retrieve generation jobs."""

    def __init__(self, repository: JobRepository, storage: StorageService) -> None:
        self._repository = repository
        self._storage = storage

    async def create_generation_job(
        self,
        *,
        prompt: str,
        image_content: bytes,
        image_filename: str,
        image_mime_type: str,
        image_width: int | None = None,
        image_height: int | None = None,
        requested_backend: VideoBackend | None = None,
        width: int | None = None,
        height: int | None = None,
        num_frames: int | None = None,
        fps: int | None = None,
        seed: int | None = None,
        guidance_scale: float | None = None,
        inference_steps: int | None = None,
        user_id: str | None = None,
    ) -> VideoGenerationJob:
        """Create a queued generation job from user input."""
        storage_ref = await self._storage.save_bytes(
            image_content,
            image_filename,
            image_mime_type,
        )
        image = ImageAsset(
            path=storage_ref.path or Path(storage_ref.uri),
            mime_type=image_mime_type,
            size_bytes=storage_ref.size_bytes,
            width=image_width,
            height=image_height,
        )
        preferences = GenerationPreferences(
            requested_backend=requested_backend,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            seed=seed,
            guidance_scale=guidance_scale,
            inference_steps=inference_steps,
        )
        request = GenerationRequest(
            prompt=prompt, image=image, preferences=preferences, user_id=user_id
        )
        job = VideoGenerationJob(request=request)
        await self._repository.save(job)
        return job
