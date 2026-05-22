"""Application service for job lifecycle management."""

from __future__ import annotations

from pathlib import Path

from video_ai.domain.models import GenerationRequest, ImageAsset, VideoGenerationJob
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
        )
        request = GenerationRequest(prompt=prompt, image=image, user_id=user_id)
        job = VideoGenerationJob(request=request)
        await self._repository.save(job)
        return job
