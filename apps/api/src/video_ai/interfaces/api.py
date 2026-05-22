"""FastAPI routes."""

from __future__ import annotations

import asyncio
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from video_ai.application.jobs import JobApplicationService
from video_ai.config.settings import get_settings
from video_ai.interfaces.dependencies import get_job_repository, get_job_service
from video_ai.interfaces.schemas import JobResponse
from video_ai.storage.memory import InMemoryJobRepository

router = APIRouter(prefix="/api", tags=["generations"])


@router.post(
    "/generations",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_generation(
    prompt: str = Form(..., min_length=3),
    image: UploadFile = File(...),
    service: JobApplicationService = Depends(get_job_service),
) -> JobResponse:
    """Create a queued text+image to video generation job."""
    settings = get_settings()
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Unsupported image type")
    content = await image.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Image is too large")
    job = await service.create_generation_job(
        prompt=prompt,
        image_content=content,
        image_filename=image.filename or "input.png",
        image_mime_type=image.content_type,
    )
    return JobResponse.from_job(job)


@router.get("/generations/{job_id}", response_model=JobResponse)
async def get_generation(
    job_id: UUID,
    repository: InMemoryJobRepository = Depends(get_job_repository),
) -> JobResponse:
    """Return a generation job."""
    job = await repository.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Generation job not found")
    return JobResponse.from_job(job)


@router.get("/generations/{job_id}/events")
async def stream_generation_events(
    job_id: UUID,
    repository: InMemoryJobRepository = Depends(get_job_repository),
) -> StreamingResponse:
    """Stream job status events as Server-Sent Events."""

    async def event_stream() -> object:
        job = await repository.get(job_id)
        if job is None:
            yield "event: error\ndata: Generation job not found\n\n"
            return
        yield f"event: status\ndata: {job.status.value}\n\n"
        await asyncio.sleep(0)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
