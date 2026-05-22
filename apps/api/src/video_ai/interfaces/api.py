"""FastAPI routes."""

from __future__ import annotations

import asyncio
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, StreamingResponse

from video_ai.application.jobs import JobApplicationService
from video_ai.application.orchestrator import VideoGenerationOrchestrator
from video_ai.config.settings import get_settings
from video_ai.domain.enums import JobStatus
from video_ai.interfaces.dependencies import get_job_repository, get_job_service, get_orchestrator
from video_ai.interfaces.schemas import JobResponse
from video_ai.storage.memory import InMemoryJobRepository

router = APIRouter(prefix="/api", tags=["generations"])


@router.post(
    "/generations",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_generation(
    background_tasks: BackgroundTasks,
    prompt: str = Form(..., min_length=3),
    image: UploadFile = File(...),
    service: JobApplicationService = Depends(get_job_service),
    orchestrator: VideoGenerationOrchestrator = Depends(get_orchestrator),
) -> JobResponse:
    """Create and asynchronously process a text+image to video generation job."""
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
    background_tasks.add_task(_run_orchestrator_safely, orchestrator, job.id)
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
        previous_status: JobStatus | None = None
        for _ in range(120):
            job = await repository.get(job_id)
            if job is None:
                yield "event: error\ndata: Generation job not found\n\n"
                return
            if job.status != previous_status:
                yield f"event: status\ndata: {job.status.value}\n\n"
                previous_status = job.status
            if job.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
                return
            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/generations/{job_id}/video")
async def get_generation_video(
    job_id: UUID,
    repository: InMemoryJobRepository = Depends(get_job_repository),
) -> FileResponse:
    """Download a generated video artifact."""
    job = await repository.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Generation job not found")
    if job.video is None or job.video.storage.path is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Generated video not available")
    return FileResponse(
        path=job.video.storage.path,
        media_type=job.video.storage.mime_type,
        filename=f"{job.id}.mp4",
    )


async def _run_orchestrator_safely(
    orchestrator: VideoGenerationOrchestrator,
    job_id: UUID,
) -> None:
    repository = get_job_repository()
    job = await repository.get(job_id)
    if job is None:
        return
    try:
        await orchestrator.run(job)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        failed = job.transition(JobStatus.FAILED, f"Generation failed: {exc}")
        await repository.save(failed)
