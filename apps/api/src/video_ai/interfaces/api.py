"""FastAPI routes."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
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
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse

from video_ai.application.jobs import JobApplicationService
from video_ai.application.metrics import MetricsService
from video_ai.application.orchestrator import VideoGenerationOrchestrator
from video_ai.config.settings import get_settings
from video_ai.domain.enums import JobStatus, VideoBackend
from video_ai.domain.models import VideoGenerationJob
from video_ai.domain.ports import JobRepository
from video_ai.infrastructure.image_validation import ImageValidationError, ImageValidationService
from video_ai.infrastructure.ltx_video import LtxVideoParameterValidator
from video_ai.infrastructure.vllm import VllmHealthChecker
from video_ai.interfaces.auth import UserContext, ensure_job_access, get_current_user
from video_ai.interfaces.dependencies import (
    create_job_task_queue,
    get_job_repository,
    get_job_service,
    get_orchestrator,
)
from video_ai.interfaces.schemas import (
    ComponentHealthResponse,
    JobMetricsResponse,
    JobResponse,
    LtxValidationRequest,
    LtxValidationResponse,
    SystemCapabilitiesResponse,
    SystemHealthResponse,
)
from video_ai.storage.urls import StorageUrlResolutionError, resolve_download_url

router = APIRouter(prefix="/api", tags=["generations"])


@router.get("/system/capabilities", response_model=SystemCapabilitiesResponse)
async def get_system_capabilities() -> SystemCapabilitiesResponse:
    """Return configured AI and video backend capabilities."""
    settings = get_settings()
    return SystemCapabilitiesResponse(
        agent_planner_provider=settings.agent_planner_provider,
        ai_provider=settings.ai_provider,
        default_video_backend=settings.video_generator_backend,
        task_queue_backend=settings.task_queue_backend,
        available_video_backends=[VideoBackend.MOCK.value, VideoBackend.LTX_VIDEO.value],
        planned_video_backends=[VideoBackend.WAN_I2V.value],
        text_model=settings.vllm_text_model,
        vision_model=settings.vllm_vision_model,
        vllm_fallback_to_mock=settings.vllm_fallback_to_mock,
    )


@router.get("/system/health", response_model=SystemHealthResponse)
async def get_system_health(
    repository: JobRepository = Depends(get_job_repository),
) -> SystemHealthResponse:
    """Return health for API, storage, repository and configured vLLM endpoints."""
    settings = get_settings()
    storage_status = ComponentHealthResponse(status="ok")
    try:
        settings.storage_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        storage_status = ComponentHealthResponse(status="unavailable", detail=str(exc))

    repository_status = ComponentHealthResponse(status="ok")
    try:
        await repository.list_all()
    except Exception as exc:  # pragma: no cover - defensive health endpoint
        repository_status = ComponentHealthResponse(status="unavailable", detail=str(exc))

    text_ok = await VllmHealthChecker(
        base_url=settings.vllm_text_base_url,
        api_key=settings.vllm_api_key,
        timeout_seconds=5.0,
    ).check()
    vision_ok = await VllmHealthChecker(
        base_url=settings.vllm_vision_base_url,
        api_key=settings.vllm_api_key,
        timeout_seconds=5.0,
    ).check()

    return SystemHealthResponse(
        api=ComponentHealthResponse(status="ok"),
        storage=storage_status,
        job_repository=repository_status,
        vllm_text=ComponentHealthResponse(status="ok" if text_ok else "unavailable"),
        vllm_vision=ComponentHealthResponse(status="ok" if vision_ok else "unavailable"),
    )


@router.get("/system/metrics", response_model=JobMetricsResponse)
async def get_system_metrics(
    repository: JobRepository = Depends(get_job_repository),
) -> JobMetricsResponse:
    """Return aggregated job metrics."""
    metrics = await MetricsService(repository).collect_job_metrics()
    return JobMetricsResponse(
        total=metrics.total,
        by_status={status.value: count for status, count in metrics.by_status.items()},
        queued=metrics.queued,
        planning=metrics.planning,
        analyzing=metrics.analyzing,
        waiting_for_approval=metrics.waiting_for_approval,
        generating=metrics.generating,
        reviewing=metrics.reviewing,
        completed=metrics.completed,
        failed=metrics.failed,
        cancelled=metrics.cancelled,
    )


@router.post("/system/ltx/validate", response_model=LtxValidationResponse)
async def validate_ltx_parameters(request: LtxValidationRequest) -> LtxValidationResponse:
    """Dry-run validate LTX-Video parameters before a costly GPU generation."""
    result = LtxVideoParameterValidator().validate(
        width=request.width,
        height=request.height,
        num_frames=request.num_frames,
        fps=request.fps,
        inference_steps=request.inference_steps,
        guidance_scale=request.guidance_scale,
    )
    return LtxValidationResponse(
        valid=result.valid,
        errors=result.errors,
        warnings=result.warnings,
        recommended=result.recommended,
    )


@router.post(
    "/generations",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_generation(
    background_tasks: BackgroundTasks,
    prompt: str = Form(..., min_length=3),
    image: UploadFile = File(...),
    requested_backend: VideoBackend | None = Form(default=None),
    width: int | None = Form(default=None),
    height: int | None = Form(default=None),
    num_frames: int | None = Form(default=None),
    fps: int | None = Form(default=None),
    seed: int | None = Form(default=None),
    guidance_scale: float | None = Form(default=None),
    inference_steps: int | None = Form(default=None),
    user: UserContext = Depends(get_current_user),
    service: JobApplicationService = Depends(get_job_service),
    orchestrator: VideoGenerationOrchestrator = Depends(get_orchestrator),
) -> JobResponse:
    """Create and asynchronously process a text+image to video generation job."""
    settings = get_settings()
    content = await image.read()
    validator = ImageValidationService(
        max_bytes=settings.max_upload_bytes, max_pixels=settings.max_image_pixels
    )
    try:
        sanitized = validator.validate_and_sanitize(content, image.content_type or "")
    except ImageValidationError as exc:
        status_code = (
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            if "large" in str(exc).lower()
            else status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        )
        raise HTTPException(status_code, str(exc)) from exc

    job = await service.create_generation_job(
        prompt=prompt,
        image_content=sanitized.content,
        image_filename=image.filename or "input.png",
        image_mime_type=sanitized.mime_type,
        image_width=sanitized.width,
        image_height=sanitized.height,
        requested_backend=requested_backend,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        seed=seed,
        guidance_scale=guidance_scale,
        inference_steps=inference_steps,
        user_id=user.user_id,
    )
    queue = create_job_task_queue(
        background_tasks=background_tasks,
        orchestrator=orchestrator,
        repository=get_job_repository(),
    )
    queue.enqueue_generation(job.id)
    return JobResponse.from_job(job)


@router.get("/generations", response_model=list[JobResponse])
async def list_generations(
    repository: JobRepository = Depends(get_job_repository),
    user: UserContext = Depends(get_current_user),
) -> list[JobResponse]:
    """Return all known generation jobs visible to the current user."""
    jobs = await repository.list_all()
    if user.auth_enabled:
        jobs = [job for job in jobs if job.request.user_id == user.user_id]
    return [JobResponse.from_job(job) for job in jobs]


@router.get("/generations/{job_id}", response_model=JobResponse)
async def get_generation(
    job_id: UUID,
    repository: JobRepository = Depends(get_job_repository),
    user: UserContext = Depends(get_current_user),
) -> JobResponse:
    """Return a generation job."""
    job = await repository.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Generation job not found")
    ensure_job_access(job, user)
    return JobResponse.from_job(job)


@router.post(
    "/generations/{job_id}/rerun",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def rerun_generation(
    job_id: UUID,
    background_tasks: BackgroundTasks,
    repository: JobRepository = Depends(get_job_repository),
    orchestrator: VideoGenerationOrchestrator = Depends(get_orchestrator),
    user: UserContext = Depends(get_current_user),
) -> JobResponse:
    """Create a new generation job using the same request and preferences."""
    source = await repository.get(job_id)
    if source is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Generation job not found")
    ensure_job_access(source, user)
    new_job = VideoGenerationJob(request=source.request)
    await repository.save(new_job)
    queue = create_job_task_queue(
        background_tasks=background_tasks,
        orchestrator=orchestrator,
        repository=get_job_repository(),
    )
    queue.enqueue_generation(new_job.id)
    return JobResponse.from_job(new_job)


@router.post(
    "/generations/{job_id}/variant",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_generation_variant(
    job_id: UUID,
    background_tasks: BackgroundTasks,
    repository: JobRepository = Depends(get_job_repository),
    orchestrator: VideoGenerationOrchestrator = Depends(get_orchestrator),
    user: UserContext = Depends(get_current_user),
) -> JobResponse:
    """Create a new generation job with the same parameters but a different seed."""
    source = await repository.get(job_id)
    if source is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Generation job not found")
    ensure_job_access(source, user)
    current_seed = source.request.preferences.seed
    if current_seed is None and source.generation_parameters is not None:
        current_seed = source.generation_parameters.seed
    next_seed = 1 if current_seed is None else current_seed + 1
    preferences = source.request.preferences.model_copy(update={"seed": next_seed})
    request = source.request.model_copy(update={"preferences": preferences})

    new_job = VideoGenerationJob(request=request)
    await repository.save(new_job)
    queue = create_job_task_queue(
        background_tasks=background_tasks,
        orchestrator=orchestrator,
        repository=get_job_repository(),
    )
    queue.enqueue_generation(new_job.id)
    return JobResponse.from_job(new_job)


@router.post("/generations/{job_id}/approve", response_model=JobResponse)
async def approve_generation(
    job_id: UUID,
    background_tasks: BackgroundTasks,
    repository: JobRepository = Depends(get_job_repository),
    orchestrator: VideoGenerationOrchestrator = Depends(get_orchestrator),
    user: UserContext = Depends(get_current_user),
) -> JobResponse:
    """Approve a generation waiting for human confirmation."""
    job = await repository.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Generation job not found")
    ensure_job_access(job, user)
    if job.status != JobStatus.WAITING_FOR_APPROVAL:
        raise HTTPException(status.HTTP_409_CONFLICT, "Generation is not waiting for approval")
    approved = job.transition(JobStatus.QUEUED, "Human approved GPU generation")
    await repository.save(approved)
    queue = create_job_task_queue(
        background_tasks=background_tasks,
        orchestrator=orchestrator,
        repository=get_job_repository(),
    )
    queue.enqueue_approved_generation(job.id)
    return JobResponse.from_job(approved)


@router.post("/generations/{job_id}/reject", response_model=JobResponse)
async def reject_generation(
    job_id: UUID,
    repository: JobRepository = Depends(get_job_repository),
    orchestrator: VideoGenerationOrchestrator = Depends(get_orchestrator),
    user: UserContext = Depends(get_current_user),
) -> JobResponse:
    """Reject and cancel a generation waiting for human confirmation."""
    job = await repository.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Generation job not found")
    ensure_job_access(job, user)
    if job.status != JobStatus.WAITING_FOR_APPROVAL:
        raise HTTPException(status.HTTP_409_CONFLICT, "Generation is not waiting for approval")
    rejected = await orchestrator.reject(job)
    return JobResponse.from_job(rejected)


@router.post("/generations/{job_id}/cancel", response_model=JobResponse)
async def cancel_generation(
    job_id: UUID,
    repository: JobRepository = Depends(get_job_repository),
    user: UserContext = Depends(get_current_user),
) -> JobResponse:
    """Cancel a queued or approval-waiting generation job."""
    job = await repository.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Generation job not found")
    ensure_job_access(job, user)
    if job.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
        raise HTTPException(status.HTTP_409_CONFLICT, "Generation cannot be cancelled")
    cancelled = job.model_copy(update={"pending_parameters": None}).transition(
        JobStatus.CANCELLED, "Generation cancelled by user"
    )
    await repository.save(cancelled)
    return JobResponse.from_job(cancelled)


@router.get("/generations/{job_id}/events")
async def stream_generation_events(
    job_id: UUID,
    repository: JobRepository = Depends(get_job_repository),
    user: UserContext = Depends(get_current_user),
) -> StreamingResponse:
    """Stream job status events as Server-Sent Events."""

    async def event_stream() -> AsyncIterator[str]:
        previous_status: JobStatus | None = None
        for _ in range(120):
            job = await repository.get(job_id)
            if job is None:
                yield "event: error\ndata: Generation job not found\n\n"
                return
            try:
                ensure_job_access(job, user)
            except HTTPException:
                yield "event: error\ndata: Generation job not found\n\n"
                return
            if job.status != previous_status:
                yield f"event: status\ndata: {job.status.value}\n\n"
                previous_status = job.status
            if job.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
                return
            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/generations/{job_id}/video", response_model=None)
async def get_generation_video(
    job_id: UUID,
    repository: JobRepository = Depends(get_job_repository),
    user: UserContext = Depends(get_current_user),
) -> Response:
    """Download a generated video artifact."""
    job = await repository.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Generation job not found")
    ensure_job_access(job, user)
    if job.video is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Generated video not available")
    if job.video.storage.path is not None:
        return FileResponse(
            path=job.video.storage.path,
            media_type=job.video.storage.mime_type,
            filename=f"{job.id}.mp4",
        )
    try:
        download_url = resolve_download_url(job.video.storage, get_settings())
    except StorageUrlResolutionError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return RedirectResponse(download_url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)

