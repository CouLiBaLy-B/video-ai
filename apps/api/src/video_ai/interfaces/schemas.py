"""HTTP interface schemas."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from video_ai.domain.enums import JobStatus
from video_ai.domain.models import AgentPlanStep, JobEvent, VideoGenerationJob


class PlanStepResponse(BaseModel):
    """Public representation of an agentic plan step."""

    name: str
    description: str
    agent: str

    @classmethod
    def from_step(cls, step: AgentPlanStep) -> PlanStepResponse:
        return cls(name=step.name, description=step.description, agent=step.agent)


class JobEventResponse(BaseModel):
    """Public representation of a job lifecycle event."""

    status: JobStatus
    message: str
    created_at: datetime

    @classmethod
    def from_event(cls, event: JobEvent) -> JobEventResponse:
        return cls(status=event.status, message=event.message, created_at=event.created_at)


class GenerationParametersResponse(BaseModel):
    """Public representation of selected generation parameters."""

    backend: str
    model_id: str
    width: int
    height: int
    num_frames: int
    fps: int
    seed: int | None
    guidance_scale: float
    inference_steps: int


class JobResponse(BaseModel):
    """Public job representation."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: JobStatus
    status_reason: str | None
    prompt: str
    created_at: datetime
    updated_at: datetime
    video_url: str | None = None
    plan_steps: list[PlanStepResponse] = []
    events: list[JobEventResponse] = []
    parameters: GenerationParametersResponse | None = None

    @classmethod
    def from_job(cls, job: VideoGenerationJob) -> JobResponse:
        """Map a domain job to an API response."""
        video_url = f"/api/generations/{job.id}/video" if job.video else None
        plan_steps = (
            [PlanStepResponse.from_step(step) for step in job.plan.steps]
            if job.plan is not None
            else []
        )
        events = [JobEventResponse.from_event(event) for event in job.events]
        parameters = None
        if job.generation_parameters is not None:
            selected = job.generation_parameters
            parameters = GenerationParametersResponse(
                backend=selected.model.backend.value,
                model_id=selected.model.id,
                width=selected.width,
                height=selected.height,
                num_frames=selected.num_frames,
                fps=selected.fps,
                seed=selected.seed,
                guidance_scale=selected.guidance_scale,
                inference_steps=selected.inference_steps,
            )
        return cls(
            id=job.id,
            status=job.status,
            status_reason=job.status_reason,
            prompt=job.request.prompt,
            created_at=job.created_at,
            updated_at=job.updated_at,
            video_url=video_url,
            plan_steps=plan_steps,
            events=events,
            parameters=parameters,
        )


class SystemCapabilitiesResponse(BaseModel):
    """Runtime capabilities exposed to the frontend."""

    agent_planner_provider: str
    ai_provider: str
    default_video_backend: str
    task_queue_backend: str
    available_video_backends: list[str]
    planned_video_backends: list[str]
    text_model: str
    vision_model: str
    vllm_fallback_to_mock: bool


class ComponentHealthResponse(BaseModel):
    """Health status for one runtime component."""

    status: str
    detail: str | None = None


class SystemHealthResponse(BaseModel):
    """Aggregated runtime health response."""

    api: ComponentHealthResponse
    storage: ComponentHealthResponse
    job_repository: ComponentHealthResponse
    vllm_text: ComponentHealthResponse
    vllm_vision: ComponentHealthResponse


class LtxValidationRequest(BaseModel):
    """LTX-Video dry-run validation request."""

    width: int = 768
    height: int = 512
    num_frames: int = 121
    fps: int = 24
    inference_steps: int = 30
    guidance_scale: float = 3.5


class LtxValidationResponse(BaseModel):
    """LTX-Video dry-run validation response."""

    valid: bool
    errors: list[str]
    warnings: list[str]
    recommended: dict[str, int | float | str]


class JobMetricsResponse(BaseModel):
    """Public job metrics response."""

    total: int
    by_status: dict[str, int]
    queued: int
    planning: int
    analyzing: int
    waiting_for_approval: int
    generating: int
    reviewing: int
    completed: int
    failed: int
    cancelled: int


class ErrorResponse(BaseModel):
    """Error payload."""

    detail: str
