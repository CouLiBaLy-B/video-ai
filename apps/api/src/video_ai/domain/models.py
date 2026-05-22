"""Framework-independent domain models."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from video_ai.domain.enums import AssetKind, JobStatus, VideoBackend


class DomainModel(BaseModel):
    """Base class for immutable domain models."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ImageAsset(DomainModel):
    """Input image metadata."""

    id: UUID = Field(default_factory=uuid4)
    path: Path
    mime_type: str
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    size_bytes: int = Field(ge=1)
    kind: AssetKind = AssetKind.INPUT_IMAGE

    @field_validator("mime_type")
    @classmethod
    def validate_image_mime(cls, value: str) -> str:
        """Ensure the uploaded asset is an image."""
        allowed = {"image/jpeg", "image/png", "image/webp"}
        if value not in allowed:
            raise ValueError(f"Unsupported image MIME type: {value}")
        return value


class GenerationRequest(DomainModel):
    """User request for text+image to video generation."""

    id: UUID = Field(default_factory=uuid4)
    prompt: str = Field(min_length=3, max_length=8_000)
    image: ImageAsset
    user_id: str | None = Field(default=None, max_length=128)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("prompt")
    @classmethod
    def normalize_prompt(cls, value: str) -> str:
        """Normalize whitespace while preserving prompt semantics."""
        normalized = " ".join(value.strip().split())
        if len(normalized) < 3:
            raise ValueError("Prompt is too short")
        return normalized


class AgentPlanStep(DomainModel):
    """Single planned agentic workflow step."""

    name: str
    description: str
    agent: str


class AgentPlan(DomainModel):
    """Plan produced by a simple planner or DeepAgents supervisor."""

    summary: str
    steps: list[AgentPlanStep] = Field(default_factory=list)
    requires_human_approval: bool = False


class ImageAnalysis(DomainModel):
    """Structured visual understanding extracted from the image."""

    subject: str
    scene: str
    style: str | None = None
    lighting: str | None = None
    composition: str | None = None
    motion_suggestions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class EnhancedPrompt(DomainModel):
    """Prompt package optimized for a video model."""

    positive: str = Field(min_length=3)
    negative: str = "low quality, blurry, distorted, flickering, artifacts"
    camera_motion: str | None = None
    style: str | None = None


class ModelProfile(DomainModel):
    """Selectable video model profile."""

    id: str
    backend: VideoBackend
    display_name: str
    supports_image_to_video: bool = True
    supports_text_to_video: bool = False
    min_vram_gb: int = Field(ge=0)
    default_width: int = Field(ge=64)
    default_height: int = Field(ge=64)
    default_fps: int = Field(default=24, ge=1, le=120)


class GenerationParameters(DomainModel):
    """Parameters sent to a concrete video generator."""

    prompt: EnhancedPrompt
    image: ImageAsset
    model: ModelProfile
    width: int = Field(ge=64, le=4096)
    height: int = Field(ge=64, le=4096)
    num_frames: int = Field(default=121, ge=1, le=1_000)
    fps: int = Field(default=24, ge=1, le=120)
    seed: int | None = Field(default=None, ge=0)
    guidance_scale: float = Field(default=3.5, ge=0.0, le=30.0)
    inference_steps: int = Field(default=30, ge=1, le=200)
    extra: dict[str, Any] = Field(default_factory=dict)


class StorageRef(DomainModel):
    """Reference to an object stored by a storage adapter."""

    uri: str
    path: Path | None = None
    mime_type: str
    size_bytes: int = Field(ge=0)


class GeneratedVideo(DomainModel):
    """Generated video metadata."""

    id: UUID = Field(default_factory=uuid4)
    storage: StorageRef
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    fps: int = Field(ge=1)
    num_frames: int = Field(ge=1)
    duration_seconds: float = Field(gt=0)
    seed: int | None = None
    model_id: str


class QualityReport(DomainModel):
    """Agentic review of a generated video."""

    score: float = Field(ge=0.0, le=1.0)
    summary: str
    issues: list[str] = Field(default_factory=list)
    suggested_improvements: list[str] = Field(default_factory=list)
    accepted: bool


class VideoGenerationJob(DomainModel):
    """Aggregate root for a generation workflow."""

    id: UUID = Field(default_factory=uuid4)
    request: GenerationRequest
    status: JobStatus = JobStatus.QUEUED
    status_reason: str | None = None
    plan: AgentPlan | None = None
    video: GeneratedVideo | None = None
    quality_report: QualityReport | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def transition(self, status: JobStatus, reason: str | None = None) -> VideoGenerationJob:
        """Return a new job with an updated status."""
        return self.model_copy(
            update={
                "status": status,
                "status_reason": reason,
                "updated_at": datetime.now(UTC),
            }
        )
