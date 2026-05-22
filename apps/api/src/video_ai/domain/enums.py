"""Domain enumerations for generation workflows."""

from enum import StrEnum


class JobStatus(StrEnum):
    """Lifecycle status for a video generation job."""

    QUEUED = "queued"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    GENERATING = "generating"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VideoBackend(StrEnum):
    """Supported video generation backend identifiers."""

    MOCK = "mock"
    LTX_VIDEO = "ltx-video"
    WAN_I2V = "wan-i2v"


class AssetKind(StrEnum):
    """Kinds of assets stored by the platform."""

    INPUT_IMAGE = "input_image"
    OUTPUT_VIDEO = "output_video"
    AGENT_ARTIFACT = "agent_artifact"
