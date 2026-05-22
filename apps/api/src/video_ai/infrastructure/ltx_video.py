"""LTX-Video generator adapter.

This adapter is intentionally lazy: heavyweight GPU dependencies are imported
only when generation is invoked. That keeps the API and tests usable in CPU-only
or CI environments while preserving a production-ready extension point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from video_ai.domain.enums import VideoBackend
from video_ai.domain.models import GeneratedVideo, GenerationParameters, ModelProfile
from video_ai.domain.ports import StorageService


class LtxVideoUnavailableError(RuntimeError):
    """Raised when LTX-Video dependencies or runtime are unavailable."""


@dataclass(frozen=True)
class LtxValidationResult:
    """Validation result for LTX-Video parameters."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommended: dict[str, int | float | str] = field(default_factory=dict)


LTX_VIDEO_2B_DISTILLED_PROFILE = ModelProfile(
    id="Lightricks/LTX-Video",
    backend=VideoBackend.LTX_VIDEO,
    display_name="LTX-Video",
    supports_image_to_video=True,
    supports_text_to_video=True,
    min_vram_gb=12,
    default_width=768,
    default_height=512,
    default_fps=24,
)


class LtxVideoParameterValidator:
    """Validate user parameters before expensive LTX-Video GPU execution."""

    def validate(
        self,
        *,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        inference_steps: int,
        guidance_scale: float,
    ) -> LtxValidationResult:
        """Return validation warnings/errors and recommended safe values."""
        errors: list[str] = []
        warnings: list[str] = []
        recommended: dict[str, int | float | str] = {}

        if width < 256 or height < 256:
            warnings.append("Very small resolutions can reduce perceived video quality.")
        if width > 1280 or height > 768:
            warnings.append("High resolutions may require substantial VRAM for LTX-Video.")
        if width % 32 != 0:
            errors.append("Width should be divisible by 32 for stable video diffusion.")
            recommended["width"] = _nearest_multiple(width, 32)
        if height % 32 != 0:
            errors.append("Height should be divisible by 32 for stable video diffusion.")
            recommended["height"] = _nearest_multiple(height, 32)
        if (num_frames - 1) % 8 != 0:
            warnings.append("LTX-Video commonly works best with frame counts of the form 8n+1.")
            recommended["num_frames"] = _nearest_frame_count(num_frames)
        if fps > 30:
            warnings.append("FPS above 30 increases output size and may not improve quality.")
        if inference_steps < 4:
            warnings.append("Very few inference steps can reduce quality.")
        if guidance_scale > 10:
            warnings.append("High guidance scale can over-constrain motion and cause artifacts.")

        return LtxValidationResult(
            valid=not errors,
            errors=errors,
            warnings=warnings,
            recommended=recommended,
        )


class LtxVideoGenerator:
    """Generate image-to-video clips with Lightricks LTX-Video via Diffusers."""

    _EXTRA_KWARG_ALLOWLIST = {
        "decode_timestep",
        "decode_noise_scale",
        "max_sequence_length",
        "stg_scale",
        "stg_rescale",
    }

    def __init__(
        self,
        *,
        storage: StorageService,
        model_id: str = "Lightricks/LTX-Video",
        torch_dtype: str = "bfloat16",
        device: str = "cuda",
    ) -> None:
        self._storage = storage
        self._model_id = model_id
        self._torch_dtype = torch_dtype
        self._device = device
        self._pipeline: Any | None = None
        self._torch: Any | None = None

    async def generate(self, parameters: GenerationParameters) -> GeneratedVideo:
        """Run LTX-Video and store the resulting MP4."""
        validator = LtxVideoParameterValidator()
        validation = validator.validate(
            width=parameters.width,
            height=parameters.height,
            num_frames=parameters.num_frames,
            fps=parameters.fps,
            inference_steps=parameters.inference_steps,
            guidance_scale=parameters.guidance_scale,
        )
        if not validation.valid:
            raise ValueError("Invalid LTX-Video parameters: " + "; ".join(validation.errors))

        pipe = self._load_pipeline()
        image = self._load_image(parameters.image.path)
        output = pipe(**self._build_inference_kwargs(parameters, image=image))
        frames = (
            output.frames[0]
            if output.frames and isinstance(output.frames[0], list)
            else output.frames
        )
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "ltx-video.mp4"
            self._export_to_video(frames, output_path, parameters.fps)
            content = output_path.read_bytes()
        storage = await self._storage.save_bytes(content, "ltx-video.mp4", "video/mp4")
        return GeneratedVideo(
            storage=storage,
            width=parameters.width,
            height=parameters.height,
            fps=parameters.fps,
            num_frames=parameters.num_frames,
            duration_seconds=parameters.num_frames / parameters.fps,
            seed=parameters.seed,
            model_id=self._model_id,
        )

    def _build_inference_kwargs(
        self, parameters: GenerationParameters, *, image: Any
    ) -> dict[str, Any]:
        """Map domain parameters to Diffusers pipeline keyword arguments."""
        kwargs: dict[str, Any] = {
            "image": image,
            "prompt": parameters.prompt.positive,
            "negative_prompt": parameters.prompt.negative,
            "width": parameters.width,
            "height": parameters.height,
            "num_frames": parameters.num_frames,
            "num_inference_steps": parameters.inference_steps,
            "guidance_scale": parameters.guidance_scale,
        }
        if parameters.seed is not None:
            kwargs["generator"] = self._make_torch_generator(parameters.seed)
        kwargs.update(
            {
                key: value
                for key, value in parameters.extra.items()
                if key in self._EXTRA_KWARG_ALLOWLIST
            }
        )
        return kwargs

    def _make_torch_generator(self, seed: int) -> Any:
        """Create a torch generator for deterministic video generations."""
        torch = self._load_torch()
        try:
            return torch.Generator(device=self._device).manual_seed(seed)
        except RuntimeError:
            return torch.Generator().manual_seed(seed)

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        torch = self._load_torch()
        try:
            from diffusers import DiffusionPipeline
        except ImportError as exc:  # pragma: no cover - optional GPU dependency
            raise LtxVideoUnavailableError(
                "Install the 'video' extra and GPU dependencies to use LTX-Video."
            ) from exc
        self._validate_runtime_device(torch)
        dtype = getattr(torch, self._torch_dtype, None)
        if dtype is None:
            raise LtxVideoUnavailableError(f"Unsupported torch dtype: {self._torch_dtype}")
        try:
            self._pipeline = DiffusionPipeline.from_pretrained(
                self._model_id,
                torch_dtype=dtype,
            ).to(self._device)
        except Exception as exc:  # pragma: no cover - depends on GPU/HF runtime
            raise LtxVideoUnavailableError(f"Failed to load LTX-Video pipeline: {exc}") from exc
        return self._pipeline

    def _load_torch(self) -> Any:
        if self._torch is not None:
            return self._torch
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional GPU dependency
            raise LtxVideoUnavailableError("PyTorch is required to use LTX-Video.") from exc
        self._torch = torch
        return torch

    def _validate_runtime_device(self, torch: Any) -> None:
        if self._device.startswith("cuda") and not torch.cuda.is_available():
            raise LtxVideoUnavailableError(
                "CUDA device requested for LTX-Video, but torch.cuda.is_available() is false."
            )

    @staticmethod
    def _load_image(path: Path) -> Any:
        try:
            from diffusers.utils import load_image
        except ImportError as exc:  # pragma: no cover - optional GPU dependency
            raise LtxVideoUnavailableError("diffusers is required to load images") from exc
        return load_image(str(path))

    @staticmethod
    def _export_to_video(frames: Any, output_path: Path, fps: int) -> None:
        try:
            from diffusers.utils import export_to_video
        except ImportError as exc:  # pragma: no cover - optional GPU dependency
            raise LtxVideoUnavailableError("diffusers is required to export videos") from exc
        export_to_video(frames, str(output_path), fps=fps)


def _nearest_multiple(value: int, multiple: int) -> int:
    return max(multiple, round(value / multiple) * multiple)


def _nearest_frame_count(value: int) -> int:
    if value <= 1:
        return 1
    return max(9, round((value - 1) / 8) * 8 + 1)
