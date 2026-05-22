"""LTX-Video generator adapter.

This adapter is intentionally lazy: heavyweight GPU dependencies are imported
only when generation is invoked. That keeps the API and tests usable in CPU-only
or CI environments while preserving a production-ready extension point.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from video_ai.domain.enums import VideoBackend
from video_ai.domain.models import GeneratedVideo, GenerationParameters, ModelProfile
from video_ai.domain.ports import StorageService


class LtxVideoUnavailableError(RuntimeError):
    """Raised when LTX-Video dependencies are not installed."""


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


class LtxVideoGenerator:
    """Generate image-to-video clips with Lightricks LTX-Video via Diffusers."""

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

    async def generate(self, parameters: GenerationParameters) -> GeneratedVideo:
        """Run LTX-Video and store the resulting MP4."""
        pipe = self._load_pipeline()
        image = self._load_image(parameters.image.path)
        output = pipe(
            image=image,
            prompt=parameters.prompt.positive,
            negative_prompt=parameters.prompt.negative,
            width=parameters.width,
            height=parameters.height,
            num_frames=parameters.num_frames,
            num_inference_steps=parameters.inference_steps,
            guidance_scale=parameters.guidance_scale,
        )
        frames = output.frames[0] if output.frames and isinstance(output.frames[0], list) else output.frames
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

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from diffusers import DiffusionPipeline
        except ImportError as exc:  # pragma: no cover - optional GPU dependency
            raise LtxVideoUnavailableError(
                "Install the 'video' extra and GPU dependencies to use LTX-Video."
            ) from exc
        dtype = getattr(torch, self._torch_dtype)
        self._pipeline = DiffusionPipeline.from_pretrained(
            self._model_id,
            torch_dtype=dtype,
        ).to(self._device)
        return self._pipeline

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
