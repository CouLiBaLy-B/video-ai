"""Image validation and sanitization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Final

from PIL import Image, UnidentifiedImageError


class ImageValidationError(ValueError):
    """Raised when an uploaded image is invalid or unsafe."""


@dataclass(frozen=True)
class SanitizedImage:
    """Sanitized image bytes and metadata."""

    content: bytes
    mime_type: str
    width: int
    height: int


class ImageValidationService:
    """Validate images and strip metadata before storage/model use."""

    _FORMAT_TO_MIME: Final[dict[str, str]] = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }

    def __init__(self, *, max_bytes: int, max_pixels: int = 16_000_000) -> None:
        self._max_bytes = max_bytes
        self._max_pixels = max_pixels

    def validate_and_sanitize(self, content: bytes, declared_mime_type: str) -> SanitizedImage:
        """Validate type/size, decode image and re-encode without metadata."""
        if len(content) > self._max_bytes:
            raise ImageValidationError("Image is too large")
        if declared_mime_type not in set(self._FORMAT_TO_MIME.values()):
            raise ImageValidationError("Unsupported image type")

        try:
            with Image.open(BytesIO(content)) as image:
                image.load()
                detected_mime = self._FORMAT_TO_MIME.get(image.format or "")
                if detected_mime is None:
                    raise ImageValidationError("Unsupported image format")
                if detected_mime != declared_mime_type:
                    raise ImageValidationError("Declared MIME type does not match image content")
                width, height = image.size
                if width * height > self._max_pixels:
                    raise ImageValidationError("Image resolution is too large")
                sanitized = self._encode_without_metadata(image, detected_mime)
        except UnidentifiedImageError as exc:
            raise ImageValidationError("Invalid image file") from exc

        return SanitizedImage(
            content=sanitized,
            mime_type=declared_mime_type,
            width=width,
            height=height,
        )

    @staticmethod
    def _encode_without_metadata(image: Image.Image, mime_type: str) -> bytes:
        output = BytesIO()
        if mime_type == "image/jpeg":
            image.convert("RGB").save(output, format="JPEG", quality=95, optimize=True)
        elif mime_type == "image/webp":
            image.save(output, format="WEBP", quality=95, method=6)
        else:
            image.save(output, format="PNG", optimize=True)
        return output.getvalue()
