from io import BytesIO

import pytest
from PIL import Image

from image_helpers import png_bytes
from video_ai.infrastructure.image_validation import ImageValidationError, ImageValidationService


def test_image_validator_accepts_and_sanitizes_png() -> None:
    service = ImageValidationService(max_bytes=1024 * 1024)

    sanitized = service.validate_and_sanitize(png_bytes(4, 3), "image/png")

    assert sanitized.mime_type == "image/png"
    assert sanitized.width == 4
    assert sanitized.height == 3
    assert sanitized.content


def test_image_validator_rejects_invalid_bytes() -> None:
    service = ImageValidationService(max_bytes=1024 * 1024)

    with pytest.raises(ImageValidationError):
        service.validate_and_sanitize(b"not an image", "image/png")


def test_image_validator_rejects_mime_mismatch() -> None:
    service = ImageValidationService(max_bytes=1024 * 1024)

    with pytest.raises(ImageValidationError):
        service.validate_and_sanitize(png_bytes(), "image/jpeg")


def test_image_validator_rejects_large_resolution() -> None:
    output = BytesIO()
    Image.new("RGB", (20, 20), color=(0, 0, 0)).save(output, format="PNG")
    service = ImageValidationService(max_bytes=1024 * 1024, max_pixels=10)

    with pytest.raises(ImageValidationError):
        service.validate_and_sanitize(output.getvalue(), "image/png")
