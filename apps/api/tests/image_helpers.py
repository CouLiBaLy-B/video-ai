"""Test helpers for creating small valid images."""

from io import BytesIO

from PIL import Image


def png_bytes(width: int = 2, height: int = 2) -> bytes:
    """Return bytes for a minimal valid PNG image."""
    output = BytesIO()
    Image.new("RGB", (width, height), color=(32, 64, 128)).save(output, format="PNG")
    return output.getvalue()
