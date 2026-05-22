"""Local filesystem storage adapter."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from video_ai.domain.models import StorageRef


class LocalStorageService:
    """Store assets under a local root directory."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    async def save_bytes(self, content: bytes, filename: str, mime_type: str) -> StorageRef:
        """Persist bytes and return a storage reference."""
        suffix = Path(filename).suffix.lower()
        safe_name = f"{uuid4()}{suffix}"
        target = self._root / safe_name
        target.write_bytes(content)
        return StorageRef(
            uri=f"local://{safe_name}",
            path=target,
            mime_type=mime_type,
            size_bytes=len(content),
        )
