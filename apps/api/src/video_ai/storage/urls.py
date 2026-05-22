"""Helpers for resolving stored video assets to downloadable URLs."""

from __future__ import annotations

from video_ai.config.settings import Settings
from video_ai.domain.models import StorageRef
from video_ai.storage.s3 import S3StorageService


class StorageUrlResolutionError(RuntimeError):
    """Raised when an asset URL cannot be resolved."""


def resolve_download_url(storage: StorageRef, settings: Settings) -> str:
    """Resolve a non-local storage reference to a downloadable URL."""
    if storage.uri.startswith("http://") or storage.uri.startswith("https://"):
        return storage.uri
    if storage.uri.startswith("s3://"):
        service = S3StorageService(
            bucket=settings.s3_bucket,
            endpoint_url=settings.s3_endpoint_url,
            region_name=settings.s3_region,
            access_key_id=settings.s3_access_key_id,
            secret_access_key=settings.s3_secret_access_key,
            public_base_url=settings.s3_public_base_url,
            presigned_url_expires_seconds=settings.s3_presigned_url_expires_seconds,
        )
        return service.presigned_url(storage.uri)
    raise StorageUrlResolutionError(f"Unsupported non-local storage URI: {storage.uri}")
