"""S3-compatible object storage adapter."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from video_ai.domain.models import StorageRef


class S3StorageUnavailableError(RuntimeError):
    """Raised when S3 dependencies or configuration are unavailable."""


class S3StorageService:
    """Store assets in S3-compatible object storage such as AWS S3, R2 or MinIO."""

    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str | None = None,
        region_name: str = "us-east-1",
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        public_base_url: str | None = None,
        presigned_url_expires_seconds: int = 3600,
        client: object | None = None,
    ) -> None:
        self._bucket = bucket
        self._public_base_url = public_base_url.rstrip("/") if public_base_url else None
        self._presigned_url_expires_seconds = presigned_url_expires_seconds
        self._client = client or self._create_client(
            endpoint_url=endpoint_url,
            region_name=region_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
        )

    async def save_bytes(self, content: bytes, filename: str, mime_type: str) -> StorageRef:
        """Persist bytes in S3 and return a storage reference."""
        suffix = Path(filename).suffix.lower()
        key = f"{uuid4()}{suffix}"
        self._client.put_object(  # type: ignore[attr-defined]
            Bucket=self._bucket,
            Key=key,
            Body=content,
            ContentType=mime_type,
        )
        return StorageRef(
            uri=self._uri_for_key(key),
            path=None,
            mime_type=mime_type,
            size_bytes=len(content),
        )

    def presigned_url(self, uri: str) -> str:
        """Return a presigned URL for a stored object URI."""
        key = self._key_from_uri(uri)
        return str(
            self._client.generate_presigned_url(  # type: ignore[attr-defined]
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=self._presigned_url_expires_seconds,
            )
        )

    def _uri_for_key(self, key: str) -> str:
        if self._public_base_url:
            return f"{self._public_base_url}/{key}"
        return f"s3://{self._bucket}/{key}"

    def _key_from_uri(self, uri: str) -> str:
        if uri.startswith(f"s3://{self._bucket}/"):
            return uri.removeprefix(f"s3://{self._bucket}/")
        if self._public_base_url and uri.startswith(f"{self._public_base_url}/"):
            return uri.removeprefix(f"{self._public_base_url}/")
        return uri.rsplit("/", maxsplit=1)[-1]

    @staticmethod
    def _create_client(
        *,
        endpoint_url: str | None,
        region_name: str,
        access_key_id: str | None,
        secret_access_key: str | None,
    ) -> object:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise S3StorageUnavailableError(
                "Install the 'storage' extra to use STORAGE_BACKEND=s3"
            ) from exc
        return boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )
