from video_ai.storage.s3 import S3StorageService


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:  # noqa: N803
        _ = ContentType
        self.objects[(Bucket, Key)] = Body

    def generate_presigned_url(self, operation: str, Params: dict[str, str], ExpiresIn: int) -> str:  # noqa: N803
        return f"https://signed.test/{operation}/{Params['Bucket']}/{Params['Key']}?exp={ExpiresIn}"


async def test_s3_storage_saves_bytes_and_returns_public_uri() -> None:
    client = FakeS3Client()
    storage = S3StorageService(
        bucket="bucket",
        public_base_url="https://cdn.test/assets",
        client=client,
    )

    ref = await storage.save_bytes(b"hello", "video.mp4", "video/mp4")

    assert ref.uri.startswith("https://cdn.test/assets/")
    assert ref.path is None
    assert ref.size_bytes == 5
    assert len(client.objects) == 1


def test_s3_storage_generates_presigned_url() -> None:
    storage = S3StorageService(bucket="bucket", client=FakeS3Client())

    url = storage.presigned_url("s3://bucket/file.mp4")

    assert url.startswith("https://signed.test/get_object/bucket/file.mp4")
