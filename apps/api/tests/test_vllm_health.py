import httpx

from video_ai.infrastructure.vllm import VllmHealthChecker


async def test_vllm_health_checker_returns_true_for_models_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    checker = VllmHealthChecker(
        base_url="http://vllm.test/v1",
        api_key="test",
        client=client,
    )

    assert await checker.check() is True
    await client.aclose()


async def test_vllm_health_checker_returns_false_for_unavailable_endpoint() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    checker = VllmHealthChecker(
        base_url="http://vllm.test/v1",
        api_key="test",
        client=client,
    )

    assert await checker.check() is False
    await client.aclose()
