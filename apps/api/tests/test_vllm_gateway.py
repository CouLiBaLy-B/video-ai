from pathlib import Path

import httpx

from video_ai.domain.models import ImageAnalysis, ImageAsset
from video_ai.infrastructure.vllm import VllmChatGateway, VllmPromptEnhancer, VllmVisionAnalyzer


def make_transport(content: str) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    return httpx.MockTransport(handler)


async def test_vllm_complete_json_parses_object() -> None:
    client = httpx.AsyncClient(transport=make_transport('{"answer": "ok"}'))
    gateway = VllmChatGateway(
        base_url="http://vllm.test/v1",
        api_key="test",
        model="model",
        client=client,
    )

    response = await gateway.complete_json([{"role": "user", "content": "hi"}])

    assert response == {"answer": "ok"}
    await client.aclose()


async def test_vision_analyzer_maps_vllm_json(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image")
    content = (
        '{"subject":"cat","scene":"beach","style":"cinematic",'
        '"motion_suggestions":["slow dolly in"],"constraints":["keep face stable"]}'
    )
    client = httpx.AsyncClient(transport=make_transport(content))
    gateway = VllmChatGateway(
        base_url="http://vllm.test/v1",
        api_key="test",
        model="vision",
        client=client,
    )
    analyzer = VllmVisionAnalyzer(gateway)

    analysis = await analyzer.analyze(
        ImageAsset(path=image_path, mime_type="image/png", size_bytes=5),
        "make it move",
    )

    assert analysis.subject == "cat"
    assert analysis.motion_suggestions == ["slow dolly in"]
    await client.aclose()


async def test_prompt_enhancer_maps_vllm_json() -> None:
    client = httpx.AsyncClient(
        transport=make_transport(
            '{"positive":"cinematic cat, slow camera",'
            '"negative":"blur", "camera_motion":"dolly in", "style":"film"}'
        )
    )
    gateway = VllmChatGateway(
        base_url="http://vllm.test/v1",
        api_key="test",
        model="text",
        client=client,
    )
    enhancer = VllmPromptEnhancer(gateway)

    enhanced = await enhancer.enhance(
        "cat",
        ImageAnalysis(subject="cat", scene="room"),
    )

    assert enhanced.positive == "cinematic cat, slow camera"
    assert enhanced.camera_motion == "dolly in"
    await client.aclose()
