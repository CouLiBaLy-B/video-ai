"""vLLM OpenAI-compatible gateway adapters."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx

from video_ai.domain.models import EnhancedPrompt, ImageAnalysis, ImageAsset


class VllmGatewayError(RuntimeError):
    """Raised when a vLLM gateway request fails."""


class VllmChatGateway:
    """Small OpenAI-compatible client for vLLM chat completions."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> VllmChatGateway:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP client if owned by the gateway."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()

    @property
    def model(self) -> str:
        """Return configured model id."""
        return self._model

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """Call `/chat/completions` and return assistant text."""
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        client = self._client or httpx.AsyncClient(timeout=self._timeout_seconds)
        try:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise VllmGatewayError(f"vLLM request failed: {exc}") from exc

        data = response.json()
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise VllmGatewayError("Invalid vLLM chat completion response") from exc

    async def complete_json(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Call vLLM and parse a JSON object response."""
        content = await self.complete(
            messages,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise VllmGatewayError("vLLM response is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise VllmGatewayError("vLLM JSON response must be an object")
        return parsed


class VllmVisionAnalyzer:
    """Vision analyzer implementation backed by a vLLM VLM endpoint."""

    def __init__(self, gateway: VllmChatGateway) -> None:
        self._gateway = gateway

    async def analyze(self, image: ImageAsset, prompt: str) -> ImageAnalysis:
        """Analyze an image and return structured context."""
        encoded = _encode_image_data_url(image.path, image.mime_type)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "You analyze images for image-to-video generation. Return strict JSON.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _vision_prompt(prompt)},
                    {"type": "image_url", "image_url": {"url": encoded}},
                ],
            },
        ]
        data = await self._gateway.complete_json(messages, max_tokens=900)
        return ImageAnalysis(
            subject=str(data.get("subject", "unknown subject")),
            scene=str(data.get("scene", "unknown scene")),
            style=_optional_str(data.get("style")),
            lighting=_optional_str(data.get("lighting")),
            composition=_optional_str(data.get("composition")),
            motion_suggestions=_str_list(data.get("motion_suggestions")),
            constraints=_str_list(data.get("constraints")),
        )


class VllmPromptEnhancer:
    """Prompt enhancer backed by a vLLM text endpoint."""

    def __init__(self, gateway: VllmChatGateway) -> None:
        self._gateway = gateway

    async def enhance(self, prompt: str, analysis: ImageAnalysis) -> EnhancedPrompt:
        """Generate a video-model-ready prompt package."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a cinematic prompt engineer for image-to-video "
                    "models. Return JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Create a concise image-to-video prompt package.\n"
                    f"User prompt: {prompt}\n"
                    f"Image analysis: {analysis.model_dump_json()}\n"
                    "JSON keys: positive, negative, camera_motion, style."
                ),
            },
        ]
        data = await self._gateway.complete_json(messages, max_tokens=900)
        return EnhancedPrompt(
            positive=str(data.get("positive", prompt)),
            negative=str(
                data.get("negative", "low quality, blurry, distorted, flickering, artifacts")
            ),
            camera_motion=_optional_str(data.get("camera_motion")),
            style=_optional_str(data.get("style")),
        )


def _encode_image_data_url(path: Path, mime_type: str) -> str:
    content = path.read_bytes()
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _vision_prompt(prompt: str) -> str:
    return (
        "Analyze this image for an image-to-video generation pipeline. "
        "Preserve identity and important visual constraints. "
        f"User prompt: {prompt}. "
        "Return JSON with subject, scene, style, lighting, composition, "
        "motion_suggestions array, constraints array."
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


class VllmHealthChecker:
    """Health checker for OpenAI-compatible vLLM endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> VllmHealthChecker:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP client if owned by the checker."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()

    async def check(self) -> bool:
        """Return True when the vLLM endpoint responds to `/models`."""
        client = self._client or httpx.AsyncClient(timeout=self._timeout_seconds)
        try:
            response = await client.get(
                f"{self._base_url}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            return response.status_code < 500 and response.status_code != 404
        except httpx.HTTPError:
            return False
        finally:
            if self._client is None:
                await client.aclose()
