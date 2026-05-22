# Runbook

## Mock profile

Runs the full API/UI workflow without GPU using the mock generator.

```bash
docker compose --profile mock up --build
```

Or locally:

```bash
PYTHONPATH=apps/api/src uvicorn video_ai.main:app --reload
cd apps/web && npm run dev
```

## vLLM profile

Requires NVIDIA Docker runtime and model access.

```bash
VLLM_TEXT_MODEL=Qwen/Qwen3.6-35B-A3B \
VLLM_VISION_MODEL=Qwen/Qwen2.5-VL-7B-Instruct \
docker compose --profile vllm up
```

## LTX-Video

Install the Python `video` extra in a GPU environment, then switch the generator
backend from `mock` to `ltx-video` once runtime wiring is enabled.
