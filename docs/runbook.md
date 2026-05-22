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


## Runtime health

Check API, storage, repository and vLLM endpoints:

```bash
curl http://localhost:8000/api/system/health
```

Check configured capabilities visible to the UI:

```bash
curl http://localhost:8000/api/system/capabilities
```

## vLLM local launch examples

Text model:

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --host 0.0.0.0 \
  --port 8000
```

Vision model:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --limit-mm-per-prompt image=1
```

Then configure the API:

```bash
AI_PROVIDER=vllm
VLLM_TEXT_BASE_URL=http://localhost:8000/v1
VLLM_VISION_BASE_URL=http://localhost:8001/v1
```
