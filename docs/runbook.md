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

## LTX-Video production preparation

Validate parameters before an expensive GPU run:

```bash
curl -X POST http://localhost:8000/api/system/ltx/validate \
  -H 'Content-Type: application/json' \
  -d '{
    "width": 768,
    "height": 512,
    "num_frames": 121,
    "fps": 24,
    "inference_steps": 30,
    "guidance_scale": 3.5
  }'
```

Recommended first GPU configuration:

```env
VIDEO_GENERATOR_BACKEND=ltx-video
LTX_VIDEO_MODEL_ID=Lightricks/LTX-Video
LTX_VIDEO_DEVICE=cuda
LTX_VIDEO_TORCH_DTYPE=bfloat16
```

Install video dependencies in a CUDA machine:

```bash
pip install -e '.[video]'
```

Operational notes:

- Prefer dimensions divisible by 32.
- Prefer frame counts of the form `8n+1`, for example `49`, `73`, `121`.
- Start with `768x512`, `121` frames, `24` FPS, `20-30` steps.
- Use a fixed seed for reproducibility.
- If `LTX_VIDEO_DEVICE=cuda` and CUDA is not available, the adapter fails fast with a clear runtime error.
- The adapter only forwards allowlisted advanced kwargs from `GenerationParameters.extra`.

## Human approval for GPU jobs

Real GPU backends such as `ltx-video` are paused before generation and placed in:

```text
waiting_for_approval
```

Approve the job:

```bash
curl -X POST http://localhost:8000/api/generations/<job_id>/approve
```

Reject/cancel the job:

```bash
curl -X POST http://localhost:8000/api/generations/<job_id>/reject
```

The UI displays a validation panel with `Valider la génération GPU` when approval is required.
Mock generations do not require approval and continue automatically.


## Cancellation

Cancel a queued, planning, generating, reviewing, or approval-waiting job:

```bash
curl -X POST http://localhost:8000/api/generations/<job_id>/cancel
```

Cancellation is checked before queued background tasks and before approved GPU execution starts.
For long-running GPU kernels already inside a model call, cancellation is best-effort until the worker returns.
