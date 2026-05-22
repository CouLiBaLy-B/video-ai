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


## Redis/RQ production worker

The API can enqueue generation jobs into Redis Queue instead of FastAPI background tasks.
Use this mode for long GPU jobs and separate worker processes.

Recommended production-like configuration:

```env
TASK_QUEUE_BACKEND=redis-rq
JOB_REPOSITORY_BACKEND=sqlite
SQLITE_DATABASE_PATH=.data/video_ai.sqlite3
REDIS_URL=redis://localhost:6379/0
RQ_QUEUE_NAME=video-ai
```

Install worker dependencies:

```bash
pip install -e '.[worker]'
```

Run an RQ worker locally:

```bash
PYTHONPATH=apps/api/src rq worker video-ai --url redis://localhost:6379/0
```

Run with Docker Compose:

```bash
docker compose --profile prod up --build api worker redis
```

Worker entrypoints are defined in:

```text
video_ai.workers.jobs.run_job_task
video_ai.workers.jobs.run_approved_job_task
```

Important: when using `redis-rq`, use a persistent job repository shared by API and worker,
such as SQLite for local production-like runs or Postgres in a future deployment.
Do not use the in-memory repository with separate worker processes.


## Postgres repository

Use Postgres for API/worker deployments where multiple processes need shared job state.

Configuration:

```env
JOB_REPOSITORY_BACKEND=postgres
DATABASE_URL=postgresql+psycopg://video_ai:video_ai_dev@localhost:5432/video_ai
```

Install database dependencies:

```bash
pip install -e '.[db]'
```

Run migrations:

```bash
DATABASE_URL=postgresql+psycopg://video_ai:video_ai_dev@localhost:5432/video_ai alembic upgrade head
```

For production-like local Docker with API, worker, Redis and Postgres:

```bash
docker compose --profile prod up --build api worker redis postgres
```

The current SQLAlchemy repository auto-creates the `generation_jobs` table as a safety net,
but Alembic migrations are the recommended production path.


## Observability

The API emits JSON structured logs with request timing and `x-request-id`.

Health endpoint:

```bash
curl http://localhost:8000/api/system/health
```

Metrics endpoint:

```bash
curl http://localhost:8000/api/system/metrics
```

Metrics currently include job counts by lifecycle status. The frontend displays a compact
summary for total, completed, failed and cancelled jobs.

## Security controls

The API adds baseline security headers to every response:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: no-referrer`
- `Permissions-Policy: camera=(), microphone=(), geolocation=()`

Rate limiting is enabled by default for non-health endpoints with an in-memory per-client sliding window:

```env
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_WINDOW_SECONDS=60
```

When exceeded, the API returns:

```http
429 Too Many Requests
Retry-After: <seconds>
```

For multi-process production deployments, prefer a shared Redis/API-gateway limiter. The in-memory limiter is intended for local, development, and single-process deployments.

Image upload security is configurable:

```env
MAX_UPLOAD_BYTES=10485760
MAX_IMAGE_PIXELS=16000000
```

## Product iteration controls

Generation history is available from:

```bash
curl http://localhost:8000/api/generations
```

Rerun a job with the same prompt, image and preferences:

```bash
curl -X POST http://localhost:8000/api/generations/<job_id>/rerun
```

Create a new seed variant:

```bash
curl -X POST http://localhost:8000/api/generations/<job_id>/variant
```

The frontend sidebar displays recent generations and lets the user reload a job, rerun it, or create a new seed variant.


## Phase 1 smoke validation

Run a dry-run LTX parameter validation without loading GPU models:

```bash
make smoke-ltx
```

Run a real LTX smoke generation on a CUDA host:

```bash
VIDEO_GENERATOR_BACKEND=ltx-video \
LTX_VIDEO_DEVICE=cuda \
python scripts/smoke_ltx.py --run --image examples/input.png
```

Run real vLLM text + vision smoke tests after starting vLLM servers:

```bash
AI_PROVIDER=vllm \
VLLM_TEXT_BASE_URL=http://localhost:8000/v1 \
VLLM_VISION_BASE_URL=http://localhost:8001/v1 \
VLLM_FALLBACK_TO_MOCK=false \
python scripts/smoke_vllm.py --image examples/input.png
```

## Phase 2 production-like stack validation

For API + Redis/RQ + Postgres, use explicit production-like env values:

```bash
TASK_QUEUE_BACKEND=redis-rq \
JOB_REPOSITORY_BACKEND=postgres \
VIDEO_GENERATOR_BACKEND=mock \
docker compose --profile prod up --build api worker redis postgres
```

Then in another shell create a mock job through the UI or API and verify that:

1. the API enqueues the job in RQ;
2. the worker consumes it;
3. Postgres stores status transitions;
4. `/api/system/metrics` shows the completed job.

For local `mock` profile without Postgres/Redis, the API defaults to `JOB_REPOSITORY_BACKEND=memory` and `TASK_QUEUE_BACKEND=fastapi`.
