# video-ai

Agentic text + image to video generation platform.

The application provides a Claude-style UI where a user sends:

- a text prompt
- an image

and receives a generated video.

## Core stack

- **Backend**: FastAPI, Pydantic, ports/adapters architecture.
- **Frontend**: React/Vite.
- **Agentic workflow**: DeepAgents integration point, specialized subagents, reusable skills.
- **LLM/VLM serving**: vLLM OpenAI-compatible gateways.
- **Video generation**: mock generator for local MVP, lazy LTX-Video adapter for CUDA hosts.
- **Persistence**: memory, SQLite, SQLAlchemy/Postgres.
- **Workers**: FastAPI background tasks for local dev, Redis/RQ for production-like workers.
- **Ops**: JSON logs, health, metrics, rate limiting, security headers.

## Repository layout

```text
apps/
  api/     FastAPI backend and agent orchestration
  web/     React/Vite frontend
docs/      architecture, runbook, MVP status
skills/    DeepAgents skills
docker/    API and web Dockerfiles
migrations/ Alembic migrations
```

## Quick start: mock mode

Backend:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
PYTHONPATH=apps/api/src uvicorn video_ai.main:app --reload
```

Frontend:

```bash
cd apps/web
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

The default mode is GPU-free and uses:

```env
AI_PROVIDER=mock
VIDEO_GENERATOR_BACKEND=mock
TASK_QUEUE_BACKEND=fastapi
JOB_REPOSITORY_BACKEND=memory
```

## Quality checks

```bash
make quality
```

Equivalent commands:

```bash
python -m ruff check apps/api/src apps/api/tests migrations
python -m mypy apps/api/src/video_ai
python -m pytest apps/api/tests
cd apps/web && npm run build
```

## vLLM mode

Launch text and vision servers separately, then configure:

```env
AI_PROVIDER=vllm
VLLM_TEXT_BASE_URL=http://localhost:8000/v1
VLLM_VISION_BASE_URL=http://localhost:8001/v1
VLLM_FALLBACK_TO_MOCK=true
```

See `docs/runbook.md` for launch examples.

## LTX-Video mode

Requires CUDA and video dependencies:

```bash
pip install -e '.[video]'
```

Configuration:

```env
VIDEO_GENERATOR_BACKEND=ltx-video
LTX_VIDEO_MODEL_ID=Lightricks/LTX-Video
LTX_VIDEO_DEVICE=cuda
LTX_VIDEO_TORCH_DTYPE=bfloat16
```

GPU jobs pause at `waiting_for_approval` until approved.

## Production-like worker mode

```env
TASK_QUEUE_BACKEND=redis-rq
JOB_REPOSITORY_BACKEND=postgres
DATABASE_URL=postgresql+psycopg://video_ai:video_ai_dev@localhost:5432/video_ai
REDIS_URL=redis://localhost:6379/0
RQ_QUEUE_NAME=video-ai
```

Run migrations:

```bash
alembic upgrade head
```

Docker Compose:

```bash
docker compose --profile prod up --build api worker redis postgres
```

## Useful endpoints

```http
GET  /health
GET  /api/system/capabilities
GET  /api/system/health
GET  /api/system/metrics
POST /api/generations
GET  /api/generations
GET  /api/generations/{job_id}
POST /api/generations/{job_id}/approve
POST /api/generations/{job_id}/reject
POST /api/generations/{job_id}/cancel
POST /api/generations/{job_id}/rerun
POST /api/generations/{job_id}/variant
```

## Documentation

- `docs/architecture.md`
- `docs/implementation-plan.md`
- `docs/runbook.md`
- `docs/MVP_STATUS.md`
