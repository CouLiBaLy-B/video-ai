# MVP Status

## Functional today

- FastAPI backend for text + image generation jobs.
- React/Vite Claude-style interface.
- Upload validation and sanitization for PNG/JPEG/WebP images.
- Agentic workflow planning with a deterministic planner and DeepAgents integration point.
- vLLM OpenAI-compatible gateways for text and vision models.
- vLLM health checks and fallback-to-mock adapters.
- Mock video generator for full GPU-free end-to-end workflow.
- Lazy LTX-Video adapter with parameter validation and GPU runtime guards.
- Human approval before expensive GPU generation backends.
- Cancellation, rerun, and seed-variant controls.
- Job history and lifecycle event timeline.
- SQLite and Postgres-compatible repositories.
- Redis/RQ worker adapter and Docker worker profile.
- Structured JSON logs, request ids, health and metrics endpoints.
- Baseline security headers and in-memory rate limiting.

## Mocked or prepared but not fully production-run in this environment

- Real LTX-Video generation requires a CUDA machine and `pip install -e '.[video]'`.
- Real vLLM mode requires separate vLLM text and vision servers.
- Redis/RQ mode requires Redis and a shared persistent repository.
- Postgres mode requires a reachable Postgres instance and Alembic migration.
- DeepAgents runtime requires installing the `agents` extra and model credentials.
- Wan I2V is represented as a planned profile; the concrete adapter is not implemented yet.

## Recommended local MVP path

1. Run the API and web app with mock defaults.
2. Validate full prompt + image -> mock video flow.
3. Enable SQLite for persistence.
4. Launch vLLM text/vision and switch `AI_PROVIDER=vllm`.
5. Move long jobs to Redis/RQ.
6. Enable LTX-Video on a CUDA host.

## Quality status

Current validation suite:

```bash
python -m ruff check apps/api/src apps/api/tests migrations
python -m mypy apps/api/src/video_ai
python -m pytest apps/api/tests
cd apps/web && npm run build
```

At the time this document was written:

- Backend tests: 69 passing.
- Backend typing: mypy success.
- Backend lint: ruff success.
- Frontend build: Vite build success.
