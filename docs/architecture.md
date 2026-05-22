# Architecture

```text
React/Vite UI
  -> FastAPI API
    -> DeepAgents Orchestrator
      -> vLLM text model
      -> vLLM vision model
      -> Video generation worker
    -> Storage
```

The backend follows a ports-and-adapters architecture:

- `domain`: framework-independent entities and contracts.
- `application`: use cases and orchestration services.
- `infrastructure`: vLLM, storage, video model adapters.
- `interfaces`: HTTP API and schemas.
- `agents`: DeepAgents integration.
- `workers`: generation worker runtime.
