# video-ai

Agentic text + image to video generation platform.

## Goal

Build a Claude-style interface where a user sends:

- a text prompt
- an image

and receives a generated video.

The system is designed around:

- **DeepAgents / LangChain / LangGraph** for planning, subagents, skills, memory and human-in-the-loop workflows.
- **vLLM** for OpenAI-compatible serving of open-source LLM/VLM models.
- **Open-source video models** from Hugging Face, starting with a mock generator for tests, then LTX-Video and Wan I2V adapters.

## Repository layout

```text
apps/
  api/     FastAPI backend and agent orchestration
  web/     React/Vite frontend
docs/      architecture and implementation documentation
```

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
ruff check .
mypy
```

Frontend:

```bash
cd apps/web
npm install
npm run dev
```
