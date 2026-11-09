# Agent Memory Workspace

A local-first conversational workspace that pairs durable, searchable memory with live access to structured data. It includes a React interface, a FastAPI service, SQLite persistence, an LLM-powered tool-using assistant, source citations, and a synthetic email/calendar/contacts/Slack dataset.

## Run locally

Requirements: Python 3.10+, [`uv`](https://docs.astral.sh/uv/), Node.js, npm, and an OpenAI API key.

```sh
uv sync
cd frontend && npm ci && npm run build && cd ..
export OPENAI_API_KEY=your-key
uv run uvicorn memory_layer.lab:app --host 127.0.0.1 --port 8013
```

Open <http://127.0.0.1:8013/workspace>. Model requests incur provider charges. You can instead put `OPENAI_API_KEY` in a local `.env` file; `.env` is ignored by Git. Never commit API keys.

Run the backend checks with `uv run pytest`. See [`memory_layer/README.md`](memory_layer/README.md) for implementation details, data boundaries, and privacy notes. The architecture visualizer is at [`docs/memory-architecture-visualizer.html`](docs/memory-architecture-visualizer.html).

## Design

- **Memory:** useful durable facts and preferences are stored as free text and retrieved semantically; structured current records remain in SQLite source tables.
- **Evidence:** the assistant can cite the chat message or source record supporting an answer. A citation identifies evidence, not a guarantee that an interpretation is correct.
- **Actions:** email, Slack, and booking actions operate on mock data and require confirmation. They never contact real services.
- **Privacy:** the included persona, conversations, and source records are synthetic. The local app and diagnostic logs can contain conversation and source content; keep them on trusted machines and do not publish runtime databases or logs.

## Configuration

`OPENAI_API_KEY` is required for model-backed chat. Optional model and tracing settings are listed in [`.env.example`](.env.example). Diagnostics that record full model context are opt-in and may log conversation and source content.
