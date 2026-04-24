# GenAI Developer Challenge

End-to-end GenAI application with a RAG pipeline, LLM integration, and a React frontend.  
Built as a technical challenge to demonstrate GenAI development, API design, and modern ML engineering practices.

---

## Table of Contents

- [Goal](#goal)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack & Justification](#tech-stack--justification)
- [Prerequisites](#prerequisites)
- [Setup & Run](#setup--run)
  - [Local Development](#local-development)
  - [Docker](#docker)
- [API Reference](#api-reference)
- [RAG Pipeline](#rag-pipeline)
- [Prompt Engineering](#prompt-engineering)
- [Configuration](#configuration)
- [Testing](#testing)
- [Cloud Deployment](#cloud-deployment)
- [GenAI Coding Assistants](#genai-coding-assistants)

---

## Goal

Build an end-to-end GenAI application that:

1. Exposes a **REST API** (FastAPI) for LLM-powered chat and document-grounded Q&A.
2. Implements a **RAG pipeline** — ingest documents, embed them, store in a vector DB, and retrieve relevant context at query time.
3. Provides a **React frontend** with separate Chat and RAG Q&A modes.
4. Runs entirely on **free/open-source** tools — local Ollama LLM, local Qdrant, open-source embeddings.

---

## Architecture

```mermaid
graph TD
    User["User (Browser)"]

    subgraph Frontend["Frontend — React + Vite :5173"]
        UI_Chat["Chat Tab"]
        UI_RAG["RAG Q&A Tab"]
    end

    subgraph Backend["Backend — FastAPI :8000"]
        HC["GET /api/v1/healthcheck"]
        Chat["POST /api/v1/chat"]
        RAG["POST /api/v1/rag-query"]

        subgraph Services
            LLMService["LLM Service"]
            RAGService["RAG Service"]
            EmbedService["Embedding Service"]
        end

        subgraph Adapters
            OllamaAdapter["Ollama Adapter"]
            QdrantAdapter["Qdrant Adapter"]
        end
    end

    subgraph Infra["Infrastructure"]
        Ollama["Ollama :11434\nqwen3.5:4b"]
        Qdrant["Qdrant :6333\nVector DB"]
    end

    subgraph Ingest["Offline — Ingest Script"]
        PDFs["PDF Documents\n/data/"]
        Chunker["Text Splitter"]
        Embedder["FastEmbed\nBAAI/bge-small-en-v1.5"]
    end

    User --> UI_Chat & UI_RAG
    UI_Chat -- "POST /api/v1/chat" --> Chat
    UI_RAG -- "POST /api/v1/rag-query" --> RAG

    Chat --> LLMService --> OllamaAdapter --> Ollama
    RAG --> RAGService --> EmbedService --> QdrantAdapter --> Qdrant
    RAGService --> LLMService

    PDFs --> Chunker --> Embedder --> Qdrant
```

**Request flow — RAG query:**

```
User query
  → FastAPI /api/v1/rag-query
    → embed query (FastEmbed)
    → search Qdrant (top-k chunks)
    → build grounded prompt (retrieved context + query)
    → call Ollama (qwen3.5:4b)
    → return answer + sources
```

---

## Project Structure

```
Challenge/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app, CORS, lifespan
│   │   ├── core/
│   │   │   ├── config.py           # Pydantic Settings (env-driven)
│   │   │   └── logging.py          # Structured JSON logging
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── router.py
│   │   │       └── routes/
│   │   │           ├── health.py   # GET /api/v1/healthcheck
│   │   │           ├── chat.py     # POST /api/v1/chat
│   │   │           └── rag.py      # POST /api/v1/rag-query
│   │   ├── services/
│   │   │   ├── llm_service.py      # LLM call + session memory
│   │   │   ├── rag_service.py      # Retrieve → prompt → answer
│   │   │   └── embedding_service.py
│   │   └── adapters/
│   │       ├── ollama_adapter.py   # langchain-ollama wrapper
│   │       └── qdrant_adapter.py   # qdrant-client wrapper
│   ├── scripts/
│   │   └── ingest.py               # Load PDFs → chunk → embed → store
│   ├── tests/
│   │   ├── test_health.py
│   │   ├── test_chat.py
│   │   └── test_rag.py
│   ├── Dockerfile
│   ├── pyproject.toml              # Dependencies managed with uv
│   └── .env.example
├── frontend/                       # React + Vite (Phase 4)
├── data/                           # Drop your PDF files here
├── docker-compose.yml
├── Makefile
└── README.md
```

---

## Tech Stack & Justification

| Layer | Tool | Version | Why |
|---|---|---|---|
| API framework | FastAPI | 0.136.x | Async-native, automatic OpenAPI docs, Pydantic-first |
| Runtime | Python | 3.12 | Latest stable, full type narrowing support |
| Dependency manager | uv | 0.8.x | 10–100× faster than pip, lock-file reproducibility |
| LLM runtime | Ollama | local | Zero-cost local inference, OpenAI-compatible API |
| LLM model | qwen3.5:4b | — | Strong instruction-following at 4B params, fast on CPU/Apple Silicon |
| LLM integration | langchain-ollama | 1.1.x | Thin, well-maintained wrapper; brings conversation memory |
| Embeddings | FastEmbed (`BAAI/bge-small-en-v1.5`) | 0.8.x | ~25 MB model, downloads automatically, strong retrieval quality |
| Vector DB | Qdrant | 1.17.x | Persistent, supports hybrid search & metadata filtering, first-class FastEmbed support |
| Text splitting | langchain-text-splitters | 1.1.x | Recursive character splitter with configurable overlap |
| PDF loading | pypdf | 6.x | Lightweight, no Java dependency |
| Validation | Pydantic v2 | 2.13.x | Request/response models + settings management |
| Frontend | React + Vite | — | Lightweight, fast HMR, no heavyweight framework |
| Testing | pytest + pytest-asyncio | 9.x / 1.3.x | Async-native test runner |
| Linting | Ruff | 0.15.x | Replaces flake8 + isort + pyupgrade in one tool |
| Containerisation | Docker + Compose | — | One-command local stack |

---

## Prerequisites

| Tool | Install |
|---|---|
| Python 3.12+ | `uv python install 3.12` |
| [uv](https://docs.astral.sh/uv/) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | Download from Docker website |
| [Ollama](https://ollama.com/) | Download from ollama.com |

Pull the required Ollama model:

```bash
ollama pull qwen3.5:4b
```

---

## Setup & Run

### Local Development

**1. Clone and enter the repo**

```bash
git clone <your-repo-url>
cd Challenge
```

**2. Start Qdrant**

```bash
make up
# or: docker compose up -d
```

**3. Install backend dependencies**

```bash
make install
# or: cd backend && uv sync
```

**4. Configure environment**

```bash
cp backend/.env.example backend/.env
# Edit backend/.env if your Ollama URL or model name differs
```

**5. Start the API**

```bash
make dev
# or: cd backend && uv run uvicorn app.main:app --reload
```

**6. Verify**

```bash
curl http://localhost:8000/api/v1/healthcheck
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "environment": "development",
  "model": "qwen3.5:4b"
}
```

**7. Ingest documents** *(after dropping PDFs into `data/`)*

```bash
make ingest
# or: cd backend && uv run python scripts/ingest.py
```

**8. Start the frontend** *(Phase 4)*

```bash
cd frontend && npm install && npm run dev
# Open http://localhost:5173
```

---

### Docker

Bring up the full stack (Qdrant + backend) with one command:

```bash
docker compose up
```

> **Note:** When running via Docker, the backend connects to Qdrant at `http://qdrant:6333` (set automatically). Ollama must still run on the host machine. Make sure Ollama is running before starting the stack.

---

## API Reference

### `GET /api/v1/healthcheck`

Returns service status.

```bash
curl http://localhost:8000/api/v1/healthcheck
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "environment": "development",
  "model": "qwen3.5:4b"
}
```

---

### `POST /api/v1/chat`

Chat with the LLM directly. Supports multi-turn conversation via `session_id`.

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is retrieval-augmented generation?",
    "session_id": "user-abc-123"
  }'
```

```json
{
  "reply": "Retrieval-Augmented Generation (RAG) is a technique that...",
  "session_id": "user-abc-123",
  "model": "qwen3.5:4b",
  "meta": {
    "latency_ms": 1240
  }
}
```

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | ✓ | User prompt |
| `session_id` | string | — | Groups messages into a conversation |
| `model_name` | string | — | Override the default model |

---

### `POST /api/v1/rag-query`

Ask a question grounded in your ingested documents.

```bash
curl -X POST http://localhost:8000/api/v1/rag-query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does chapter 3 say about clean code?",
    "top_k": 5
  }'
```

```json
{
  "answer": "According to the retrieved documents, chapter 3 discusses...",
  "sources": [
    {
      "chunk_id": "doc_003_chunk_12",
      "source": "clean_code_part3.pdf",
      "page": 3,
      "score": 0.91
    }
  ]
}
```

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | ✓ | User question |
| `top_k` | int | — | Number of chunks to retrieve (default: 5) |
| `source_filter` | string | — | Filter by document filename |

---

## RAG Pipeline

### Document Ingestion

Place PDF files in the `data/` directory, then run:

```bash
make ingest
```

The script:
1. Loads each PDF with `pypdf`
2. Splits into chunks (`chunk_size=512`, `overlap=64`) using `RecursiveCharacterTextSplitter`
3. Embeds each chunk with FastEmbed (`BAAI/bge-small-en-v1.5`)
4. Stores vectors + metadata in Qdrant collection `documents`

Metadata stored per chunk:

| Field | Description |
|---|---|
| `source` | Original filename |
| `page` | Page number |
| `chunk_id` | Unique identifier |

### Retrieval

At query time:
1. The query is embedded with the same FastEmbed model
2. Qdrant performs a cosine similarity search and returns `top_k` chunks
3. The retrieved chunks are injected into the LLM prompt as context

---

## Prompt Engineering

### RAG System Prompt

The RAG endpoint uses a strict grounding prompt to reduce hallucinations:

```
You are a helpful assistant that answers questions strictly based on the provided context.
If the answer cannot be found in the context, say "I don't have enough information in the
provided documents to answer this question." Do not use prior knowledge beyond the context.

Context:
{retrieved_chunks}

Question: {user_query}
Answer:
```

**Why this reduces hallucinations:**
- The explicit instruction *"strictly based on the provided context"* suppresses the model's tendency to fill gaps with training data.
- The fallback phrase *"I don't have enough information"* gives the model a safe exit instead of fabricating an answer.
- Context is injected before the question so the model attends to it first.

### Chat System Prompt

The direct chat endpoint uses a general assistant prompt with few-shot formatting guidance to keep responses concise and structured.

---

## Configuration

All configuration is driven by environment variables. Copy `.env.example` to `.env` and adjust:

```bash
cp backend/.env.example backend/.env
```

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | `development` or `production` |
| `APP_VERSION` | `0.1.0` | Reported in healthcheck |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen3.5:4b` | Model name as listed by `ollama list` |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant REST API URL |
| `QDRANT_COLLECTION_NAME` | `documents` | Collection to store/query embeddings |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed model (downloaded on first run) |
| `RAG_TOP_K` | `5` | Default number of chunks to retrieve |
| `RAG_CHUNK_SIZE` | `512` | Tokens per chunk during ingestion |
| `RAG_CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |

---

## Testing

```bash
# Run all tests
make test
# or: cd backend && uv run pytest -v

# Run with coverage (coming)
cd backend && uv run pytest --cov=app -v
```

Current test coverage:

| Test | Status |
|---|---|
| `test_health.py` — healthcheck returns 200 + ok | ✓ |
| `test_chat.py` — chat endpoint (Phase 2) | pending |
| `test_rag.py` — RAG query endpoint (Phase 3) | pending |

---

## Cloud Deployment

### Strategy (Azure Container Apps)

The recommended deployment path uses **Azure Container Apps** (free tier available):

```bash
# 1. Build and push images
docker build -t <registry>/genai-backend:latest ./backend
docker push <registry>/genai-backend:latest

# 2. Deploy Qdrant as a Container App with persistent volume
az containerapp create \
  --name qdrant \
  --image qdrant/qdrant:v1.17.1 \
  --target-port 6333

# 3. Deploy backend
az containerapp create \
  --name genai-backend \
  --image <registry>/genai-backend:latest \
  --env-vars QDRANT_URL=<qdrant-internal-url> OLLAMA_BASE_URL=<ollama-url>
```

> For Ollama in the cloud, use a GPU-enabled VM or substitute with a free-tier cloud LLM (e.g. Hugging Face Inference API) by changing `OLLAMA_BASE_URL`.

**Deployment status:** Local only (no public URL at this stage).

---

## GenAI Coding Assistants

This project was built using **Claude Code** (Anthropic's CLI coding assistant) as the primary AI-assisted development tool.

**How it helped:**
- Scaffolding the layered FastAPI architecture (`api/`, `services/`, `adapters/`, `core/`)
- Resolving dependency version conflicts across the LangChain + Qdrant + FastEmbed ecosystem
- Generating Pydantic models, structured logging, and test boilerplate
- Verifying library versions against PyPI in real time before pinning them

**Limitations and corrections:**
- Initial dependency draft used `qdrant-client[fastembed]` which pins `fastembed<0.8`. This was caught by querying PyPI directly and resolved by using `fastembed>=0.8.0` standalone (no extra).
- The Qdrant Docker image version in the first draft was outdated (`v1.13.6`); corrected to `v1.17.1` after a live registry check.

All generated code was reviewed, tested, and adjusted before committing.
