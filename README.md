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
- [Observability — LangFuse](#observability--langfuse)
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
├── frontend/
│   ├── src/
│   │   ├── App.tsx                 # Tab switcher (Chat / RAG Q&A)
│   │   ├── types.ts
│   │   ├── api/client.ts           # Typed fetch wrappers (no direct LLM calls)
│   │   └── components/
│   │       ├── ConversationPane.tsx
│   │       ├── MessageBubble.tsx   # Markdown rendering, source cards
│   │       ├── SourceCard.tsx      # Collapsible source with images
│   │       └── LoadingBubble.tsx
│   ├── Dockerfile                  # Multi-stage: node build → nginx serve
│   ├── nginx.conf                  # SPA routing + static asset caching
│   ├── package.json
│   └── vite.config.ts
├── data/                           # 10 PDFs — Listado de las Aves Argentinas
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
| PDF loading | PyMuPDF (`fitz`) | 1.27.x | Fast text + image extraction from PDF pages |
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

> `make install` also installs the pre-push git hook that runs lint + tests locally before every push.

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

**8. Start the frontend**

```bash
make frontend-install   # first time only
make frontend-dev
# or: cd frontend && npm install && npm run dev
# Open http://localhost:5173
```

---

### Docker

Bring up the full stack (Qdrant + backend + frontend):

```bash
# First run, or after any source change:
make up-build

# Subsequent runs (images already built):
make up
```

- Frontend → http://localhost
- Backend API → http://localhost:8000

> **Note:** The frontend image bakes in `VITE_API_BASE_URL=http://localhost:8000` at build time, so the browser talks directly to the backend on your machine. Ollama must still run on the host — make sure it's running before starting the stack.

---

## Authentication

The chat and RAG endpoints are protected with JWT Bearer tokens. The healthcheck is public.

**Default credentials** (set in `backend/.env`):

| Variable | Default |
|---|---|
| `AUTH_USERNAME` | `admin` |
| Password | `changeme` |
| `JWT_SECRET_KEY` | `change-me-in-production` |

> Change these before any real deployment. Generate a secret with `openssl rand -hex 32` and a new password hash with `make hash-password`.

### Frontend design decision

The frontend hardcodes the service account credentials in `src/api/client.ts` and automatically exchanges them for a JWT on the first request. This is a **conscious tradeoff**: in a production app the frontend would show a login screen so the user supplies the password and no secret ever lives in client-side code. For this demo we skip the login screen to keep the UI simple — the credentials are visible to anyone who reads the bundle.

The API still enforces JWT auth for every caller that isn't the frontend (curl, Postman, external integrations). That is where the security boundary actually matters for this challenge.

### Get a token

```bash
curl -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=admin&password=changeme"
```

```json
{ "access_token": "<jwt>", "token_type": "bearer" }
```

### Use the token

Pass it as a Bearer header on every subsequent request:

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=admin&password=changeme" | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

### Testing auth manually

**Step 1 — store a token in a shell variable:**

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=admin&password=changeme" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

echo $TOKEN   # should print a long JWT string
```

**Step 2 — working cases (expect 200):**

```bash
# Chat with valid token
curl -i -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'

# RAG with valid token
curl -i -X POST http://localhost:8000/api/v1/rag-query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What birds live in Argentina?"}'

# Healthcheck requires no token
curl -i http://localhost:8000/api/v1/healthcheck
```

**Step 3 — failing cases (expect 401):**

```bash
# No token
curl -i -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'

# Wrong password → token request fails
curl -i -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=admin&password=wrongpassword"

# Malformed token
curl -i -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer thisisnotavalidtoken" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

> The `-i` flag prints the HTTP status line (`HTTP/1.1 401 Unauthorized`) at the top of the response so you can confirm the code without reading the body.

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

> Requires a Bearer token — see [Authentication](#authentication) for how to obtain one.

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | ✓ | User prompt |
| `session_id` | string | — | Groups messages into a conversation |
| `model_name` | string | — | Override the default model |

**Response:**

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

---

### `POST /api/v1/rag-query`

Ask a question grounded in your ingested documents.

> Requires a Bearer token — see [Authentication](#authentication) for how to obtain one.

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | ✓ | User question |
| `top_k` | int | — | Number of chunks to retrieve (default: 5) |
| `source_filter` | string | — | Filter by document filename |

**Response:**

```json
{
  "answer": "The national bird of Argentina is the Rufous Hornero (Furnarius rufus)...",
  "sources": [
    {
      "chunk_id": "abc-123",
      "source": "LISTADO DE LAS AVES ARGENTINAS-1.pdf",
      "page": 4,
      "score": 0.92,
      "text_snippet": "El Hornero (Furnarius rufus) fue declarado ave nacional...",
      "image_urls": ["/images/LISTADO_DE_LAS_AVES_ARGENTINAS-1_p4_1.jpeg"]
    }
  ],
  "meta": {
    "latency_ms": 1840,
    "hits": 5
  }
}
```

---

## RAG Pipeline

### Document Ingestion

Place PDF files in the `data/` directory, then run:

```bash
make ingest
```

The script (`scripts/ingest.py`):
1. Loads each PDF with **PyMuPDF** (`fitz`)
2. Extracts page text and any embedded images (images &lt; 80×80 px are skipped as decorative noise)
3. Splits text into chunks (`chunk_size=512`, `overlap=64`) using `RecursiveCharacterTextSplitter`
4. Embeds each chunk with FastEmbed (`BAAI/bge-small-en-v1.5`, ~25 MB, auto-downloaded)
5. Stores vectors + metadata in Qdrant collection `documents`

Result for the Argentine Birds corpus: **~1,743 chunks** and **146 extracted images**.

Metadata stored per chunk:

| Field | Description |
|---|---|
| `source` | Original PDF filename |
| `page` | Page number |
| `chunk_id` | Unique UUID |
| `image_filenames` | List of images extracted from that page |

### Retrieval

At query time:
1. The query is embedded with the same FastEmbed model
2. Qdrant performs a cosine similarity search and returns `top_k` chunks
3. The retrieved chunks are injected into the LLM prompt as context

---

## Prompt Engineering

### RAG System Prompt

The RAG endpoint (`app/services/rag_service.py`) uses a domain-specific grounding prompt:

```
You are a knowledgeable ornithology assistant specialising in Argentine birds.
Answer the user's question based strictly on the context provided below.
Cite the source document and page number when relevant.
If the context does not contain enough information to answer the question, say:
"I don't have enough information in the provided documents to answer this question."
Do not use prior knowledge beyond what is in the context.
```

The prompt is constructed as:
```
[System prompt above]

Context:
[Source: filename.pdf, Page N]
<chunk text>

---

[Source: filename.pdf, Page M]
<chunk text>

Question: <user query>
```

**Why this reduces hallucinations:**
- **Domain framing** ("ornithology assistant") anchors the model's role and reduces off-topic drift.
- **"Strictly on the context"** suppresses the model's tendency to fill gaps with training data.
- **Explicit citation instruction** encourages the model to reference sources, making it auditable.
- **Named fallback phrase** gives the model a safe, pre-scripted exit instead of fabricating an answer.
- **Context injected before the question** ensures the model attends to retrieved chunks first.

### Chat System Prompt

The direct chat endpoint (`app/services/llm_service.py`) uses a concise general assistant prompt:

```
You are a knowledgeable and concise assistant.
Answer questions clearly and accurately based on your training knowledge.
When you are unsure about something, say so rather than guessing.
Keep responses focused and to the point — avoid unnecessary padding.
```

Full conversation history (up to 10 turns) is passed to the model on each request, enabling multi-turn dialogue.

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
| `test_health.py` — healthcheck returns 200 + `status: ok` | ✓ |
| `test_chat.py` — reply shape, empty-message rejection (422), session auto-generation | ✓ |
| `test_rag.py` — answer + sources shape, empty-query rejection (422), `top_k` out-of-range (422), source filter passthrough | ✓ |

Chat and RAG tests mock the service layer so they run without a live Ollama or Qdrant instance.

---

## Observability — LangFuse

LangFuse is integrated for end-to-end tracing of every LLM call — both direct chat and RAG queries. Each trace captures the full prompt, the model response, latency, and the session ID, visible in the LangFuse dashboard.

**Tracing is optional.** If `LANGFUSE_PUBLIC_KEY` is empty the app runs normally with no tracing. No exception is raised.

### Start LangFuse (self-hosted, no account needed)

LangFuse is included in `docker-compose.yml`. Run the full stack:

```bash
docker compose up -d
```

Then visit **http://localhost:3000** and complete the one-time setup:

1. Click **Sign Up** and create a local account (stored in the Postgres container — no external service involved).
2. Create an **organisation** and a **project** (any name, e.g. `aves-argentinas`).
3. Go to **Settings → API Keys** and click **Create new API key**.
4. Copy the **Public Key** and **Secret Key**.

### Configure the backend

Paste the keys into `backend/.env`:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

Restart the backend:

```bash
make dev
# or: docker compose up -d --build backend
```

### What gets traced

| Trace name | Triggered by | Data captured |
|---|---|---|
| `chat` | `POST /api/v1/chat` | Session ID, full message history sent to the model, model reply, wall-clock latency |
| `rag-query` | `POST /api/v1/rag-query` | System prompt + all retrieved chunks injected as context, model answer, wall-clock latency |

### Exploring the dashboard

Open **http://localhost:3000 → Tracing → Traces** after making a few requests.

**Traces list** — one row per API call, named `chat` or `rag-query`. Click any row to drill in.

**Inside a trace** — you'll see the `llm-call` generation with:
- **Input**: the exact messages array sent to the model (system prompt, conversation history or RAG context, user query)
- **Output**: the raw model reply
- **Latency**: wall-clock time for the LLM call

**Latency breakdown (from real runs):**
- `chat` — p50 ~1–2 s, p90 ~7 s (grows with conversation length as history accumulates)
- `rag-query` — p50 ~30–35 s (embedding + Qdrant retrieval + LLM on full retrieved context)

**Session grouping** — `chat` traces carry a `session_id`. In the Traces view, filter by session to see all turns of a conversation in order.

**Model costs / Token counts** — show `$0.00` and `0 tokens` because Ollama runs locally and does not report token counts through LangChain. This is expected for any self-hosted local model setup — cost tracking applies when using cloud providers (OpenAI, Anthropic, etc.).

**Scores** — not wired up; this is a LangFuse feature for collecting human feedback (thumbs up/down) on individual traces, out of scope for this challenge.

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
