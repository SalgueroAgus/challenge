.PHONY: build up up-build down logs dev install lint test ingest frontend-install frontend-dev

## ── Docker ────────────────────────────────────────────────────────────────────
build:
	docker compose build

up:
	docker compose up -d

up-build:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f

## ── Backend (local dev) ───────────────────────────────────────────────────────
install:
	cd backend && uv sync

dev:
	cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

lint:
	cd backend && uv run ruff check . && uv run ruff format --check .

test:
	cd backend && uv run pytest -v

## ── Frontend ──────────────────────────────────────────────────────────────────
frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

## ── RAG ───────────────────────────────────────────────────────────────────────
ingest:
	cd backend && uv run python scripts/ingest.py
