.PHONY: build up up-build down logs dev install lint test ingest frontend-install frontend-dev hash-password install-hooks

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
install: install-hooks
	cd backend && uv sync

install-hooks:
	ln -sf ../../scripts/pre-push.sh .git/hooks/pre-push
	@echo "pre-push hook installed"

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

## ── Auth ──────────────────────────────────────────────────────────────────────
hash-password:
	@read -p "Password: " pw && cd backend && uv run python -c \
	"import bcrypt, sys; print(bcrypt.hashpw(sys.argv[1].encode(), bcrypt.gensalt()).decode())" "$$pw"

## ── RAG ───────────────────────────────────────────────────────────────────────
ingest:
	cd backend && uv run python scripts/ingest.py
