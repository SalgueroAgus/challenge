#!/usr/bin/env bash
# Runs the same checks as CI before pushing. Fail fast — stop on first error.
set -e

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT/backend"

echo "→ ruff check"
uv run ruff check .

echo "→ ruff format"
uv run ruff format --check .

echo "→ pytest"
uv run pytest -v

echo "✓ All checks passed"
