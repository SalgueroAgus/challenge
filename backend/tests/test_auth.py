from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from jose import jwt

from app.core.config import settings
from app.main import app

# ── helpers ───────────────────────────────────────────────────────────────────

VALID_CREDENTIALS = {"username": settings.auth_username, "password": "changeme"}


async def _get_token(client: AsyncClient) -> str:
    resp = await client.post("/api/v1/auth/token", data=VALID_CREDENTIALS)
    return resp.json()["access_token"]


def _expired_token() -> str:
    payload = {
        "sub": settings.auth_username,
        "exp": datetime.now(UTC) - timedelta(minutes=1),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


# ── auth endpoint ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_login_valid_credentials():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/auth/token", data=VALID_CREDENTIALS)
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_wrong_password():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/auth/token", data={"username": "admin", "password": "wrong"}
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_login_wrong_username():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/auth/token", data={"username": "nobody", "password": "changeme"}
        )
    assert resp.status_code == 401


# ── protected routes ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_without_token_is_rejected():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/chat", json={"message": "hello"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_rag_without_token_is_rejected():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/rag-query", json={"query": "birds"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_chat_with_valid_token():
    mock_result = {
        "reply": "Hello!",
        "session_id": "abc",
        "model": "qwen3.5:4b",
        "meta": {"latency_ms": 100},
    }
    with patch(
        "app.api.v1.routes.chat.llm_service.chat",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            token = await _get_token(client)
            resp = await client.post(
                "/api/v1/chat",
                json={"message": "hello"},
                headers={"Authorization": f"Bearer {token}"},
            )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_chat_with_expired_token_is_rejected():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/chat",
            json={"message": "hello"},
            headers={"Authorization": f"Bearer {_expired_token()}"},
        )
    assert resp.status_code == 401
