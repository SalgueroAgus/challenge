from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_chat_returns_reply(auth_headers):
    mock_result = {
        "reply": "RAG stands for Retrieval-Augmented Generation.",
        "session_id": "test-session",
        "model": "qwen3.5:4b",
        "meta": {"latency_ms": 42},
    }

    with patch(
        "app.api.v1.routes.chat.llm_service.chat",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat",
                json={"message": "What is RAG?", "session_id": "test-session"},
                headers=auth_headers,
            )

    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == mock_result["reply"]
    assert data["session_id"] == "test-session"
    assert "model" in data
    assert "meta" in data


@pytest.mark.asyncio
async def test_chat_empty_message_is_rejected(auth_headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/chat", json={"message": ""}, headers=auth_headers)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_generates_session_id_when_omitted(auth_headers):
    mock_result = {
        "reply": "Hello!",
        "session_id": "auto-generated-uuid",
        "model": "qwen3.5:4b",
        "meta": {"latency_ms": 10},
    }

    with patch(
        "app.api.v1.routes.chat.llm_service.chat",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat", json={"message": "Hi"}, headers=auth_headers
            )

    assert response.status_code == 200
    assert response.json()["session_id"] == "auto-generated-uuid"
