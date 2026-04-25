from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_chat_returns_reply():
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
            )

    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == mock_result["reply"]
    assert data["session_id"] == "test-session"
    assert "model" in data
    assert "meta" in data


@pytest.mark.asyncio
async def test_chat_empty_message_is_rejected():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/chat", json={"message": ""})

    assert response.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_chat_generates_session_id_when_omitted():
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
            response = await client.post("/api/v1/chat", json={"message": "Hi"})

    assert response.status_code == 200
    assert response.json()["session_id"] == "auto-generated-uuid"
