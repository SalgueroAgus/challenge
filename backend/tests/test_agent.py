from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


def _mock_agent_result(route: str = "rag", retries: int = 0) -> dict:
    sources = (
        [
            {
                "chunk_id": "abc-123",
                "source": "birds_part1.pdf",
                "page": 4,
                "score": 0.92,
                "text_snippet": "The Rufous Hornero builds oven-shaped nests...",
                "image_urls": ["/images/birds_part1_p4_1.png"],
            }
        ]
        if route == "rag"
        else []
    )
    return {
        "answer": "The Rufous Hornero is Argentina's national bird.",
        "sources": sources,
        "route": route,
        "meta": {"latency_ms": 500, "retries": retries},
    }


@pytest.mark.asyncio
async def test_agent_rag_route_returns_answer_and_sources(auth_headers):
    with patch(
        "app.api.v1.routes.agent.agent_service.run",
        new_callable=AsyncMock,
        return_value=_mock_agent_result(route="rag"),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/agent",
                json={"query": "What is Argentina's national bird?"},
                headers=auth_headers,
            )

    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "rag"
    assert "answer" in data
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) == 1
    assert data["sources"][0]["image_urls"] == ["/images/birds_part1_p4_1.png"]
    assert data["meta"]["retries"] == 0


@pytest.mark.asyncio
async def test_agent_direct_route_returns_empty_sources(auth_headers):
    with patch(
        "app.api.v1.routes.agent.agent_service.run",
        new_callable=AsyncMock,
        return_value=_mock_agent_result(route="direct"),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/agent",
                json={"query": "Hello, how are you?"},
                headers=auth_headers,
            )

    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "direct"
    assert data["sources"] == []


@pytest.mark.asyncio
async def test_agent_retries_reflected_in_meta(auth_headers):
    with patch(
        "app.api.v1.routes.agent.agent_service.run",
        new_callable=AsyncMock,
        return_value=_mock_agent_result(route="direct", retries=1),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/agent",
                json={"query": "Some obscure bird question"},
                headers=auth_headers,
            )

    assert response.status_code == 200
    assert response.json()["meta"]["retries"] == 1


@pytest.mark.asyncio
async def test_agent_empty_query_rejected(auth_headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/agent", json={"query": ""}, headers=auth_headers)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_agent_requires_auth():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/agent", json={"query": "test"})

    assert response.status_code == 401
