from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


def _mock_rag_result(answer: str = "The Rufous Hornero is Argentina's national bird.") -> dict:
    return {
        "answer": answer,
        "sources": [
            {
                "chunk_id": "abc-123",
                "source": "birds_part1.pdf",
                "page": 4,
                "score": 0.92,
                "text_snippet": "The Rufous Hornero builds oven-shaped nests...",
                "image_urls": ["/images/birds_part1_p4_1.png"],
            }
        ],
        "meta": {"latency_ms": 800, "hits": 1},
    }


@pytest.mark.asyncio
async def test_rag_query_returns_answer_and_sources():
    with patch(
        "app.api.v1.routes.rag.rag_service.query",
        new_callable=AsyncMock,
        return_value=_mock_rag_result(),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/rag-query",
                json={"query": "What is Argentina's national bird?"},
            )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) == 1
    assert data["sources"][0]["image_urls"] == ["/images/birds_part1_p4_1.png"]
    assert "meta" in data


@pytest.mark.asyncio
async def test_rag_query_empty_string_is_rejected():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/rag-query", json={"query": ""})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_rag_query_top_k_out_of_range_rejected():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/rag-query", json={"query": "birds", "top_k": 99})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_rag_query_with_source_filter():
    with patch(
        "app.api.v1.routes.rag.rag_service.query",
        new_callable=AsyncMock,
        return_value=_mock_rag_result(),
    ) as mock_query:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/api/v1/rag-query",
                json={"query": "hornero", "source_filter": "birds_part1.pdf"},
            )

    mock_query.assert_called_once_with(
        query="hornero",
        top_k=None,
        source_filter="birds_part1.pdf",
    )
