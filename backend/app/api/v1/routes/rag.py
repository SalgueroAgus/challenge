from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.core.security import get_current_user
from app.services.rag_service import rag_service

logger = get_logger(__name__)
router = APIRouter()


class RAGRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    top_k: int | None = Field(None, ge=1, le=20, description="Number of chunks to retrieve")
    source_filter: str | None = Field(None, description="Filter by PDF filename")


class RAGSource(BaseModel):
    chunk_id: str
    source: str
    page: int | None
    score: float
    text_snippet: str
    image_urls: list[str]
    common_name: str = ""
    scientific_name: str = ""


class RAGMeta(BaseModel):
    latency_ms: int
    hits: int


class RAGResponse(BaseModel):
    answer: str
    sources: list[RAGSource]
    meta: RAGMeta


@router.post("/rag-query", response_model=RAGResponse)
async def rag_query(request: RAGRequest, _: str = Depends(get_current_user)) -> RAGResponse:
    try:
        result = await rag_service.query(
            query=request.query,
            top_k=request.top_k,
            source_filter=request.source_filter,
        )
    except Exception as exc:
        logger.exception("RAG query failed: %s", exc)
        raise HTTPException(status_code=502, detail="RAG service unavailable") from exc

    return RAGResponse(
        answer=result["answer"],
        sources=[RAGSource(**s) for s in result["sources"]],
        meta=RAGMeta(**result["meta"]),
    )
