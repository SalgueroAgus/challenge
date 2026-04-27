from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.core.security import get_current_user
from app.services.agent_service import agent_service

logger = get_logger(__name__)
router = APIRouter()


class AgentRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")


class AgentSource(BaseModel):
    chunk_id: str
    source: str
    page: int | None
    score: float
    text_snippet: str
    image_urls: list[str]
    common_name: str = ""
    scientific_name: str = ""


class AgentMeta(BaseModel):
    latency_ms: int
    retries: int


class AgentResponse(BaseModel):
    answer: str
    sources: list[AgentSource]
    route: str
    meta: AgentMeta


@router.post("/agent", response_model=AgentResponse)
async def agent_query(request: AgentRequest, _: str = Depends(get_current_user)) -> AgentResponse:
    try:
        result = await agent_service.run(query=request.query)
    except Exception as exc:
        logger.exception("Agent query failed: %s", exc)
        raise HTTPException(status_code=502, detail="Agent service unavailable") from exc

    return AgentResponse(
        answer=result["answer"],
        sources=[AgentSource(**s) for s in result["sources"]],
        route=result["route"],
        meta=AgentMeta(**result["meta"]),
    )
