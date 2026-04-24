from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.services.llm_service import llm_service

logger = get_logger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User prompt")
    session_id: str | None = Field(None, description="Groups messages into a conversation")
    model_name: str | None = Field(None, description="Override the default Ollama model")


class ChatMeta(BaseModel):
    latency_ms: int


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    model: str
    meta: ChatMeta


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = await llm_service.chat(
            message=request.message,
            session_id=request.session_id,
            model_name=request.model_name,
        )
    except Exception as exc:
        logger.exception("LLM call failed: %s", exc)
        raise HTTPException(status_code=502, detail="LLM service unavailable") from exc

    return ChatResponse(
        reply=result["reply"],
        session_id=result["session_id"],
        model=result["model"],
        meta=ChatMeta(latency_ms=result["meta"]["latency_ms"]),
    )
