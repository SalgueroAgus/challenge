from fastapi import APIRouter

from app.adapters.llm_adapter import active_model
from app.core.config import settings

router = APIRouter()


@router.get("/healthcheck")
async def healthcheck() -> dict:
    return {
        "status": "ok",
        "version": settings.app_version,
        "environment": settings.app_env,
        "llm_provider": settings.llm_provider,
        "model": active_model(),
    }
