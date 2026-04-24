from fastapi import APIRouter

from app.core.config import settings

router = APIRouter()


@router.get("/healthcheck")
async def healthcheck() -> dict:
    return {
        "status": "ok",
        "version": settings.app_version,
        "environment": settings.app_env,
        "model": settings.ollama_model,
    }
