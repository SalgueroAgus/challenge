from fastapi import APIRouter

from app.api.v1.routes import agent, auth, chat, health, rag

router = APIRouter(prefix="/api/v1")

router.include_router(health.router, tags=["health"])
router.include_router(auth.router, tags=["auth"])
router.include_router(chat.router, tags=["chat"])
router.include_router(rag.router, tags=["rag"])
router.include_router(agent.router, tags=["agent"])
