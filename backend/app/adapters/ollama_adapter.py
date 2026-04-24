from langchain_ollama import ChatOllama

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class OllamaAdapter:
    """Thin wrapper around ChatOllama. Supports per-request model override."""

    def __init__(self) -> None:
        self._base_url = settings.ollama_base_url
        self._default_model = settings.ollama_model

    def get_client(self, model: str | None = None) -> ChatOllama:
        resolved = model or self._default_model
        logger.info("OllamaAdapter: using model=%s", resolved)
        return ChatOllama(
            base_url=self._base_url,
            model=resolved,
        )


# Module-level singleton — created once, reused across requests
ollama_adapter = OllamaAdapter()
