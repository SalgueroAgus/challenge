from langchain_core.language_models import BaseChatModel

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def get_llm(model: str | None = None) -> BaseChatModel:
    """Return a LangChain chat model for the configured provider.

    Switch providers by setting LLM_PROVIDER in your .env:
      LLM_PROVIDER=ollama   (default, local)
      LLM_PROVIDER=groq     (cloud, free tier)
    """
    provider = settings.llm_provider.lower()

    if provider == "groq":
        from langchain_groq import ChatGroq

        resolved = model or settings.groq_model
        logger.info("LLMAdapter: provider=groq model=%s", resolved)
        return ChatGroq(api_key=settings.groq_api_key, model=resolved)

    # Default: Ollama (local)
    from langchain_ollama import ChatOllama

    resolved = model or settings.ollama_model
    logger.info("LLMAdapter: provider=ollama model=%s", resolved)
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=resolved,
        num_ctx=2048,
        reasoning=False,
    )


def active_model(override: str | None = None) -> str:
    """Return the model name that would be used without constructing the client."""
    if override:
        return override
    provider = settings.llm_provider.lower()
    return settings.groq_model if provider == "groq" else settings.ollama_model
