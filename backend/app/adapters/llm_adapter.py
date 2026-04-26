from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=8)
def _build_llm(provider: str, model: str) -> BaseChatModel:
    """Build and cache a LangChain chat model instance keyed by (provider, model).

    Cached because LangChain models are stateless HTTP clients — safe to reuse
    across requests and avoids reconstructing the client object on every call.
    """
    if provider == "groq":
        from langchain_groq import ChatGroq

        logger.info("LLMAdapter: provider=groq model=%s (cached)", model)
        return ChatGroq(api_key=settings.groq_api_key, model=model)

    from langchain_ollama import ChatOllama

    logger.info("LLMAdapter: provider=ollama model=%s (cached)", model)
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=model,
        num_ctx=2048,
        reasoning=False,
    )


def get_llm(model: str | None = None) -> BaseChatModel:
    """Return the cached LangChain chat model for the configured provider.

    Switch providers by setting LLM_PROVIDER in your .env:
      LLM_PROVIDER=ollama   (default, local)
      LLM_PROVIDER=groq     (cloud, free tier)
    """
    provider = settings.llm_provider.lower()
    resolved = model or (settings.groq_model if provider == "groq" else settings.ollama_model)
    return _build_llm(provider, resolved)


def active_model(override: str | None = None) -> str:
    """Return the model name that would be used without constructing the client."""
    if override:
        return override
    provider = settings.llm_provider.lower()
    return settings.groq_model if provider == "groq" else settings.ollama_model
