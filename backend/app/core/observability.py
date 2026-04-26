from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_lf_client = None


def _get_client():
    global _lf_client
    if _lf_client is not None:
        return _lf_client
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None
    try:
        from langfuse import Langfuse

        _lf_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("LangFuse tracing enabled (host=%s)", settings.langfuse_host)
    except Exception as exc:
        logger.warning("LangFuse init failed, tracing disabled: %s", exc)
    return _lf_client


def _messages_to_dicts(messages: list) -> list[dict]:
    """Convert LangChain message objects to LangFuse-friendly dicts."""
    role_map = {
        "SystemMessage": "system",
        "HumanMessage": "user",
        "AIMessage": "assistant",
    }
    result = []
    for m in messages:
        role = role_map.get(type(m).__name__, "user")
        result.append({"role": role, "content": getattr(m, "content", str(m))})
    return result


class LLMTrace:
    """Holds an open LangFuse trace + generation; call finish() after the LLM responds."""

    def __init__(self, trace, generation, client) -> None:
        self._trace = trace
        self._gen = generation
        self._client = client

    def finish(self, output: str, latency_ms: int | None = None) -> None:
        try:
            self._gen.end(output=output)
            # Mirror the final answer at the trace level so it's visible without drilling in
            self._trace.update(output=output)
            self._client.flush()
        except Exception as exc:
            logger.warning("LangFuse flush failed: %s", exc)


def start_trace(
    name: str,
    model: str,
    input_messages: list,
    session_id: str | None = None,
) -> "LLMTrace | None":
    """Open a LangFuse trace + generation. Returns None when tracing is off."""
    lf = _get_client()
    if not lf:
        return None
    try:
        # Use the last human message as the trace-level input so it's visible at a glance
        last_human = next(
            (m["content"] for m in reversed(_messages_to_dicts(input_messages)) if m["role"] == "user"),
            None,
        )
        trace = lf.trace(name=name, session_id=session_id, input=last_human)
        generation = trace.generation(
            name="llm-call",
            model=model,
            input=_messages_to_dicts(input_messages),
        )
        return LLMTrace(trace, generation, lf)
    except Exception as exc:
        logger.warning("LangFuse trace start failed: %s", exc)
        return None
