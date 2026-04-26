import time
import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.adapters.llm_adapter import active_model, get_llm
from app.core.logging import get_logger
from app.core.observability import start_trace

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a knowledgeable and concise assistant.
Answer questions clearly and accurately based on your training knowledge.
When you are unsure about something, say so rather than guessing.
Keep responses focused and to the point — avoid unnecessary padding."""

# In-memory session store: {session_id: [BaseMessage, ...]}
# Each session starts with the system prompt and accumulates turns.
_sessions: dict[str, list[BaseMessage]] = {}

# Prevent unbounded memory growth — trim oldest human/ai pairs beyond this limit.
_MAX_TURNS = 10


def _get_or_create_session(session_id: str) -> list[BaseMessage]:
    if session_id not in _sessions:
        _sessions[session_id] = [SystemMessage(content=SYSTEM_PROMPT)]
    return _sessions[session_id]


def _trim_session(messages: list[BaseMessage]) -> None:
    """Keep the SystemMessage + last _MAX_TURNS * 2 messages."""
    system = messages[:1]
    history = messages[1:]
    if len(history) > _MAX_TURNS * 2:
        messages.clear()
        messages.extend(system + history[-(_MAX_TURNS * 2) :])


class LLMService:
    async def chat(
        self,
        message: str,
        session_id: str | None = None,
        model_name: str | None = None,
    ) -> dict:
        sid = session_id or str(uuid.uuid4())
        history = _get_or_create_session(sid)

        history.append(HumanMessage(content=message))

        client = get_llm(model_name)
        resolved_model = active_model(model_name)

        logger.info("LLMService.chat session=%s model=%s", sid, resolved_model)

        tracer = start_trace("chat", model=resolved_model, input_messages=history, session_id=sid)

        start = time.perf_counter()
        response = await client.ainvoke(history)
        latency_ms = int((time.perf_counter() - start) * 1000)

        if tracer:
            tracer.finish(output=response.content, latency_ms=latency_ms)

        history.append(AIMessage(content=response.content))
        _trim_session(history)

        return {
            "reply": response.content,
            "session_id": sid,
            "model": resolved_model,
            "meta": {"latency_ms": latency_ms},
        }


# Module-level singleton
llm_service = LLMService()
