import time
import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langfuse.decorators import langfuse_context, observe

from app.adapters.llm_adapter import active_model, get_llm
from app.core.logging import get_logger

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


def _msg_to_dict(m: BaseMessage) -> dict:
    role_map = {"SystemMessage": "system", "HumanMessage": "user", "AIMessage": "assistant"}
    return {"role": role_map.get(type(m).__name__, "user"), "content": m.content}


@observe(as_type="generation")
async def _invoke_history(messages: list[BaseMessage], model: str | None = None) -> str:
    langfuse_context.update_current_observation(
        model=active_model(model),
        input=[_msg_to_dict(m) for m in messages],
    )
    response = await get_llm(model).ainvoke(messages)
    text = response.content
    langfuse_context.update_current_observation(output=text)
    return text


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
    @observe(name="chat")
    async def chat(
        self,
        message: str,
        session_id: str | None = None,
        model_name: str | None = None,
    ) -> dict:
        sid = session_id or str(uuid.uuid4())
        resolved_model = active_model(model_name)
        history = _get_or_create_session(sid)
        history.append(HumanMessage(content=message))

        langfuse_context.update_current_trace(session_id=sid)
        langfuse_context.update_current_observation(
            input=message, metadata={"model": resolved_model}
        )

        logger.info("LLMService.chat session=%s model=%s", sid, resolved_model)

        start = time.perf_counter()
        reply = await _invoke_history(history, model_name)
        latency_ms = int((time.perf_counter() - start) * 1000)

        history.append(AIMessage(content=reply))
        _trim_session(history)

        langfuse_context.update_current_observation(output=reply)

        return {
            "reply": reply,
            "session_id": sid,
            "model": resolved_model,
            "meta": {"latency_ms": latency_ms},
        }


# Module-level singleton
llm_service = LLMService()
