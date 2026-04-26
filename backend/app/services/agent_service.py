import time
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langfuse.decorators import langfuse_context, observe
from langgraph.graph import END, START, StateGraph

from app.adapters.llm_adapter import active_model, get_llm
from app.core.logging import get_logger
from app.services.rag_service import rag_service

logger = get_logger(__name__)

_CLASSIFY_SYSTEM = (
    "You are a query router for an Argentine birds knowledge base.\n"
    "Respond with EXACTLY TWO lines:\n"
    "Line 1: 'rag' or 'direct'\n"
    "  rag = question needs facts from scientific bird documents (species, taxonomy, distribution, conservation)\n"
    "  direct = greeting, general chat, or topic unrelated to Argentine birds\n"
    "Line 2: the key search terms — strip ALL conversational filler\n"
    "  Remove phrases like: 'que me podes decir', 'contame sobre', 'sabes algo del',\n"
    "  'que sabes de', 'hablame del', 'tenes algo de', 'dame info sobre', etc.\n"
    "  Keep only: species name, scientific name, location, or topic\n\n"
    "Examples:\n"
    "Input: Que me podes decir del tero?\n"
    "rag\ntero\n\n"
    "Input: Contame sobre la garza bruja coronada\n"
    "rag\ngarza bruja coronada\n\n"
    "Input: What can you tell me about Vanellus chilensis?\n"
    "rag\nVanellus chilensis\n\n"
    "Input: Hola como estas?\n"
    "direct\ngreeting"
)

_GRADE_SYSTEM = (
    "Does the retrieved context contain ENOUGH information to answer the question?\n"
    "Look at ALL chunks. If even ONE chunk has multiple sentences of descriptive content "
    "(distribution, field records, taxonomy, conservation notes), reply 'good'.\n"
    "Reply 'poor' ONLY if EVERY chunk is a brief one-liner reference entry "
    "(like '212. Tero (Vanellus chilensis) / Southern Lapwing') "
    "or if ALL chunks are about completely different topics.\n"
    "Reply with exactly one word: good or poor"
)

_REWRITE_SYSTEM = (
    "You improve search queries for an Argentine birds scientific checklist database.\n"
    "Rewrite the query using more specific scientific or Spanish ornithological terminology "
    "that would better match entries in the document corpus.\n"
    "Return only the rewritten query, nothing else."
)

_DIRECT_SYSTEM = (
    "You are a knowledgeable ornithology assistant specialising in Argentine birds.\n"
    "Answer based on your training knowledge. Be concise and accurate.\n"
    "If the question is unrelated to birds, answer helpfully from general knowledge."
)

_RAG_SYSTEM = (
    "You are a knowledgeable ornithology assistant specialising in Argentine birds.\n"
    "Answer the user's question using only the context provided below.\n"
    "Cite the source document and page number when relevant.\n"
    "If the context contains partial information, share what is available and note what is missing.\n"
    "Only say 'I don't have enough information in the provided documents' if the context "
    "contains absolutely nothing related to the question.\n"
    "Do not use prior knowledge beyond what is in the context."
)


class AgentState(TypedDict):
    query: str  # original user query — never mutated
    current_query: str  # may be rewritten once by rewrite_node
    route: str  # "rag" | "direct" — set by generate nodes (reflects what ran)
    hits: list  # Qdrant ScoredPoint results
    grade: str  # "good" | "poor"
    retries: int  # incremented by rewrite_node; caps the cycle at 1 retry
    answer: str
    sources: list


@observe(as_type="generation")
async def _llm_call(system: str, user: str) -> str:
    """Single LLM call tracked as a Langfuse generation nested under the calling node's span."""
    langfuse_context.update_current_observation(
        model=active_model(),
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    response = await get_llm().ainvoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
    )
    text = response.content.strip()
    langfuse_context.update_current_observation(output=text)
    return text


@observe()
async def classify_node(state: AgentState) -> dict:
    raw = await _llm_call(_CLASSIFY_SYSTEM, state["query"])
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    route = "rag" if lines and "rag" in lines[0].lower() else "direct"
    # Use the extracted search terms when available; fall back to original query
    if route == "rag" and len(lines) > 1 and lines[1].lower() not in ("rag", "direct", ""):
        current_query = lines[1]
    else:
        current_query = state["query"]
    langfuse_context.update_current_observation(
        input=state["query"],
        output=route,
        metadata={"current_query": current_query},
    )
    logger.info("agent.classify route=%s query=%r -> %r", route, state["query"], current_query)
    return {"current_query": current_query, "route": route}


@observe()
async def retrieve_node(state: AgentState) -> dict:
    hits = await rag_service.retrieve(state["current_query"])
    langfuse_context.update_current_observation(
        input=state["current_query"],
        output=[
            {"source": h.payload.get("source"), "page": h.payload.get("page"), "score": round(h.score, 4)}
            for h in hits
        ],
        metadata={"hits": len(hits)},
    )
    logger.info("agent.retrieve hits=%d query=%r", len(hits), state["current_query"])
    return {"hits": hits}


@observe()
async def grade_node(state: AgentState) -> dict:
    context = rag_service.build_context(state["hits"])
    user = f"Question: {state['current_query']}\n\nContext:\n{context}"
    raw = await _llm_call(_GRADE_SYSTEM, user)
    grade = "good" if "good" in raw.lower() else "poor"
    langfuse_context.update_current_observation(
        input=state["current_query"],
        output=grade,
    )
    logger.info("agent.grade grade=%s retries=%d", grade, state["retries"])
    return {"grade": grade}


@observe()
async def rewrite_node(state: AgentState) -> dict:
    rewritten = await _llm_call(_REWRITE_SYSTEM, state["current_query"])
    langfuse_context.update_current_observation(
        input=state["current_query"],
        output=rewritten,
    )
    logger.info("agent.rewrite %r -> %r", state["current_query"], rewritten)
    return {"current_query": rewritten, "retries": state["retries"] + 1}


@observe()
async def generate_rag_node(state: AgentState) -> dict:
    context = rag_service.build_context(state["hits"])
    user_content = f"Context:\n{context}\n\nQuestion: {state['query']}"
    answer = await _llm_call(_RAG_SYSTEM, user_content)
    sources = rag_service.hits_to_sources(state["hits"])
    langfuse_context.update_current_observation(
        input=state["query"],
        output=answer,
        metadata={"sources": len(sources)},
    )
    logger.info("agent.generate_rag sources=%d", len(sources))
    return {"answer": answer, "sources": sources, "route": "rag"}


@observe()
async def generate_direct_node(state: AgentState) -> dict:
    answer = await _llm_call(_DIRECT_SYSTEM, state["query"])
    langfuse_context.update_current_observation(
        input=state["query"],
        output=answer,
    )
    logger.info("agent.generate_direct")
    return {"answer": answer, "sources": [], "route": "direct"}


def _route_after_classify(state: AgentState) -> str:
    return state["route"]


def _route_after_grade(state: AgentState) -> str:
    if state["grade"] == "good":
        return "generate_rag"
    if state["retries"] >= 1:
        return "generate_direct"
    return "rewrite"


def _build_graph():
    g = StateGraph(AgentState)

    g.add_node("classify", classify_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("generate_rag", generate_rag_node)
    g.add_node("generate_direct", generate_direct_node)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify",
        _route_after_classify,
        {
            "rag": "retrieve",
            "direct": "generate_direct",
        },
    )
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        _route_after_grade,
        {
            "generate_rag": "generate_rag",
            "generate_direct": "generate_direct",
            "rewrite": "rewrite",
        },
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate_rag", END)
    g.add_edge("generate_direct", END)

    return g.compile()


_graph = _build_graph()


class AgentService:
    @observe(name="agent-run")
    async def run(self, query: str) -> dict:
        start = time.perf_counter()

        langfuse_context.update_current_observation(
            input=query,
            metadata={"service": "agent"},
        )

        initial: AgentState = {
            "query": query,
            "current_query": query,
            "route": "",
            "hits": [],
            "grade": "",
            "retries": 0,
            "answer": "",
            "sources": [],
        }

        final = await _graph.ainvoke(initial)
        latency_ms = int((time.perf_counter() - start) * 1000)

        langfuse_context.update_current_observation(
            output=final["answer"],
            metadata={
                "route": final["route"],
                "retries": final["retries"],
                "latency_ms": latency_ms,
            },
        )

        logger.info(
            "AgentService.run route=%s retries=%d latency_ms=%d",
            final["route"],
            final["retries"],
            latency_ms,
        )

        return {
            "answer": final["answer"],
            "sources": final["sources"],
            "route": final["route"],
            "meta": {"latency_ms": latency_ms, "retries": final["retries"]},
        }


agent_service = AgentService()
