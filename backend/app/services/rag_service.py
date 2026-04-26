import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.adapters.llm_adapter import active_model, get_llm
from app.adapters.qdrant_adapter import qdrant_adapter
from app.core.config import settings
from app.core.logging import get_logger
from app.core.observability import start_trace
from app.services.embedding_service import embedding_service, sparse_embedding_service

logger = get_logger(__name__)

_RAG_SYSTEM_PROMPT = (
    "You are a knowledgeable ornithology assistant specialising in Argentine birds.\n"
    "Answer the user's question based strictly on the context provided below.\n"
    "Cite the source document and page number when relevant.\n"
    "If the context does not contain enough information to answer the question, say:\n"
    '"I don\'t have enough information in the provided documents to answer this question."\n'
    "Do not use prior knowledge beyond what is in the context."
)


def _build_context(hits: list) -> str:
    parts = []
    for hit in hits:
        p = hit.payload or {}
        header = f"[Source: {p.get('source', 'unknown')}, Page {p.get('page', '?')}]"
        parts.append(f"{header}\n{p.get('text', '')}")
    return "\n\n---\n\n".join(parts)


class RAGService:
    async def query(
        self,
        query: str,
        top_k: int | None = None,
        source_filter: str | None = None,
    ) -> dict:
        resolved_top_k = top_k or settings.rag_top_k

        # 1 — embed query (dense + sparse for hybrid search)
        query_vector = embedding_service.embed_one(query)
        sparse_vector = sparse_embedding_service.embed_one(query)

        # 2 — retrieve from Qdrant using RRF hybrid search
        hits = qdrant_adapter.hybrid_search(
            dense_vector=query_vector,
            sparse_vector=sparse_vector,
            top_k=resolved_top_k,
            source_filter=source_filter,
        )

        if not hits:
            return {
                "answer": "No relevant documents found for your query.",
                "sources": [],
                "meta": {"latency_ms": 0, "hits": 0},
            }

        # 3 — build grounded prompt
        context = _build_context(hits)
        messages = [
            SystemMessage(content=_RAG_SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]

        # 4 — call LLM (no session — RAG queries are stateless)
        client = get_llm()
        tracer = start_trace("rag-query", model=active_model(), input_messages=messages)

        start = time.perf_counter()
        response = await client.ainvoke(messages)
        latency_ms = int((time.perf_counter() - start) * 1000)

        if tracer:
            tracer.finish(output=response.content, latency_ms=latency_ms)

        logger.info("RAGService.query hits=%d latency_ms=%d", len(hits), latency_ms)

        # 5 — build sources with image URLs
        sources = []
        for hit in hits:
            p = hit.payload or {}
            image_filenames: list[str] = p.get("image_filenames", [])
            sources.append(
                {
                    "chunk_id": p.get("chunk_id", ""),
                    "source": p.get("source", ""),
                    "page": p.get("page"),
                    "score": round(hit.score, 4),
                    "text_snippet": p.get("text", "")[:200],
                    "image_urls": [f"/images/{fn}" for fn in image_filenames],
                }
            )

        return {
            "answer": response.content,
            "sources": sources,
            "meta": {"latency_ms": latency_ms, "hits": len(hits)},
        }


rag_service = RAGService()
