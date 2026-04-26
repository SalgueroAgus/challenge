import time

from langchain_core.messages import HumanMessage, SystemMessage
from langfuse.decorators import langfuse_context, observe
from qdrant_client.models import ScoredPoint

from app.adapters.llm_adapter import active_model, get_llm
from app.adapters.qdrant_adapter import qdrant_adapter
from app.core.config import settings
from app.core.logging import get_logger
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


@observe(as_type="generation")
async def _llm_call(system: str, user: str) -> str:
    langfuse_context.update_current_observation(
        model=active_model(),
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    response = await get_llm().ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
    text = response.content
    langfuse_context.update_current_observation(output=text)
    return text


def _build_context(hits: list[ScoredPoint]) -> str:
    parts = []
    for hit in hits:
        p = hit.payload or {}
        header = f"[Source: {p.get('source', 'unknown')}, Page {p.get('page', '?')}]"
        parts.append(f"{header}\n{p.get('text', '')}")
    return "\n\n---\n\n".join(parts)


class RAGService:
    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source_filter: str | None = None,
    ) -> list[ScoredPoint]:
        resolved_top_k = top_k or settings.rag_top_k
        query_vector = embedding_service.embed_one(query)
        sparse_vector = sparse_embedding_service.embed_one(query)
        return qdrant_adapter.hybrid_search(
            dense_vector=query_vector,
            sparse_vector=sparse_vector,
            top_k=resolved_top_k,
            source_filter=source_filter,
        )

    def build_context(self, hits: list[ScoredPoint]) -> str:
        return _build_context(hits)

    def hits_to_sources(self, hits: list[ScoredPoint]) -> list[dict]:
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
        return sources

    @observe(name="rag-query")
    async def query(
        self,
        query: str,
        top_k: int | None = None,
        source_filter: str | None = None,
    ) -> dict:
        start = time.perf_counter()

        hits = await self.retrieve(query, top_k, source_filter)

        if not hits:
            return {
                "answer": "No relevant documents found for your query.",
                "sources": [],
                "meta": {"latency_ms": 0, "hits": 0},
            }

        context = self.build_context(hits)
        answer = await _llm_call(_RAG_SYSTEM_PROMPT, f"Context:\n{context}\n\nQuestion: {query}")
        latency_ms = int((time.perf_counter() - start) * 1000)

        logger.info("RAGService.query hits=%d latency_ms=%d", len(hits), latency_ms)

        return {
            "answer": answer,
            "sources": self.hits_to_sources(hits),
            "meta": {"latency_ms": latency_ms, "hits": len(hits)},
        }


rag_service = RAGService()
