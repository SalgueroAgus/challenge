from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    ScoredPoint,
    SparseVector,
)

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class QdrantAdapter:
    def __init__(self) -> None:
        self._client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> list[ScoredPoint]:
        collection = settings.qdrant_collection_name

        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source_filter),
                    )
                ]
            )

        response = self._client.query_points(
            collection_name=collection,
            prefetch=[
                Prefetch(query=dense_vector, using="dense", limit=top_k * 5),
                Prefetch(query=sparse_vector, using="sparse", limit=top_k * 5),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        results = response.points
        logger.info(
            "QdrantAdapter.hybrid_search collection=%s top_k=%d hits=%d",
            collection,
            top_k,
            len(results),
        )
        return results

    def upsert(self, collection_name: str, points: list) -> None:
        self._client.upsert(collection_name=collection_name, points=points)


qdrant_adapter = QdrantAdapter()
