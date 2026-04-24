from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, ScoredPoint

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class QdrantAdapter:
    def __init__(self) -> None:
        self._client = QdrantClient(url=settings.qdrant_url)

    def search(
        self,
        query_vector: list[float],
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

        # query_points() is the unified search API in qdrant-client >= 1.7
        response = self._client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        results = response.points
        logger.info(
            "QdrantAdapter.search collection=%s top_k=%d hits=%d",
            collection,
            top_k,
            len(results),
        )
        return results


qdrant_adapter = QdrantAdapter()
