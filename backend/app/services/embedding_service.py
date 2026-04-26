from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client.models import SparseVector

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
        logger.info("Loading embedding model: %s", settings.embed_model)
        self._model = TextEmbedding(model_name=settings.embed_model)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_one(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]


class SparseEmbeddingService:
    def __init__(self) -> None:
        logger.info("Loading sparse embedding model: Qdrant/bm25")
        self._model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def embed_batch(self, texts: list[str]) -> list[SparseVector]:
        return [
            SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())
            for emb in self._model.embed(texts)
        ]

    def embed_one(self, text: str) -> SparseVector:
        return self.embed_batch([text])[0]


embedding_service = EmbeddingService()
sparse_embedding_service = SparseEmbeddingService()
