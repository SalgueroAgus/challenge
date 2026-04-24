from fastembed import TextEmbedding

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Wraps FastEmbed TextEmbedding. Model is downloaded on first use (~25 MB)."""

    def __init__(self) -> None:
        logger.info("Loading embedding model: %s", settings.embed_model)
        self._model = TextEmbedding(model_name=settings.embed_model)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns a list of float vectors."""
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text. Returns a float vector."""
        return self.embed_batch([text])[0]


embedding_service = EmbeddingService()
