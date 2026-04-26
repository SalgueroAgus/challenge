"""Unit tests for hybrid search: SparseEmbeddingService and QdrantAdapter.hybrid_search."""

from unittest.mock import MagicMock, patch

from qdrant_client.models import Fusion, FusionQuery, SparseVector

from app.adapters.qdrant_adapter import QdrantAdapter
from app.services.embedding_service import SparseEmbeddingService


class TestSparseEmbeddingService:
    def test_embed_one_returns_sparse_vector(self):
        svc = SparseEmbeddingService()
        result = svc.embed_one("Hornero rufous nest Argentina")

        assert isinstance(result, SparseVector)
        assert len(result.indices) > 0
        assert len(result.values) == len(result.indices)

    def test_embed_batch_returns_list_of_sparse_vectors(self):
        svc = SparseEmbeddingService()
        texts = ["Rufous Hornero", "Patagonian Mockingbird"]
        results = svc.embed_batch(texts)

        assert len(results) == 2
        for sv in results:
            assert isinstance(sv, SparseVector)
            assert len(sv.indices) > 0

    def test_sparse_values_are_positive(self):
        svc = SparseEmbeddingService()
        sv = svc.embed_one("Argentine birds taxonomy")
        assert all(v > 0 for v in sv.values)


class TestQdrantAdapterHybridSearch:
    def _make_adapter(self) -> QdrantAdapter:
        with patch("app.adapters.qdrant_adapter.QdrantClient"):
            return QdrantAdapter()

    def test_hybrid_search_calls_query_points_with_prefetch_and_rrf(self):
        adapter = self._make_adapter()
        mock_point = MagicMock()
        mock_point.score = 0.95
        adapter._client.query_points.return_value = MagicMock(points=[mock_point])

        dense = [0.1] * 384
        sparse = SparseVector(indices=[1, 5, 42], values=[0.8, 0.3, 0.5])

        result = adapter.hybrid_search(dense_vector=dense, sparse_vector=sparse, top_k=5)

        adapter._client.query_points.assert_called_once()
        call_kwargs = adapter._client.query_points.call_args.kwargs

        # Two prefetch branches: dense + sparse
        prefetches = call_kwargs["prefetch"]
        assert len(prefetches) == 2
        usings = {p.using for p in prefetches}
        assert usings == {"dense", "sparse"}

        # Each prefetch fetches more than top_k to give RRF enough candidates
        for p in prefetches:
            assert p.limit >= 5

        # Outer query must be RRF fusion
        assert isinstance(call_kwargs["query"], FusionQuery)
        assert call_kwargs["query"].fusion == Fusion.RRF

        assert call_kwargs["limit"] == 5
        assert result == [mock_point]

    def test_hybrid_search_applies_source_filter(self):
        adapter = self._make_adapter()
        adapter._client.query_points.return_value = MagicMock(points=[])

        dense = [0.0] * 384
        sparse = SparseVector(indices=[1], values=[1.0])

        adapter.hybrid_search(
            dense_vector=dense,
            sparse_vector=sparse,
            top_k=3,
            source_filter="birds_part1.pdf",
        )

        call_kwargs = adapter._client.query_points.call_args.kwargs
        query_filter = call_kwargs.get("query_filter")
        assert query_filter is not None
        assert query_filter.must[0].key == "source"
        assert query_filter.must[0].match.value == "birds_part1.pdf"

    def test_hybrid_search_no_filter_when_source_filter_is_none(self):
        adapter = self._make_adapter()
        adapter._client.query_points.return_value = MagicMock(points=[])

        adapter.hybrid_search(
            dense_vector=[0.0] * 384,
            sparse_vector=SparseVector(indices=[1], values=[1.0]),
            top_k=3,
            source_filter=None,
        )

        call_kwargs = adapter._client.query_points.call_args.kwargs
        assert call_kwargs.get("query_filter") is None
