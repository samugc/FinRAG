
import pytest

from finrag.documents import Chunk
from finrag.rerank import rerank_pages
from finrag.retrieval import SearchResult


class FakeCrossEncoder:
    def predict(self, pairs):
        return [0.2, 0.9]


def test_rerank_pages_changes_order():
    hits = [
        SearchResult(Chunk("report:p5", 0, "Revenue"), 0.9),
        SearchResult(Chunk("report:p7", 0, "Operating profit"), 0.8),
        SearchResult(Chunk("report:p9", 0, "Debt"), 0.7),
    ]

    result = rerank_pages(
        "What was operating profit?",
        hits,
        FakeCrossEncoder(),
        top_n=2,
    )

    assert result == [7, 5, 9]


def test_rerank_rejects_invalid_top_n():
    with pytest.raises(ValueError):
        rerank_pages("Revenue?", [], FakeCrossEncoder(), top_n=0)

def test_rerank_uses_multiple_chunks_per_page():
    hits = [
        SearchResult(Chunk("report:p5", 0, "General information"), 0.9),
        SearchResult(Chunk("report:p7", 0, "General results"), 0.8),
        SearchResult(Chunk("report:p5", 1, "Revenue"), 0.7),
        SearchResult(Chunk("report:p7", 1, "Operating profit"), 0.6),
    ]

    class FakeModel:
        def predict(self, pairs):
            assert len(pairs) == 4
            return [0.8, 0.3, 0.1, 0.95]

    result = rerank_pages(
        "What was operating profit?",
        hits,
        FakeModel(),
        top_n=2,
        chunks_per_page=3,
    )

    assert result == [7, 5]