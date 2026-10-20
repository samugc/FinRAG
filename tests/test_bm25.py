
import pytest

from finrag.bm25 import BM25Retriever
from finrag.documents import Chunk


def test_bm25_ranks_relevant_chunk_first():
    chunks = [
        Chunk("report:p0", 0, "Revenue increased"),
        Chunk("report:p1", 0, "Debt securities and bonds"),
        Chunk("report:p2", 0, "Operating cash flow"),
    ]

    retriever = BM25Retriever()
    retriever.fit(chunks)

    results = retriever.search("debt securities", k=2)

    assert results[0].chunk.source == "report:p1"
    assert len(results) == 2


def test_bm25_rejects_empty_corpus():
    with pytest.raises(ValueError):
        BM25Retriever().fit([])


def test_bm25_rejects_invalid_queries():
    retriever = BM25Retriever()
    retriever.fit([Chunk("report:p0", 0, "Revenue")])

    with pytest.raises(ValueError):
        retriever.search("")

    with pytest.raises(ValueError):
        retriever.search("revenue", k=0)
