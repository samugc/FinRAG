import numpy as np
import pytest

from finrag.documents import Chunk
from finrag.sparse import SparseRetriever


@pytest.fixture(params=["sparse", "dense"])
def retriever(request, monkeypatch):
    if request.param == "sparse":
        return SparseRetriever()

    import finrag.dense as dense

    class Encoder:
        def __init__(self, model_name):
            pass

        def encode(self, texts, **kwargs):
            vectors = np.array([[text.count("revenue"), text.count("cash")]
                                for text in texts], dtype=float)
            return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    monkeypatch.setattr(dense, "SentenceTransformer", Encoder)
    return dense.DenseRetriever()


def test_retrieval_boundaries_and_ties(retriever):
    with pytest.raises(RuntimeError):
        retriever.search("revenue")
    with pytest.raises(ValueError):
        retriever.fit([])
    chunks = [Chunk("report.txt", 0, "revenue"),
              Chunk("report.txt", 1, "cash"),
              Chunk("other.txt", 0, "revenue")]
    retriever.fit(chunks)
    results = retriever.search("revenue", k=10)
    assert [result.chunk for result in results] == [chunks[0], chunks[2], chunks[1]]
    assert [result.score for result in results] == pytest.approx([1, 1, 0])
    for k in (0, -1):
        with pytest.raises(ValueError):
            retriever.search("revenue", k=k)
    with pytest.raises(ValueError):
        retriever.search("  ")
    with pytest.raises(ValueError):
        retriever.fit([])
    assert retriever.search("revenue", k=1)[0].chunk == chunks[0]
