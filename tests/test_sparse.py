from finrag.documents import Chunk
from finrag.sparse import SparseRetriever


def test_sparse_retrieval_ranks_relevant_chunk_first():
    chunks = [
        Chunk(
            source="report.txt",
            chunk_id=0,
            text="Revenue increased to 420 million euros in 2025.",
        ),
        Chunk(
            source="report.txt",
            chunk_id=1,
            text="The company employed 3200 people.",
        ),
    ]

    retriever = SparseRetriever()
    retriever.fit(chunks)

    results = retriever.search(
        "What was revenue in 2025?",
        k=1,
    )

    assert results[0].chunk.chunk_id == 0
    assert results[0].score > 0
