import pytest

from finrag.documents import Chunk
from finrag.evaluation import recall_at_k
from finrag.sparse import SparseRetriever


def test_recall_at_k():
    chunks = [
        Chunk(
            source="report.txt",
            chunk_id=0,
            text="Revenue was 420 million euros in 2025.",
        ),
        Chunk(
            source="report.txt",
            chunk_id=1,
            text="The company employed 3200 people.",
        ),
    ]

    questions = [
        {
            "question": "What was revenue in 2025?",
            "source": "report.txt",
            "chunk_ids": [0],
        },
        {
            "question": "How many people did the company employ?",
            "source": "report.txt",
            "chunk_ids": [1],
        },
    ]

    retriever = SparseRetriever()
    retriever.fit(chunks)

    assert recall_at_k(retriever, questions, k=1) == 1.0


def test_recall_counts_questions_not_matching_chunks():
    retriever = SparseRetriever()
    retriever.fit([Chunk("report.txt", 0, "revenue"),
                   Chunk("report.txt", 1, "revenue cash"),
                   Chunk("other.txt", 2, "debt")])
    questions = [
        {"question": "revenue", "source": "report.txt", "chunk_ids": [1, 0]},
        {"question": "debt", "source": "report.txt", "chunk_ids": [2]},
    ]
    assert recall_at_k(retriever, questions, 1) == 0.5
    assert recall_at_k(retriever, questions, 3) == 0.5
    assert recall_at_k(retriever, [dict(questions[0], chunk_ids=[1])], 1) == 0
    assert recall_at_k(retriever, [dict(questions[0], chunk_ids=[1])], 3) == 1
    with pytest.raises(ValueError):
        recall_at_k(retriever, [], 1)
    with pytest.raises(ValueError):
        recall_at_k(retriever, questions, 0)
