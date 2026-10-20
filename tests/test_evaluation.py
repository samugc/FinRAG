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


@pytest.mark.parametrize(
    "questions",
    [
        {},
        [],
        [7],
        [{"question": "", "source": "report.txt", "chunk_ids": [0]}],
        [{"question": "Revenue?", "source": "", "chunk_ids": [0]}],
        [{"question": "Revenue?", "source": "report.txt", "chunk_ids": []}],
        [{"question": "Revenue?", "source": "report.txt", "chunk_ids": [True]}],
        [{"question": "Revenue?", "source": "report.txt", "chunk_ids": [-1]}],
        [{"question": "Revenue?", "source": "report.txt", "chunk_ids": [0, 0]}],
    ],
)
def test_load_questions_rejects_invalid_annotations(tmp_path, questions):
    import json

    from finrag.evaluation import load_questions

    path = tmp_path / "questions.json"
    path.write_text(json.dumps(questions), encoding="utf-8")
    with pytest.raises(ValueError):
        load_questions(path)


def test_load_questions_preserves_valid_annotations(tmp_path):
    import json

    from finrag.evaluation import load_questions

    questions = [{
        "question": "Revenue in 2025?",
        "source": "report.txt",
        "chunk_ids": [0, 1],
        "type": "paraphrase",
    }]
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(questions), encoding="utf-8")
    assert load_questions(path) == questions

from finrag.evaluation import first_evidence_rank, rank_pages
from finrag.retrieval import SearchResult


def test_rank_pages_removes_duplicates():
    hits = [
        SearchResult(Chunk("AMD:p42", 0, "A"), 0.9),
        SearchResult(Chunk("AMD:p42", 1, "B"), 0.8),
        SearchResult(Chunk("AMD:p55", 0, "C"), 0.7),
    ]

    assert rank_pages(hits) == [42, 55]

def test_first_evidence_rank():
    assert first_evidence_rank([42, 55, 11], [55]) == 2
    assert first_evidence_rank([42, 55], [3]) is None

from finrag.evaluation import all_evidence_rank


def test_all_evidence_rank():
    assert all_evidence_rank([63, 42, 61], [61, 63]) == 3
    assert all_evidence_rank([63, 42], [61, 63]) is None