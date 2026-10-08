from finrag.answer import extractive_answer
from finrag.documents import Chunk
from finrag.retrieval import SearchResult


def test_extractive_answer_uses_best_result():
    result = SearchResult(
        chunk=Chunk(
            source="report.txt",
            chunk_id=0,
            text="Operating profit was 58 million euros.",
        ),
        score=0.8,
    )

    answer = extractive_answer([result], min_score=0.1)

    assert answer.sufficient_evidence
    assert answer.text == result.chunk.text


def test_extractive_answer_can_abstain():
    result = SearchResult(
        chunk=Chunk(
            source="report.txt",
            chunk_id=0,
            text="Unrelated content.",
        ),
        score=0.0,
    )

    answer = extractive_answer([result], min_score=0.05)

    assert not answer.sufficient_evidence


def test_extractive_answer_without_results():
    assert not extractive_answer([]).sufficient_evidence
