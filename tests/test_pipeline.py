import pytest

from finrag.pipeline import FinRAGPipeline


@pytest.fixture
def pipeline(tmp_path):
    (tmp_path / "report.txt").write_text(
        "Northstar operating profit was 58 million euros in 2025.",
        encoding="utf-8",
    )
    return FinRAGPipeline(tmp_path)


def test_query_retrieves_source_and_caches_retriever(pipeline):
    first = pipeline.query("operating profit")
    assert first.sufficient_evidence
    assert "58 million" in first.answer
    assert first.evidence[0].chunk.source == "report.txt"
    fitted = pipeline._retrievers["tfidf"]
    pipeline.query("operating profit")
    assert pipeline._retrievers["tfidf"] is fitted


@pytest.mark.parametrize("params", [
    {"question": " "},
    {"question": "revenue", "k": 0},
    {"question": "revenue", "min_score": -1},
    {"question": "revenue", "retriever": "rrf"},
    {"question": "revenue", "generator": "fake"},
])
def test_pipeline_rejects_invalid_input(pipeline, params):
    with pytest.raises(ValueError):
        pipeline.query(**params)


def test_empty_corpus_raises(tmp_path):
    with pytest.raises(ValueError, match="extractable"):
        FinRAGPipeline(tmp_path)
