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

def test_document_scope_filters_retrieval_and_is_cached(tmp_path):
    (tmp_path / "a.txt").write_text("Revenue 42 euros", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Operating profit 58 euros", encoding="utf-8")
    pipeline = FinRAGPipeline(tmp_path)
    result = pipeline.query("revenue", source="b.txt")
    assert result.source == "b.txt"
    assert result.evidence
    assert all(hit.chunk.source == "b.txt" for hit in result.evidence)
    cached = pipeline._retrievers[("tfidf", "b.txt")]
    pipeline.query("revenue", source="b.txt")
    assert pipeline._retrievers[("tfidf", "b.txt")] is cached
    assert pipeline.query("revenue").source is None


def test_document_scope_rejects_unknown_and_empty(tmp_path):
    (tmp_path / "a.txt").write_text("Revenue 42 euros", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    pipeline = FinRAGPipeline(tmp_path)
    with pytest.raises(ValueError, match="Unknown document"):
        pipeline.query("revenue", source="missing.txt")
    with pytest.raises(ValueError, match="no extractable"):
        pipeline.query("revenue", source="empty.txt")
