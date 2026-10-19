from fastapi.testclient import TestClient
import pytest

from finrag.api import create_app
from finrag.pipeline import FinRAGPipeline


@pytest.fixture
def client(tmp_path):
    (tmp_path / "report.txt").write_text(
        "Northstar operating profit was 58 million euros in 2025. ",
        encoding="utf-8",
    )
    return TestClient(create_app(lambda: FinRAGPipeline(tmp_path)))


def test_health(client):
    result = client.get("/health")
    assert result.status_code == 200
    assert result.json() == {"status": "ok", "documents": 1, "chunks": 1}


def test_query_returns_answer_and_source(client):
    result = client.post("/query", json={"question": "What was operating profit?"})
    assert result.status_code == 200
    data = result.json()
    assert "58 million euros" in data["answer"]
    assert data["sufficient_evidence"] is True
    assert data["retriever"] == "tfidf"
    assert data["evidence"][0]["source"] == "report.txt"
    assert data["evidence"][0]["chunk_id"] == 0
    assert data["evidence"][0]["score"] > 0


@pytest.mark.parametrize("payload", [
    {"question": "   "},
    {"question": ""},
    {"question": "Question?", "k": 0},
    {"question": "Question?", "k": 1.5},
    {"question": "Question?", "k": 21},
    {"question": "Question?", "retriever": "unregistered"},
    {"question": "Question?", "generator": "external"},
    {"question": "Question?", "min_score": -0.1},
    {"question": "Question?", "unexpected": "ignored?"},
])
def test_invalid_query_returns_422(client, payload):
    assert client.post("/query", json=payload).status_code == 422


def test_unavailable_corpus_returns_503(tmp_path):
    app = create_app(lambda: FinRAGPipeline(tmp_path / "missing"))
    client = TestClient(app)
    assert client.get("/health").status_code == 503
    assert client.post("/query", json={"question": "Profit?"}).status_code == 503


def test_min_score_abstention_and_evidence(client):
    result = client.post(
        "/query", json={"question": "Operating profit?", "min_score": 1.0}
    )
    assert result.status_code == 200
    assert result.json()["sufficient_evidence"] is False
    assert result.json()["evidence"]


def test_reuses_fitted_pipeline_between_requests(tmp_path):
    (tmp_path / "report.txt").write_text("Revenue 42 euros", encoding="utf-8")
    calls = []

    def factory():
        calls.append(1)
        return FinRAGPipeline(tmp_path)

    client = TestClient(create_app(factory))
    client.get("/health")
    client.post("/query", json={"question": "Revenue?"})
    client.post("/query", json={"question": "Revenue?"})
    assert calls == [1]

def test_documents_lists_corpus_without_leaking_content(client):
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json() == {
        "documents": [{"source": "report.txt", "chunks": 1}]
    }
    assert "Northstar operating profit" not in response.text


def test_documents_missing_corpus_returns_503(tmp_path):
    client = TestClient(create_app(lambda: FinRAGPipeline(tmp_path / "missing")))
    assert client.get("/documents").status_code == 503

def test_query_scopes_document_sources(tmp_path):
    (tmp_path / "alpha.txt").write_text("Revenue 42 euros", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("Profit 58 euros", encoding="utf-8")
    client = TestClient(create_app(lambda: FinRAGPipeline(tmp_path)))
    response = client.post(
        "/query", json={"question": "What was revenue?", "source": "beta.txt"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "beta.txt"
    assert data["evidence"]
    assert all(item["source"] == "beta.txt" for item in data["evidence"])
    assert client.post(
        "/query", json={"question": "Profit?", "source": "missing.txt"}
    ).status_code == 422
