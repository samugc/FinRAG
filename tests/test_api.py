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


def _make_pdf(text: str) -> bytes:
    from io import BytesIO
    from pypdf import PdfWriter
    # Blank PDFs have no extractable text, so this helper tests validation.
    stream = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.write(stream)
    return stream.getvalue()


def test_pdf_upload_rejects_invalid_files_and_requires_header(tmp_path):
    client = TestClient(create_app(lambda: FinRAGPipeline(tmp_path), corpus_dir=tmp_path))
    body = {"file": ("example.pdf", b"not-a-pdf", "application/pdf")}
    assert client.post("/documents", files=body).status_code == 403
    assert client.post("/documents", files=body, headers={"X-FinRAG-Upload": "1"}).status_code == 422
    assert not (tmp_path / "example.pdf").exists()


def test_pdf_upload_rejects_scanned_pdf_and_duplicate(tmp_path):
    (tmp_path / "report.txt").write_text("Profit 58 million euros", encoding="utf-8")
    client = TestClient(create_app(lambda: FinRAGPipeline(tmp_path), corpus_dir=tmp_path))
    headers = {"X-FinRAG-Upload": "1"}
    empty_pdf = _make_pdf("")
    response = client.post("/documents", files={"file": ("scan.pdf", empty_pdf, "application/pdf")}, headers=headers)
    assert response.status_code == 422
    assert "No selectable text" in response.json()["detail"]
    assert not (tmp_path / "scan.pdf").exists()
    response = client.post("/documents", files={"file": ("report.txt", empty_pdf, "application/pdf")}, headers=headers)
    assert response.status_code == 422
    response = client.post("/documents", files={"file": ("../escape.pdf", empty_pdf, "application/pdf")}, headers=headers)
    assert response.status_code == 422
    assert not (tmp_path.parent / "escape.pdf").exists()

def test_pdf_upload_updates_searchable_library_without_restart(tmp_path):
    from io import BytesIO
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

    (tmp_path / "existing.txt").write_text("Existing financial report.", encoding="utf-8")
    writer = PdfWriter()
    page = writer.add_blank_page(width=500, height=500)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref}),
    })
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 50 400 Td (Acme operating profit was 123 million euros.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = BytesIO()
    writer.write(buffer)
    payload = buffer.getvalue()

    client = TestClient(create_app(lambda: FinRAGPipeline(tmp_path), corpus_dir=tmp_path))
    assert client.get("/health").json()["documents"] == 1
    response = client.post("/documents", files={"file": ("annual.pdf", payload, "application/pdf")},
                           headers={"X-FinRAG-Upload": "1"})
    assert response.status_code == 201, response.text
    assert response.json()["source"] == "annual.pdf"
    assert response.json()["chunks"] > 0
    assert client.get("/health").json()["documents"] == 2
    assert any(d["source"] == "annual.pdf" for d in client.get("/documents").json()["documents"])
    query = client.post("/query", json={"question": "What was Acme operating profit?", "source": "annual.pdf"})
    assert query.status_code == 200, query.text
    assert "123 million euros" in query.json()["answer"]
    assert all(hit["source"] == "annual.pdf" for hit in query.json()["evidence"])
    dupe = client.post("/documents", files={"file": ("annual.pdf", payload, "application/pdf")},
                       headers={"X-FinRAG-Upload": "1"})
    assert dupe.status_code == 409


def test_original_pdf_can_be_opened_only_if_registered(tmp_path):
    (tmp_path / "report.txt").write_text("Revenue 42 euros", encoding="utf-8")
    client = TestClient(create_app(lambda: FinRAGPipeline(tmp_path), corpus_dir=tmp_path))
    assert client.get("/documents/report.txt/file").status_code == 404
    assert client.get("/documents/missing.pdf/file").status_code == 404
    assert client.get("/documents/..%2Fother.pdf/file").status_code == 404
