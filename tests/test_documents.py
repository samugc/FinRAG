import pytest

from finrag.documents import Document, chunk_document, chunk_corpus, load_corpus


def test_chunk_document_keeps_source():
    document = Document(
        source="report.txt",
        text="one two three four five six seven eight",
    )

    chunks = chunk_document(
        document,
        chunk_size=4,
        overlap=1,
    )

    assert chunks[0].source == "report.txt"
    assert chunks[0].text == "one two three four"
    assert chunks[1].text == "four five six seven"
    assert [chunk.chunk_id for chunk in chunks] == [0, 1, 2]
    assert chunks[2].text == "seven eight"
    assert all(chunk.source == "report.txt" for chunk in chunks)



@pytest.mark.parametrize("size, overlap", [(0, 0), (3, -1), (3, 3)])
def test_invalid_chunk_settings_even_without_documents(size, overlap):
    with pytest.raises(ValueError):
        chunk_document(Document("empty.txt", ""), size, overlap)
    with pytest.raises(ValueError):
        chunk_corpus([], size, overlap)


def test_empty_text_and_exact_boundary():
    assert chunk_document(Document("empty.txt", "  ")) == []
    chunks = chunk_document(Document("report.txt", "one two three four"), 4, 1)
    assert len(chunks) == 1


def test_loading_filters_directories_and_preserves_sources(tmp_path):
    (tmp_path / "folder.txt").mkdir()
    (tmp_path / "b.md").write_text("Revenue €42 million", encoding="utf-8")
    (tmp_path / "a.txt").write_text("Cash flow", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("ignored", encoding="utf-8")
    documents = load_corpus(tmp_path)
    assert [doc.source for doc in documents] == ["a.txt", "b.md"]
    assert documents[1].text == "Revenue €42 million"

from pypdf import PdfWriter
from finrag.documents import load_pdf_pages


def test_load_pdf_pages_preserves_metadata(tmp_path):
    path = tmp_path / "report.pdf"

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)

    writer.write(path)

    pages = load_pdf_pages(path)

    assert len(pages) == 2
    assert [p.page_index for p in pages] == [0, 1]
    assert all(p.doc_name == "report" for p in pages)
    assert all(p.text == "" for p in pages)