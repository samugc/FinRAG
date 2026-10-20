from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Document:
    source: str
    text: str


@dataclass
class Chunk:
    source: str
    chunk_id: int
    text: str
    document_source: str | None = None
    page_index: int | None = None


@dataclass
class Page:
    doc_name: str
    page_index: int
    text: str

def load_document(path: str | Path) -> Document:
    path = Path(path)

    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)
    else:
        text = path.read_text(encoding="utf-8")

    return Document(source=path.name, text=text)


def load_corpus(directory: str | Path) -> list[Document]:
    directory = Path(directory)

    paths = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}
    ]

    return [load_document(path) for path in sorted(paths)]


def chunk_document(
    document: Document,
    chunk_size: int = 180,
    overlap: int = 30,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    words = document.text.split()
    step = chunk_size - overlap

    chunks = []

    for start in range(0, len(words), step):
        block = words[start : start + chunk_size]

        if not block:
            break

        chunks.append(
            Chunk(
                source=document.source,
                chunk_id=len(chunks),
                text=" ".join(block),
            )
        )

        if start + chunk_size >= len(words):
            break

    return chunks


def chunk_corpus(
    documents: list[Document],
    chunk_size: int = 180,
    overlap: int = 30,
) -> list[Chunk]:
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")

    chunks = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    return chunks

def load_pdf_pages(path: str | Path) -> list[Page]:
    path = Path(path)
    reader = PdfReader(path)

    pages = []

    for i, pdf_page in enumerate(reader.pages):
        text = pdf_page.extract_text() or ""

        pages.append(
            Page(
                doc_name=path.stem,
                page_index=i,
                text=text,
            )
        )

    return pages