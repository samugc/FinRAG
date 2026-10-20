"""HTTP interface for the existing local FinRAG pipeline."""

import os
import re
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Literal
from urllib.parse import quote

from fastapi import FastAPI, File, Header, HTTPException, Response, UploadFile
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pypdf import PdfReader

from finrag.generation import AnswerStatus
from finrag.pipeline import FinRAGPipeline

MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 500


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    retriever: Literal["tfidf", "minilm"] = "tfidf"
    generator: Literal["extractive", "qwen"] = "extractive"
    k: int = Field(default=3, ge=1, le=20, strict=True)
    min_score: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)
    source: str | None = Field(default=None, min_length=1, max_length=255)

    retrieval_mode: Literal["direct", "multiquery"] = "direct"
    max_pages: int = Field(default=5, ge=1, le=10, strict=True)
    chunks_per_page: int = Field(default=2, ge=1, le=5, strict=True)

    @field_validator("question")
    @classmethod
    def nonempty_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value.strip()


class EvidenceItem(BaseModel):
    citation_id: int
    page_index: int | None = None
    page_number: int | None = None
    kind: str = "chunk"
    source: str
    chunk_id: int
    text: str
    score: float


class QueryResponse(BaseModel):
    status: AnswerStatus
    reason: str
    error: str | None = None
    retrieval_mode: str
    period_check: dict
    timings: dict
    calculation: dict | None = None
    question: str
    answer: str
    sufficient_evidence: bool
    retriever: str
    generator: str
    source: str | None
    evidence: list[EvidenceItem]


class DocumentItem(BaseModel):
    source: str
    chunks: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentItem]


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    documents: int
    chunks: int


def create_app(
    pipeline_factory: Callable[[], FinRAGPipeline] | None = None,
    *,
    corpus_dir: str | Path | None = None,
    upload_dir: str | Path | None = None,
) -> FastAPI:
    app = FastAPI(title="FinRAG API", version="0.1.0")

    corpus_path = Path(corpus_dir if corpus_dir is not None else os.environ.get("FINRAG_CORPUS", "data/raw"))
    uploads = Path(upload_dir) if upload_dir is not None else (
        corpus_path if corpus_dir is not None else Path(os.environ.get("FINRAG_UPLOADS", ".local/uploads"))
    )
    if pipeline_factory is None:
        pipeline_factory = lambda: FinRAGPipeline(corpus_path, upload_dir=uploads)
    pipeline = None
    pipeline_lock = Lock()
    upload_lock = Lock()

    def get_pipeline():
        nonlocal pipeline
        with pipeline_lock:
            if pipeline is None:
                pipeline = pipeline_factory()
            return pipeline

    @app.get("/health", response_model=HealthResponse)
    def health(response: Response):
        try:
            pipeline = get_pipeline()
        except (OSError, UnicodeError, ValueError):
            response.status_code = 503
            return HealthResponse(status="unavailable", documents=0, chunks=0)
        return HealthResponse(
            status="ok",
            documents=len(pipeline.documents),
            chunks=len(pipeline.chunks),
        )

    @app.get("/documents", response_model=DocumentsResponse)
    def documents():
        try:
            pipeline = get_pipeline()
        except (OSError, UnicodeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Corpus unavailable") from exc

        counts = {document.source: 0 for document in pipeline.documents}
        for chunk in pipeline.chunks:
            counts[FinRAGPipeline.source_of(chunk)] += 1
        return DocumentsResponse(
            documents=[
                DocumentItem(source=source, chunks=count)
                for source, count in sorted(counts.items())
            ]
        )

    @app.get("/documents/{source}/file")
    def open_pdf_document(source: str):
        if Path(source).name != source or not source.lower().endswith(".pdf") or "\\" in source:
            raise HTTPException(status_code=404, detail="PDF not found")
        try:
            document_path = get_pipeline().files.get(source)
            if document_path is None or document_path.is_symlink():
                raise HTTPException(status_code=404, detail="PDF not found")
            roots = (corpus_path.resolve(), uploads.resolve())
            if document_path.parent not in roots:
                raise HTTPException(status_code=404, detail="PDF not found")
            # Open without following a replaced symlink, before sending any bytes.
            fd = os.open(document_path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as file:
                payload = file.read(MAX_PDF_BYTES + 1)
            if len(payload) > MAX_PDF_BYTES or not payload.startswith(b"%PDF-"):
                raise HTTPException(status_code=404, detail="PDF not available")
        except OSError as exc:
            raise HTTPException(status_code=404, detail="PDF not found") from exc
        return Response(payload, media_type="application/pdf", headers={
            "Content-Disposition": "inline; filename*=UTF-8''" + quote(source),
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "sandbox",
        })

    @app.post("/documents", response_model=DocumentItem, status_code=201)
    def upload_document(
        file: UploadFile = File(...),  # noqa: B008 - FastAPI declaration
        upload_guard: str | None = Header(default=None, alias="X-FinRAG-Upload"),
    ):
        nonlocal pipeline
        # Requiring a custom header prevents ordinary cross-site HTML form posts
        # from silently writing to this local-only service. Not authentication.
        if upload_guard != "1":
            raise HTTPException(status_code=403, detail="Upload header required")

        filename = file.filename or ""
        if (
            not re.fullmatch(r"[\w .()\-]{1,115}\.pdf", filename, re.IGNORECASE)
            or filename.startswith(".")
            or ".." in filename
            or "/" in filename
            or "\\" in filename
        ):
            raise HTTPException(status_code=422, detail="Use a simple PDF filename")

        if file.content_type not in {"application/pdf", "application/octet-stream"}:
            raise HTTPException(status_code=415, detail="Only PDF documents are supported")

        payload = file.file.read(MAX_PDF_BYTES + 1)
        if len(payload) > MAX_PDF_BYTES:
            raise HTTPException(status_code=413, detail="PDF exceeds the 25 MB limit")
        if not payload.startswith(b"%PDF-"):
            raise HTTPException(status_code=422, detail="Invalid PDF file")

        try:
            reader = PdfReader(BytesIO(payload), strict=False)
            if reader.is_encrypted:
                raise HTTPException(status_code=422, detail="Password-protected PDFs are unsupported")
            if len(reader.pages) > MAX_PDF_PAGES:
                raise HTTPException(status_code=413, detail="PDF exceeds the 500-page limit")
            text_blocks = [page.extract_text() or "" for page in reader.pages]
            if not any(text.strip() for text in text_blocks):
                raise HTTPException(status_code=422, detail="No selectable text found; scanned PDFs need OCR")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=422, detail="Unable to read this PDF") from exc

        # The corpus is changed only after validation. No overwrite of existing
        # files; a new pipeline is built so document filters/indexes are refreshed.
        with upload_lock:
            current = get_pipeline()
            if filename in current.files:
                raise HTTPException(status_code=409, detail="A document with that name already exists")
            target = uploads / filename
            created = False
            try:
                uploads.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    created = True
                    output.write(payload)
                candidate = pipeline_factory()
                with pipeline_lock:
                    pipeline = candidate
            except FileExistsError as exc:
                raise HTTPException(status_code=409, detail="A document with that name already exists") from exc
            except (OSError, UnicodeError, ValueError) as exc:
                if created:
                    target.unlink(missing_ok=True)
                raise HTTPException(status_code=503, detail="Unable to update local document library") from exc

        return DocumentItem(
            source=filename,
            chunks=sum(FinRAGPipeline.source_of(chunk) == filename for chunk in candidate.chunks),
        )

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest):
        try:
            pipeline = get_pipeline()
        except (OSError, UnicodeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Corpus unavailable") from exc

        try:
            result = pipeline.query(
                request.question,
                retriever=request.retriever,
                generator=request.generator,
                k=request.k,
                min_score=request.min_score,
                source=request.source,
                retrieval_mode=request.retrieval_mode,
                max_pages=request.max_pages,
                chunks_per_page=request.chunks_per_page,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail="Query backend unavailable") from exc

        return QueryResponse(
            status=result.status, reason=result.reason,
            error=result.reason if result.status is AnswerStatus.BACKEND_ERROR else None,
            retrieval_mode=result.retrieval_mode, period_check=result.period_check,
            timings=result.timings, calculation=result.calculation,
            question=result.question,
            answer=result.answer,
            sufficient_evidence=result.sufficient_evidence,
            retriever=result.retriever,
            generator=result.generator,
            source=result.source,
            evidence=[
                EvidenceItem(
                    citation_id=rank,
                    source=FinRAGPipeline.source_of(hit.chunk),
                    page_index=hit.chunk.page_index,
                    page_number=hit.chunk.page_index + 1 if hit.chunk.page_index is not None else None,
                    kind="reporting_context" if hit.chunk.chunk_id == -1 else "chunk",
                    chunk_id=hit.chunk.chunk_id,
                    text=hit.chunk.text,
                    score=hit.score,
                )
                for rank, hit in enumerate(result.evidence, 1)
            ],
        )

    return app


app = create_app()
