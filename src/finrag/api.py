"""HTTP interface for the existing local FinRAG pipeline."""

import os
from functools import lru_cache
from typing import Callable, Literal

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from finrag.pipeline import FinRAGPipeline


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    retriever: Literal["tfidf", "minilm"] = "tfidf"
    generator: Literal["extractive", "qwen"] = "extractive"
    k: int = Field(default=3, ge=1, le=20, strict=True)
    min_score: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)

    @field_validator("question")
    @classmethod
    def nonempty_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value.strip()


class EvidenceItem(BaseModel):
    source: str
    chunk_id: int
    text: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sufficient_evidence: bool
    retriever: str
    generator: str
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
) -> FastAPI:
    app = FastAPI(title="FinRAG API", version="0.1.0")

    if pipeline_factory is None:
        pipeline_factory = lambda: FinRAGPipeline(
            os.environ.get("FINRAG_CORPUS", "data/raw")
        )

    # One fitted pipeline per server process. Heavy models load on first use.
    get_pipeline = lru_cache(maxsize=1)(pipeline_factory)

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
            counts[chunk.source] += 1
        return DocumentsResponse(
            documents=[
                DocumentItem(source=source, chunks=count)
                for source, count in sorted(counts.items())
            ]
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
            )
        except (OSError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail="Query backend unavailable") from exc

        return QueryResponse(
            question=result.question,
            answer=result.answer,
            sufficient_evidence=result.sufficient_evidence,
            retriever=result.retriever,
            generator=result.generator,
            evidence=[
                EvidenceItem(
                    source=hit.chunk.source,
                    chunk_id=hit.chunk.chunk_id,
                    text=hit.chunk.text,
                    score=hit.score,
                )
                for hit in result.evidence
            ],
        )

    return app


app = create_app()
