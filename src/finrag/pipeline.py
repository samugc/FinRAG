"""Local orchestration with immutable evidence snapshots per query."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from threading import Lock
from time import perf_counter

from finrag.answer import extractive_answer
from finrag.documents import (
    Chunk,
    Document,
    chunk_document,
    load_document,
    load_pdf_pages,
)
from finrag.evidence import retrieve_evidence
from finrag.generation import AnswerStatus, GenerationResult
from finrag.grounding import check_periods
from finrag.retrieval import SearchResult
from finrag.sparse import SparseRetriever


@dataclass
class QueryResult:
    question: str
    answer: str
    sufficient_evidence: bool
    retriever: str
    generator: str
    source: str | None
    evidence: list[SearchResult]
    status: AnswerStatus
    reason: str
    retrieval_mode: str
    period_check: dict
    timings: dict
    calculation: dict | None = None


class FinRAGPipeline:
    def __init__(self, corpus: str | Path = "data/raw", *, upload_dir: str | Path | None = None):
        self.documents = []
        self.chunks = []
        self.page_chunks = []
        self.pages = {}
        self.files = {}
        roots = [Path(corpus)]
        if upload_dir is not None and Path(upload_dir) != roots[0]:
            roots.append(Path(upload_dir))
        for root in roots:
            if not root.exists():
                if root == roots[0]:
                    raise FileNotFoundError("Corpus directory unavailable")
                continue
            for path in sorted(root.iterdir()):
                if path.is_symlink() or not path.is_file() or path.suffix.lower() not in {".pdf", ".txt", ".md"}:
                    continue
                if path.name in self.files:
                    raise ValueError("Duplicate document names across corpus and uploads")
                self.files[path.name] = path.resolve()
                if path.suffix.lower() == ".pdf":
                    pages = load_pdf_pages(path)
                    self.documents.append(Document(path.name, "\n".join(p.text for p in pages)))
                    for page in pages:
                        self.pages[(path.name, page.page_index)] = page.text
                        doc = Document(f"{path.name}:p{page.page_index}", page.text)
                        for size, overlap, target in [(25, 5, self.chunks), (80, 15, self.page_chunks)]:
                            for chunk in chunk_document(doc, chunk_size=size, overlap=overlap):
                                chunk.document_source = path.name
                                chunk.page_index = page.page_index
                                target.append(chunk)
                else:
                    doc = load_document(path)
                    self.documents.append(doc)
                    self.chunks.extend(chunk_document(doc, chunk_size=25, overlap=5))
        self._retrievers = {}
        self._generator = None
        self._initialization_lock = Lock()
        self._generation_lock = Lock()

    @staticmethod
    def source_of(chunk):
        return chunk.document_source or chunk.source

    def _retriever(self, name, source=None, retrieval_mode="direct"):
        key = (name if source is None else (name, source)) if retrieval_mode == "direct" else (name, source, retrieval_mode)
        with self._initialization_lock:
            if key not in self._retrievers:
                if name == "tfidf":
                    retriever = SparseRetriever()
                else:
                    from finrag.dense import DenseRetriever
                    retriever = DenseRetriever()
                chunks = self.chunks if retrieval_mode == "direct" else self.page_chunks
                chunks = [c for c in chunks if source is None or self.source_of(c) == source]
                retriever.fit(chunks)
                self._retrievers[key] = retriever
            return self._retrievers[key]

    def _local_generator(self):
        with self._initialization_lock:
            if self._generator is None:
                from finrag.generation import LocalGenerator
                self._generator = LocalGenerator()
            return self._generator

    def query(self, question, *, retriever="tfidf", generator="extractive", k=3,
              min_score=0.0, source=None, retrieval_mode="direct", max_pages=5,
              chunks_per_page=2):
        started = perf_counter()
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must not be empty")
        if retriever not in ("tfidf", "minilm") or generator not in ("extractive", "qwen"):
            raise ValueError("Unsupported retriever or generator")
        if type(k) is not int or k <= 0:
            raise ValueError("k must be a positive integer")
        if not isfinite(min_score) or not 0 <= min_score <= 1:
            raise ValueError("min_score must be between 0 and 1")
        if retrieval_mode not in ("direct", "multiquery"):
            raise ValueError("Unsupported retrieval mode")
        if any(type(n) is not int or n <= 0 for n in (max_pages, chunks_per_page)):
            raise ValueError("Page limits must be positive integers")
        if source is not None and source not in self.files:
            raise ValueError("Unknown document source")
        if retrieval_mode == "multiquery" and (source is None or not source.lower().endswith(".pdf")):
            raise ValueError("Multi-Query requires one selected PDF")
        available = [c for c in self.chunks if source is None or self.source_of(c) == source]
        if source is not None and not available:
            raise ValueError("Selected document contains no extractable text")
        if not available:
            evidence = []
        elif retrieval_mode == "multiquery":
            evidence = retrieve_evidence(question, self._retriever(retriever, source, retrieval_mode),
                                         max_pages=max_pages, chunks_per_page=chunks_per_page)
        else:
            evidence = self._retriever(retriever, source).search(question, k=k)
        period_check = check_periods(question, evidence, self.pages, self.documents)
        retrieval_seconds = perf_counter() - started
        if not evidence:
            generated = GenerationResult("", AnswerStatus.NO_EVIDENCE, "No evidence text supplied")
        elif period_check["status"] == "insufficient_context":
            generated = GenerationResult("", AnswerStatus.INSUFFICIENT_CONTEXT,
                                         "Requested period not verified in the available reporting context")
        elif generator == "extractive":
            result = extractive_answer(evidence, min_score=min_score)
            generated = GenerationResult(result.text,
                AnswerStatus.PRODUCED if result.sufficient_evidence else AnswerStatus.INSUFFICIENT_CONTEXT,
                "Extractive passage; not fact-verified" if result.sufficient_evidence else "Extractive score cutoff not met")
        else:
            headers = {}
            for reference in period_check["references"]:
                key = (reference["source"], reference["page_index"])
                headers.setdefault(key, set()).add(reference["quote"])
            for (document, page), quotes in headers.items():
                text = "\n".join(sorted(quotes))
                if not all(any(quote in hit.chunk.text for hit in evidence) for quote in quotes):
                    evidence.append(SearchResult(Chunk(
                        f"{document}:p{page}" if page is not None else document,
                        -1, text, document, page,
                    ), 0.0))
            with self._generation_lock:
                generated = self._local_generator().generate(question, evidence)
        return QueryResult(question, generated.raw_answer,
            generated.status is AnswerStatus.PRODUCED, retriever, generator, source,
            evidence, generated.status, generated.reason, retrieval_mode, period_check,
            {"retrieval_seconds": retrieval_seconds, "generation_seconds": generated.elapsed_seconds,
             "total_seconds": perf_counter() - started})
