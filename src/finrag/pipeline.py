"""Shared orchestration for the CLI and HTTP API.

Retrievers and the optional local generator are created once per process.
This module intentionally leaves ranking and evidence decisions to their
existing implementations.
"""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from threading import Lock

from finrag.answer import extractive_answer
from finrag.documents import chunk_corpus, load_corpus
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


class FinRAGPipeline:
    def __init__(self, corpus: str | Path = "data/raw") -> None:
        self.documents = load_corpus(corpus)
        self.chunks = chunk_corpus(self.documents, chunk_size=25, overlap=5)
        if not self.chunks:
            raise ValueError("Corpus contains no extractable text")

        self._retrievers = {}
        self._generator = None
        self._initialization_lock = Lock()
        self._generation_lock = Lock()

    def _retriever(self, name: str, source: str | None = None):
        key = name if source is None else (name, source)
        with self._initialization_lock:
            if key not in self._retrievers:
                if name == "tfidf":
                    retriever = SparseRetriever()
                elif name == "minilm":
                    from finrag.dense import DenseRetriever
                    retriever = DenseRetriever()
                else:
                    raise ValueError(f"Unknown retriever: {name}")
                chunks = self.chunks if source is None else [
                    chunk for chunk in self.chunks if chunk.source == source
                ]
                retriever.fit(chunks)
                self._retrievers[key] = retriever
            return self._retrievers[key]

    def _local_generator(self):
        with self._initialization_lock:
            if self._generator is None:
                from finrag.generation import LocalGenerator
                self._generator = LocalGenerator()
            return self._generator

    def query(
        self,
        question: str,
        *,
        retriever: str = "tfidf",
        generator: str = "extractive",
        k: int = 3,
        min_score: float = 0.0,
        source: str | None = None,
    ) -> QueryResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must not be empty")
        if retriever not in ("tfidf", "minilm"):
            raise ValueError("Unsupported retriever")
        if generator not in ("extractive", "qwen"):
            raise ValueError("Unsupported generator")
        if type(k) is not int or k <= 0:
            raise ValueError("k must be a positive integer")
        if not isfinite(min_score) or not 0 <= min_score <= 1:
            raise ValueError("min_score must be between 0 and 1")

        if source is not None:
            if not isinstance(source, str) or source not in {
                doc.source for doc in self.documents
            }:
                raise ValueError("Unknown document source")
            if not any(chunk.source == source for chunk in self.chunks):
                raise ValueError("Selected document contains no extractable text")

        evidence = self._retriever(retriever, source).search(question, k=k)
        if generator == "extractive":
            result = extractive_answer(evidence, min_score=min_score)
            answer, sufficient = result.text, result.sufficient_evidence
        else:
            local_generator = self._local_generator()
            with self._generation_lock:
                answer, sufficient = local_generator.answer(question, evidence)

        return QueryResult(
            question=question,
            answer=answer,
            sufficient_evidence=sufficient,
            retriever=retriever,
            generator=generator,
            source=source,
            evidence=evidence,
        )
