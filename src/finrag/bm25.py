
import re

import numpy as np
from rank_bm25 import BM25Okapi

from finrag.documents import Chunk
from finrag.retrieval import SearchResult


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.model = None

    def fit(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Corpus must contain at least one chunk")

        self.chunks = list(chunks)
        tokenized = [tokenize(chunk.text) for chunk in chunks]
        self.model = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        if self.model is None:
            raise RuntimeError("Retriever has not been fitted")

        if k <= 0:
            raise ValueError("k must be positive")

        if not query.strip():
            raise ValueError("Query must not be empty")

        scores = self.model.get_scores(tokenize(query))
        indices = np.argsort(-scores, kind="stable")[:k]

        return [
            SearchResult(
                chunk=self.chunks[i],
                score=float(scores[i]),
            )
            for i in indices
        ]
