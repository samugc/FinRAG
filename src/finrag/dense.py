import numpy as np
from sentence_transformers import SentenceTransformer

from finrag.documents import Chunk
from finrag.retrieval import SearchResult


class DenseRetriever:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)
        self.chunks: list[Chunk] = []
        self.embeddings = None

    def fit(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Corpus must contain at least one chunk")
        texts = [chunk.text for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        self.embeddings = embeddings
        self.chunks = list(chunks)

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        if self.embeddings is None:
            raise RuntimeError("Retriever has not been fitted")

        if k <= 0:
            raise ValueError("k must be positive")
        if not query.strip():
            raise ValueError("Query must not be empty")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(-scores, kind="stable")[:k]

        return [
            SearchResult(
                chunk=self.chunks[i],
                score=float(scores[i]),
            )
            for i in top_indices
        ]
