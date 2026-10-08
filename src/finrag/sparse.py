from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from finrag.documents import Chunk
from finrag.retrieval import SearchResult


class SparseRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.chunks: list[Chunk] = []
        self.matrix = None

    def fit(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Corpus must contain at least one chunk")
        texts = [chunk.text for chunk in chunks]
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(texts)
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.chunks = list(chunks)

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        if self.matrix is None:
            raise RuntimeError("Retriever has not been fitted")

        if k <= 0:
            raise ValueError("k must be positive")
        if not query.strip():
            raise ValueError("Query must not be empty")

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        top_indices = (-scores).argsort(kind="stable")[:k]

        return [
            SearchResult(
                chunk=self.chunks[i],
                score=float(scores[i]),
            )
            for i in top_indices
        ]
