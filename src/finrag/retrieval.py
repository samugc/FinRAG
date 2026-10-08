from dataclasses import dataclass

from finrag.documents import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
