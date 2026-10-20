from dataclasses import dataclass

from finrag.context import format_context
from finrag.documents import Chunk
from finrag.evaluation import rank_pages
from finrag.fusion import reciprocal_rank_fusion
from finrag.query_planning import plan_queries
from finrag.retrieval import SearchResult


@dataclass(frozen=True)
class Evidence:
    source: str
    text: str


def find_evidence_chunks(
    evidence: Evidence,
    chunks: list[Chunk],
) -> list[int]:
    """Find chunks containing the evidence in its source document."""
    if not evidence.source.strip():
        raise ValueError("Evidence source cannot be empty")

    if not evidence.text.strip():
        raise ValueError("Evidence text cannot be empty")

    matches = []

    for chunk in chunks:
        if chunk.source != evidence.source:
            continue

        if evidence.text in chunk.text:
            matches.append(chunk.chunk_id)

    return matches


def build_context(
    hits: list[SearchResult],
    max_chunks: int = 5,
) -> str:
    """Format distinct, non-empty chunks, keeping their first occurrence."""
    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive")

    selected = []
    seen = set()

    for hit in hits:
        chunk = hit.chunk
        key = (chunk.source, chunk.chunk_id)

        if key in seen or not chunk.text.strip():
            continue

        seen.add(key)

        selected.append(SearchResult(
            Chunk(chunk.source, chunk.chunk_id, chunk.text.strip()),
            hit.score,
        ))

        if len(selected) >= max_chunks:
            break

    return format_context(selected)


def select_page_chunks(
    ranked_pages: list[int],
    hits: list[SearchResult],
    max_pages: int = 5,
    chunks_per_page: int = 2,
) -> list[SearchResult]:
    """Select nearby chunks in reading order from pages of a single document."""
    if max_pages <= 0 or chunks_per_page <= 0:
        raise ValueError("Limits must be positive")

    selected = []

    for page in dict.fromkeys(ranked_pages[:max_pages]):
        page_hits = [
            hit for hit in hits
            if int(hit.chunk.source.rsplit(":p", 1)[1]) == page
        ]

        if not page_hits:
            continue

        best = max(page_hits, key=lambda hit: hit.score)
        center = best.chunk.chunk_id

        by_id = {
            hit.chunk.chunk_id: hit
            for hit in page_hits
        }

        nearest = sorted(
            by_id,
            key=lambda i: (abs(i - center), i),
        )[:chunks_per_page]

        for chunk_id in sorted(nearest):
            selected.append(by_id[chunk_id])

    return selected


def retrieve_evidence(
    question: str,
    retriever,
    max_pages: int = 5,
    chunks_per_page: int = 2,
) -> list[SearchResult]:
    """Retrieve evidence from a fitted, single-document retriever.

    Fuse page rankings across queries, then select neighbors using the
    original query's chunk scores. Sources must end in :p<page index>.
    """

    if max_pages <= 0 or chunks_per_page <= 0:
        raise ValueError("Limits must be positive")

    queries = plan_queries(question)

    all_hits = [
        retriever.search(query, k=len(retriever.chunks))
        for query in queries
    ]

    rankings = [
        rank_pages(hits)
        for hits in all_hits
    ]

    if len(rankings) == 1:
        ranked_pages = rankings[0]
    else:
        ranked_pages = reciprocal_rank_fusion(rankings)

    return select_page_chunks(
        ranked_pages,
        all_hits[0],
        max_pages=max_pages,
        chunks_per_page=chunks_per_page,
    )
