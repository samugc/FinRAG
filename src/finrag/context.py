from finrag.retrieval import SearchResult


def format_context(results: list[SearchResult]) -> str:
    parts = []

    for rank, result in enumerate(results, start=1):
        parts.append(
            f"[{rank}] {result.chunk.source} "
            f"(chunk {result.chunk.chunk_id}, score={result.score:.4f})\n"
            f"{result.chunk.text}"
        )

    return "\n\n".join(parts)
