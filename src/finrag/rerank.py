
from finrag.evaluation import rank_pages


def rerank_pages(
    query,
    hits,
    model,
    top_n=20,
    chunks_per_page=1,
):
    """Rerank candidate pages using Cross-Encoder chunk scores."""
    if top_n <= 0 or chunks_per_page <= 0:
        raise ValueError("top_n and chunks_per_page must be positive")

    pages = rank_pages(hits)

    if not pages:
        return []

    candidates = pages[:top_n]
    selected = {page: [] for page in candidates}

    for hit in hits:
        page = int(hit.chunk.source.rsplit(":p", 1)[1])

        if page in selected and len(selected[page]) < chunks_per_page:
            selected[page].append(hit.chunk.text)

    pairs = [
        (query, text)
        for page in candidates
        for text in selected[page]
    ]

    scores = model.predict(pairs)

    page_scores = {}
    offset = 0

    for page in candidates:
        count = len(selected[page])
        page_scores[page] = max(
            float(score)
            for score in scores[offset:offset + count]
        )
        offset += count

    original_order = {
        page: i for i, page in enumerate(candidates)
    }

    reranked = sorted(
        candidates,
        key=lambda page: (
            -page_scores[page],
            original_order[page],
        ),
    )

    return reranked + pages[top_n:]
