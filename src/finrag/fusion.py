
from collections import defaultdict


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    constant: int = 60,
) -> list[int]:
    """Combine page rankings using Reciprocal Rank Fusion."""
    if constant <= 0:
        raise ValueError("constant must be positive")

    scores = defaultdict(float)

    for ranking in rankings:
        unique_pages = dict.fromkeys(ranking)

        for position, page in enumerate(unique_pages, start=1):
            scores[page] += 1.0 / (constant + position)

    return sorted(
        scores,
        key=lambda page: (-scores[page], page),
    )
