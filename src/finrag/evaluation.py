import json
from pathlib import Path


def load_questions(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def recall_at_k(
    retriever,
    questions: list[dict],
    k: int,
) -> float:
    """Fraction of questions with at least one accepted evidence chunk in top-k."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not questions:
        raise ValueError("Evaluation needs at least one question")

    hits = 0

    for item in questions:
        results = retriever.search(item["question"], k=k)

        found = any(
            result.chunk.source == item["source"]
            and result.chunk.chunk_id in item["chunk_ids"]
            for result in results
        )

        if found:
            hits += 1

    return hits / len(questions)
