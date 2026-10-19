import json
from pathlib import Path


def load_questions(path: str | Path) -> list[dict]:
    """Load and validate accepted-evidence retrieval questions."""
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)

    if not isinstance(questions, list) or not questions:
        raise ValueError("Evaluation questions must be a non-empty list")

    for index, item in enumerate(questions):
        if not isinstance(item, dict):
            raise ValueError(f"Question {index} must be an object")

        for field in ("question", "source"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"Question {index} needs a non-empty {field}")

        ids = item.get("chunk_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(chunk_id) is not int or chunk_id < 0 for chunk_id in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError(
                f"Question {index} needs distinct non-negative integer chunk_ids"
            )

    return questions


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
