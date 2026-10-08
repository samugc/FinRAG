from dataclasses import dataclass

from finrag.retrieval import SearchResult


@dataclass
class Answer:
    text: str
    evidence: list[SearchResult]
    sufficient_evidence: bool


def extractive_answer(
    results: list[SearchResult],
    min_score: float = 0.0,
) -> Answer:
    if not results or results[0].score <= min_score:
        return Answer(
            text="The available documents do not contain enough evidence to answer this question.",
            evidence=results,
            sufficient_evidence=False,
        )

    return Answer(
        text=results[0].chunk.text,
        evidence=results,
        sufficient_evidence=True,
    )
