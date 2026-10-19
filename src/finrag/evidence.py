
from dataclasses import dataclass

from finrag.documents import Chunk


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
