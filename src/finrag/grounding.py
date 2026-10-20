"""Small conservative checks applied before generation.

These checks can establish that a required fact is *missing* from the
retrieved text. Passing them does not establish that an answer is correct.
"""
import re

from finrag.retrieval import SearchResult


_YEAR = re.compile(r"(?<![0-9])(?:19|20)[0-9]{2}(?![0-9])")


def evidence_covers_requested_years(
    question: str,
    evidence: list[SearchResult],
) -> bool:
    """Reject requests for explicit years not present in the retrieved text.

    Works for both extractive and generated answers. If the query has no
    explicit four-digit year, no temporal restriction is imposed. This is
    deliberately conservative: year presence alone does not prove that a
    numerical claim relates to that year.
    """
    requested = set(_YEAR.findall(question))
    if not requested:
        return True
    observed = {
        year
        for result in evidence
        for year in _YEAR.findall(result.chunk.text)
    }
    return requested <= observed
