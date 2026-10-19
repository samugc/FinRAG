import pytest

from finrag.generation import LocalGenerator, validate_answer


@pytest.mark.parametrize("text", ["", "[1]", "[1], [2].", "...",
                                 "INSUFFICIENT_EVIDENCE.", "€58 million [3]"])
def test_rejects_invalid_output(text):
    assert validate_answer(text, 2) == ("INSUFFICIENT_EVIDENCE", False)


def test_valid_output_is_not_proof_of_grounding():
    assert validate_answer("€58 million [1]", 2) == ("€58 million [1]", True)
    assert validate_answer("€58 million", 2) == ("€58 million", True)
    assert validate_answer("An unsupported claim [1]", 2)[1]


def test_no_evidence_abstains_without_generation():
    generator = LocalGenerator.__new__(LocalGenerator)
    assert generator.answer("Profit?", []) == ("INSUFFICIENT_EVIDENCE", False)

def test_numeric_guard_rejects_unsupported_dividend_or_score():
    from finrag.documents import Chunk
    from finrag.retrieval import SearchResult
    from finrag.generation import validate_numeric_evidence
    evidence = [SearchResult(chunk=Chunk(source="risk.txt", chunk_id=0,
        text="Foreign exchange exposure comes from US dollars and British pounds."), score=0.0920)]
    assert validate_numeric_evidence("0.0920", evidence) == ("INSUFFICIENT_EVIDENCE", False)
    assert validate_numeric_evidence("€58 million", evidence) == ("INSUFFICIENT_EVIDENCE", False)


def test_numeric_guard_allows_supported_number_and_nonnumeric_answer():
    from finrag.documents import Chunk
    from finrag.retrieval import SearchResult
    from finrag.generation import validate_numeric_evidence
    evidence = [SearchResult(chunk=Chunk(source="report.txt", chunk_id=0,
        text="Operating profit increased to €58 million in 2025."), score=0.82)]
    assert validate_numeric_evidence("€58 million [1]", evidence) == ("€58 million [1]", True)
    assert validate_numeric_evidence("The profit increased [1]", evidence) == ("The profit increased [1]", True)
