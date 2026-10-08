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
