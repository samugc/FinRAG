
import pytest

from finrag.documents import Chunk
from finrag.evidence import Evidence, find_evidence_chunks


def test_finds_evidence_in_correct_document():
    chunks = [
        Chunk("report_A.txt", 0, "Revenue was €420 million."),
        Chunk("report_B.txt", 0, "Profit was €58 million."),
        Chunk("report_A.txt", 1, "Profit was €58 million."),
    ]

    evidence = Evidence(
        source="report_A.txt",
        text="Profit was €58 million.",
    )

    assert find_evidence_chunks(evidence, chunks) == [1]


def test_finds_multiple_matching_chunks():
    chunks = [
        Chunk("report.txt", 0, "Revenue increased by 12%."),
        Chunk("report.txt", 1, "Revenue increased by 12% in 2025."),
    ]

    evidence = Evidence("report.txt", "Revenue increased by 12%")

    assert find_evidence_chunks(evidence, chunks) == [0, 1]


def test_returns_empty_list_when_evidence_is_missing():
    chunks = [
        Chunk("report.txt", 0, "Revenue was €420 million."),
    ]

    evidence = Evidence("report.txt", "Profit was €58 million.")

    assert find_evidence_chunks(evidence, chunks) == []


@pytest.mark.parametrize(
    "source,text",
    [
        ("", "Revenue increased"),
        ("   ", "Revenue increased"),
        ("report.txt", ""),
        ("report.txt", "   "),
    ],
)
def test_rejects_empty_evidence(source, text):
    evidence = Evidence(source, text)

    with pytest.raises(ValueError):
        find_evidence_chunks(evidence, [])
