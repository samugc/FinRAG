import pytest

from finrag.documents import Chunk
from finrag.evidence import (
    Evidence,
    build_context,
    find_evidence_chunks,
    retrieve_evidence,
    select_page_chunks,
)
from finrag.retrieval import SearchResult


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


def test_context_contains_sources():
    hits = [
        SearchResult(
            Chunk("AMD:p55", 0, "Cash: 4835"),
            0.9,
        ),
        SearchResult(
            Chunk("AMD:p56", 0, "Debt: 2500"),
            0.8,
        ),
    ]

    context = build_context(hits)

    assert "[1] AMD:p55" in context
    assert "[2] AMD:p56" in context
    assert "Cash: 4835" in context


def test_context_removes_duplicates():
    hit = SearchResult(
        Chunk("AMD:p55", 0, "Revenue"),
        0.9,
    )

    context = build_context([hit, hit])

    assert context.count("Revenue") == 1


def test_context_limits_chunks():
    hits = [
        SearchResult(Chunk("doc:p0", i, str(i)), 0.9)
        for i in range(5)
    ]

    context = build_context(hits, max_chunks=2)

    assert "[1]" in context
    assert "[2]" in context
    assert "[3]" not in context


def test_invalid_limit():
    with pytest.raises(ValueError):
        build_context([], max_chunks=0)


def test_select_chunks_respects_page_ranking():
    hits = [
        SearchResult(Chunk("AMD:p5", 0, "Revenue"), 0.7),
        SearchResult(Chunk("AMD:p7", 0, "Cash"), 0.9),
        SearchResult(Chunk("AMD:p7", 1, "Debt"), 0.6),
        SearchResult(Chunk("AMD:p9", 0, "Assets"), 0.95),
    ]

    selected = select_page_chunks(
        ranked_pages=[7, 5, 9],
        hits=hits,
        max_pages=2,
        chunks_per_page=2,
    )

    assert [hit.chunk.text for hit in selected] == [
        "Cash", "Debt", "Revenue"
    ]


def test_retrieve_evidence_combines_query_rankings():
    question = "What was the quick ratio in FY2022?"

    original_hits = [
        SearchResult(Chunk("report:p5", 0, "General information"), 0.9),
        SearchResult(Chunk("report:p7", 0, "Current liabilities"), 0.6),
    ]

    expanded_hits = [
        SearchResult(Chunk("report:p7", 0, "Current liabilities"), 0.9),
        SearchResult(Chunk("report:p5", 0, "General information"), 0.2),
    ]

    class FakeRetriever:
        chunks = [hit.chunk for hit in original_hits]  # noqa: RUF012 - test fixture

        def search(self, query, k=3):
            assert k == 2
            if query == question:
                return original_hits
            return expanded_hits

    selected = retrieve_evidence(
        question,
        FakeRetriever(),
        max_pages=1,
        chunks_per_page=1,
    )

    assert selected[0].chunk.source == "report:p7"

    context = build_context(selected)
    assert "Current liabilities" in context
    assert "[1] report:p7" in context


def test_selects_neighbor_instead_of_distant_chunk():
    hits = [
        SearchResult(Chunk("AMD:p55", 0, "Cash"), 0.9),
        SearchResult(Chunk("AMD:p55", 1, "Current liabilities"), 0.4),
        SearchResult(Chunk("AMD:p55", 2, "Long-term debt"), 0.3),
        SearchResult(Chunk("AMD:p55", 3, "Equity"), 0.8),
    ]

    selected = select_page_chunks(
        ranked_pages=[55],
        hits=hits,
        max_pages=1,
        chunks_per_page=2,
    )

    assert [hit.chunk.chunk_id for hit in selected] == [0, 1]


def test_single_query_preserves_page_order_and_neighbor_ties():
    hits = [
        SearchResult(Chunk('report:p7', 1, 'Center'), 0.9),
        SearchResult(Chunk('report:p5', 0, 'Other page'), 0.9),
        SearchResult(Chunk('report:p7', 2, 'Right'), 0.8),
        SearchResult(Chunk('report:p7', 0, 'Left'), 0.1),
    ]

    class FakeRetriever:
        chunks = [hit.chunk for hit in hits]  # noqa: RUF012 - test fixture

        def search(self, query, k):
            assert query == 'Revenue?'
            assert k == len(hits)
            return hits

    selected = retrieve_evidence('Revenue?', FakeRetriever(), max_pages=1)
    assert [hit.chunk.text for hit in selected] == ['Left', 'Center']


@pytest.mark.parametrize('limits', [
    {'max_pages': 0}, {'chunks_per_page': 0},
])
def test_page_selection_and_retrieval_reject_invalid_limits(limits):
    with pytest.raises(ValueError):
        select_page_chunks([], [], **limits)
    with pytest.raises(ValueError):
        retrieve_evidence('Revenue?', None, **limits)
