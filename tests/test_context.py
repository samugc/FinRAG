from finrag.context import format_context
from finrag.documents import Chunk
from finrag.evidence import build_context
from finrag.retrieval import SearchResult


def test_format_preserves_order_scores_and_citation_positions():
    hit = SearchResult(Chunk('report:p1', 0, ' Cash: 42 '), 0.123456)
    assert format_context([hit, hit]) == (
        '[1] report:p1 (chunk 0, score=0.1235)\n Cash: 42 \n\n'
        '[2] report:p1 (chunk 0, score=0.1235)\n Cash: 42 '
    )
    assert format_context([]) == ''


def test_build_context_filters_before_limiting_without_mutating_hits():
    blank = SearchResult(Chunk('report:p1', 0, '  '), 1.0)
    first = SearchResult(Chunk('report:p1', 0, ' Cash: 42 '), 0.9)
    duplicate = SearchResult(Chunk('report:p1', 0, 'Different text'), 0.8)
    second = SearchResult(Chunk('report:p2', 0, 'Debt: 10'), 0.7)
    extra = SearchResult(Chunk('report:p3', 0, 'Other'), 0.6)
    assert build_context([blank, first, duplicate, second, extra], 2) == (
        '[1] report:p1 (chunk 0, score=0.9000)\nCash: 42\n\n'
        '[2] report:p2 (chunk 0, score=0.7000)\nDebt: 10'
    )
    assert first.chunk.text == ' Cash: 42 '
    assert build_context([blank]) == ''
