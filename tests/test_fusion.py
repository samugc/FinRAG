
import pytest

from finrag.fusion import reciprocal_rank_fusion


def test_rrf_combines_rankings():
    bm25 = [10, 20, 30]
    minilm = [20, 30, 10]

    ranking = reciprocal_rank_fusion([bm25, minilm])

    assert ranking[0] == 20
    assert set(ranking) == {10, 20, 30}


def test_rrf_removes_duplicate_pages():
    ranking = reciprocal_rank_fusion([
        [5, 5, 8],
        [8, 5],
    ])

    assert len(ranking) == 2
    assert set(ranking) == {5, 8}


def test_rrf_rejects_invalid_constant():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1, 2]], constant=0)
