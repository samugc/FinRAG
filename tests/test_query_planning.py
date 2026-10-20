
import pytest

from finrag.query_planning import plan_queries


def test_simple_question_is_preserved():
    question = "What was revenue in FY2022?"
    assert plan_queries(question) == [question]


def test_quick_ratio_expands_into_balance_sheet_queries():
    queries = plan_queries("What was the quick ratio in FY2022?")

    assert len(queries) == 3
    assert "cash" in queries[1]
    assert "current liabilities" in queries[2]
    assert "FY2022" in queries[1]


def test_ebitda_less_capex_expands_into_components():
    queries = plan_queries("Calculate EBITDA less capex for FY2022")

    assert len(queries) == 4
    assert any("operating income" in q for q in queries)
    assert any("depreciation" in q for q in queries)
    assert any("capital expenditures" in q for q in queries)


def test_empty_question_raises_error():
    with pytest.raises(ValueError):
        plan_queries("   ")
