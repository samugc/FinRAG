from finrag.documents import Chunk, Document
from finrag.grounding import check_periods, reporting_periods
from finrag.retrieval import SearchResult

# A small verbatim public balance-sheet excerpt, not a production rule.
HEADER = "Advanced Micro Devices, Inc.\nConsolidated Balance Sheets\nDecember 31,\n2022\nDecember 25,\n2021\n(In millions, except par value amounts)"
VALUES = "Cash and cash equivalents $ 4,835 $ 2,535\nTotal current liabilities 6,369 4,240"


def test_statement_periods_preserve_column_header_order():
    periods = reporting_periods(HEADER + '\n' + VALUES)
    assert [p['year'] for p in periods] == [2022, 2021]
    assert 'December 25,\n2021' in periods[1]['quote']


def test_header_outside_selected_chunk_is_recovered_from_page():
    hit = SearchResult(Chunk('report.pdf:p55', 1, VALUES, 'report.pdf', 55), 0.9)
    pages = {('report.pdf', 55): HEADER + '\n' + VALUES}
    for year in (2022, 2021):
        result = check_periods(f'AMD FY{year} quick ratio?', [hit], pages, [])
        assert result['status'] == 'available'
        assert result['references'][0]['year'] == year
        assert result['references'][0]['page_index'] == 55
        assert str(year) in result['references'][0]['quote']
    missing = check_periods('AMD FY2040 quick ratio?', [hit], pages, [])
    assert missing['status'] == 'insufficient_context'
    assert missing['references'] == []


def test_year_only_as_amount_never_verifies_period():
    text = 'Cash 2022 and liabilities 2040.'
    hit = SearchResult(Chunk('report.txt', 0, text), 0.8)
    result = check_periods('FY2022 revenue?', [hit], {}, [Document('report.txt', text)])
    assert result['status'] == 'unverified'
    assert result['verified_years'] == []
    assert result['warnings']
    page = HEADER.replace('2022', '2020').replace('2021', '2019') + '\n' + text
    hit = SearchResult(Chunk('report.pdf:p0', 0, text, 'report.pdf', 0), 0.8)
    assert check_periods('FY2022 revenue?', [hit], {('report.pdf', 0): page}, [])['status'] == 'insufficient_context'


def test_unknown_format_warns_without_blocking():
    text = 'Consolidated Balance Sheets\n31/12/2022\nCash 42'
    hit = SearchResult(Chunk('report.pdf:p0', 1, 'Cash 42', 'report.pdf', 0), 1)
    result = check_periods('FY2022 cash?', [hit], {('report.pdf', 0): text}, [])
    assert result['status'] == 'unverified'


def test_other_documents_do_not_supply_period():
    hit = SearchResult(Chunk('a.pdf:p0', 0, 'Cash 42', 'a.pdf', 0), 1)
    result = check_periods('FY2022 cash?', [hit], {('a.pdf', 0): 'No header', ('b.pdf', 0): HEADER}, [])
    assert result['status'] == 'unverified'


def test_mixed_recognized_and_unknown_headers_do_not_block():
    hit = SearchResult(Chunk('report.pdf:p0', 0, VALUES, 'report.pdf', 0), 1)
    pages = {('report.pdf', 0): HEADER,
             ('report.pdf', 1): 'Consolidated Balance Sheets\n31/12/2023\nCash 42'}
    assert check_periods('FY2023 cash?', [hit], pages, [])['status'] == 'unverified'
