import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

from finrag.context import format_context
from finrag.documents import Chunk, Page
from finrag.retrieval import SearchResult


def test_demo_preview_matches_generated_context(monkeypatch, capsys, tmp_path):
    root = Path(__file__).resolve().parents[1]
    demo = runpy.run_path(str(root / 'scripts/demo_evidence.py'))
    main = demo['main']
    namespace = main.__globals__
    hits = [SearchResult(Chunk('AMD:p0', 0, 'Cash: 42'), 0.9)]

    class FakeRetriever:
        def fit(self, chunks):
            assert chunks == [Chunk('AMD:p0', 0, 'Cash: 42')]

    class FakeGenerator:
        def __init__(self, model_name):
            assert model_name == 'qwen3:4b-instruct'
            self.model_name = model_name

        def generate(self, question, results):
            assert results is hits
            assert question == "What was AMD's quick ratio in FY2022?"
            from finrag.generation import classify_answer
            return classify_answer('42 [1]', 1)

    def fake_retrieve(question, retriever, max_pages, chunks_per_page):
        assert (max_pages, chunks_per_page) == (5, 2)
        return hits

    monkeypatch.setitem(sys.modules, 'finrag.dense', SimpleNamespace(DenseRetriever=FakeRetriever))
    monkeypatch.setitem(namespace, 'load_pdf_pages', lambda path: [Page('AMD', 0, 'Cash: 42')])
    monkeypatch.setitem(namespace, 'retrieve_evidence', fake_retrieve)
    monkeypatch.setitem(namespace, 'LocalGenerator', FakeGenerator)
    trace = tmp_path / 'trace.json'
    main(['--trace', str(trace)])
    import json
    record = json.loads(trace.read_text())
    assert record['context'] == format_context(hits)
    assert record['generation']['raw_answer'] == '42 [1]'
    assert record['generation']['status'] == 'produced'
    assert record['sources'][0]['chunk']['source'] == 'AMD:p0'
    assert record['configuration']['model'] == 'qwen3:4b-instruct'
    assert record['timings']['total_seconds'] >= 0
    import pytest
    before = trace.read_bytes()
    with pytest.raises(SystemExit):
        main(['--trace', str(trace)])
    assert trace.read_bytes() == before
    output = capsys.readouterr().out
    assert format_context(hits) in output
    assert 'Generation status: produced' in output


def test_cli_tfidf_extractive(tmp_path, monkeypatch, capsys):
    from finrag.cli import main

    (tmp_path / 'report.txt').write_text('Revenue was 42 million.', encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', [
        'finrag', 'Revenue?', '--retriever', 'tfidf',
        '--corpus', str(tmp_path), '--k', '1',
    ])
    main()
    output = capsys.readouterr().out
    assert 'Revenue was 42 million.' in output
    assert '[1] report.txt (chunk 0, score=' in output


def test_cli_reports_model_abstention_without_claiming_missing_evidence(tmp_path, monkeypatch, capsys):
    from finrag.cli import main
    from finrag.generation import LocalGenerator, classify_answer
    (tmp_path / 'report.txt').write_text('Revenue was 42 million.', encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', [
        'finrag', 'Revenue?', '--retriever', 'tfidf', '--generator', 'qwen',
        '--corpus', str(tmp_path),
    ])
    monkeypatch.setattr(LocalGenerator, 'generate', lambda *a, **k:
        classify_answer('INSUFFICIENT_EVIDENCE because the formula is missing.', 1))
    main()
    output = capsys.readouterr().out
    assert 'Generation status: model_abstention' in output
    assert 'Evidence status: insufficient' not in output
