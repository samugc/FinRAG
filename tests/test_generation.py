import pytest

from finrag.generation import LocalGenerator, validate_answer


@pytest.mark.parametrize("text", ["", "[1]", "[1], [2].", "...",
                                 "INSUFFICIENT_EVIDENCE.", "€58 million [3]"])
def test_rejects_invalid_output(text):
    assert validate_answer(text, 2) == ("INSUFFICIENT_EVIDENCE", False)


def test_valid_output_is_not_proof_of_grounding():
    assert validate_answer("€58 million [1]", 2) == ("€58 million [1]", True)
    assert validate_answer("€58 million", 2) == ("INSUFFICIENT_EVIDENCE", False)
    assert validate_answer("An unsupported claim [1]", 2)[1]


def test_no_evidence_abstains_without_generation():
    generator = LocalGenerator.__new__(LocalGenerator)
    assert generator.answer("Profit?", []) == ("INSUFFICIENT_EVIDENCE", False)


@pytest.mark.parametrize('response_text,expected_valid', [
    ('Cash is 42 [2]', True),
    ('Cash is 42 [3]', False),
])
def test_ollama_payload_and_validation(monkeypatch, response_text, expected_valid):
    import io
    import json

    from finrag.context import format_context
    from finrag.documents import Chunk
    from finrag.retrieval import SearchResult

    hits = [
        SearchResult(Chunk('report:p1', 0, 'Cash: 42'), 0.9),
        SearchResult(Chunk('report:p2', 0, 'Debt: 10'), 0.8),
    ]

    def fake_urlopen(request, timeout):
        assert request.full_url == 'http://127.0.0.1:11434/api/chat'
        assert request.get_method() == 'POST'
        assert timeout == 180
        payload = json.loads(request.data)
        assert payload['model'] == 'qwen3:4b-instruct'
        assert payload['stream'] is False
        assert payload['think'] is False
        assert payload['options'] == {'temperature': 0}
        assert payload['messages'][1]['content'] == (
            f'Evidence:\n{format_context(hits)}\n\nQuestion: Cash?'
        )
        return io.BytesIO(json.dumps({
            'message': {'content': response_text},
        }).encode())

    monkeypatch.setattr('finrag.generation.urlopen', fake_urlopen)
    answer, valid = LocalGenerator().answer('Cash?', hits)
    assert valid is expected_valid
    assert answer == (response_text if expected_valid else 'INSUFFICIENT_EVIDENCE')


@pytest.mark.parametrize('text,status', [
    ('INSUFFICIENT_EVIDENCE because a formula is missing.', 'model_abstention'),
    ('Explanation first. insufficient_evidence [1]', 'model_abstention'),
    ('INSUFFICIENT_EVIDENCE [99]', 'model_abstention'),
    ('42 [1, 2]', 'produced'),
    ('42 [1][2]', 'produced'),
    ('42 [1, 99]', 'invalid_output'),
    ('42 [0]', 'invalid_output'),
    ('42 [3]', 'invalid_output'),
    ('42 [1,]', 'invalid_output'),
    ('42 [1-2]', 'invalid_output'),
    ('42 [one]', 'invalid_output'),
    ('42 [1] [', 'invalid_output'),
    ('42 [[1]]', 'invalid_output'),
    ('42', 'invalid_output'),
    ('[1, 2]', 'invalid_output'),
    ('', 'invalid_output'),
])
def test_structural_classification(text, status):
    from finrag.generation import classify_answer
    result = classify_answer(text, 2)
    assert result.raw_answer == text
    assert result.status.value == status
    assert validate_answer(text, 2)[1] == (status == 'produced')


def test_no_evidence_does_not_call_backend(monkeypatch):
    from finrag.documents import Chunk
    from finrag.generation import AnswerStatus
    from finrag.retrieval import SearchResult

    def unexpected(*args, **kwargs):
        pytest.fail('Backend called without evidence text')
    monkeypatch.setattr('finrag.generation.urlopen', unexpected)
    generator = LocalGenerator()
    for hits in ([], [SearchResult(Chunk('doc', 0, '  '), 1)]):
        assert generator.generate('Question?', hits).status is AnswerStatus.NO_EVIDENCE
        assert generator.answer('Question?', hits) == ('INSUFFICIENT_EVIDENCE', False)


@pytest.mark.parametrize('body', [b'not json', b'{}', b'null', b'{"message":{"content":42}}'])
def test_malformed_backend_response_is_not_missing_evidence(monkeypatch, body):
    import io

    from finrag.documents import Chunk
    from finrag.generation import AnswerStatus
    from finrag.retrieval import SearchResult
    monkeypatch.setattr('finrag.generation.urlopen', lambda *a, **k: io.BytesIO(body))
    result = LocalGenerator().generate('Question?', [SearchResult(Chunk('doc', 0, 'Text'), 1)])
    assert result.status is AnswerStatus.BACKEND_ERROR
    assert result.elapsed_seconds >= 0


@pytest.mark.parametrize('error', [TimeoutError('private details'), OSError('private details')])
def test_backend_failure_is_safe_and_compatible(monkeypatch, error):
    from finrag.documents import Chunk
    from finrag.generation import AnswerStatus
    from finrag.retrieval import SearchResult
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr('finrag.generation.urlopen', fail)
    hits = [SearchResult(Chunk('doc', 0, 'Text'), 1)]
    result = LocalGenerator().generate('Question?', hits)
    assert result.status is AnswerStatus.BACKEND_ERROR
    assert 'private details' not in result.reason
    assert LocalGenerator().answer('Question?', hits) == ('INSUFFICIENT_EVIDENCE', False)


def test_raw_answer_retains_whitespace_and_abstention_reason(monkeypatch):
    import io
    import json

    from finrag.documents import Chunk
    from finrag.generation import AnswerStatus
    from finrag.retrieval import SearchResult
    raw = '  INSUFFICIENT_EVIDENCE because the formula is missing.\n'
    monkeypatch.setattr('finrag.generation.urlopen', lambda *a, **k: io.BytesIO(
        json.dumps({'message': {'content': raw}}).encode()))
    result = LocalGenerator().generate('Question?', [SearchResult(Chunk('doc', 0, 'Text'), 1)])
    assert result.raw_answer == raw
    assert result.status is AnswerStatus.MODEL_ABSTENTION
