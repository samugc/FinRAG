from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from finrag.api import (
    BACKEND_ERROR_MESSAGE,
    BACKEND_RESPONSE_ERROR_MESSAGE,
    backend_error_message,
    create_app,
)
from finrag.generation import AnswerStatus, GenerationResult
from finrag.pipeline import FinRAGPipeline


def pdf_bytes(pages):
    writer = PdfWriter()
    font = writer._add_object(DictionaryObject({NameObject('/Type'): NameObject('/Font'),
        NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')}))
    for text in pages:
        page = writer.add_blank_page(width=500, height=700)
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'):
            DictionaryObject({NameObject('/F1'): font})})
        stream = DecodedStreamObject()
        lines = ['BT /F1 10 Tf 20 650 Td']
        for line in text.splitlines():
            safe = line.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
            lines.append(f'({safe}) Tj 0 -14 Td')
        lines.append('ET')
        stream.set_data('\n'.join(lines).encode('ascii'))
        page[NameObject('/Contents')] = writer._add_object(stream)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


BALANCE = ('Consolidated Balance Sheets\nDecember 31, 2022 December 25, 2021\n'
           '(In millions)\nCash and cash equivalents 4,835 2,535\n'
           'Short-term investments 1,020 1,073\nAccounts receivable 4,126 2,706\n'
           'Receivables from related parties 2 2\nTotal current liabilities 6,369 4,240')


@pytest.mark.parametrize('status', list(AnswerStatus))
def test_api_preserves_generation_states(tmp_path, status):
    (tmp_path / 'report.txt').write_text('Revenue was 42 million.')
    pipeline = FinRAGPipeline(tmp_path)
    class Generator:
        def generate(self, question, evidence):
            return GenerationResult('raw [1]', status, 'reason', 0.25)
    pipeline._generator = Generator()
    client = TestClient(create_app(lambda: pipeline, corpus_dir=tmp_path))
    data = client.post('/query', json={'question': 'Revenue?', 'generator': 'qwen'}).json()
    assert data['status'] == status.value
    assert data['answer'] == 'raw [1]'
    assert data['sufficient_evidence'] == (status is AnswerStatus.PRODUCED)
    assert data['calculation'] is None
    expected_error = (
        BACKEND_RESPONSE_ERROR_MESSAGE
        if status is AnswerStatus.BACKEND_ERROR
        else None
    )
    assert data['error'] == expected_error
    if status is AnswerStatus.BACKEND_ERROR:
        assert 'unavailable' not in data['error']
        assert data['reason'] == 'reason'
    assert data['evidence'][0]['page_number'] is None
    assert data['evidence'][0]['citation_id'] == 1


@pytest.mark.parametrize('reason,expected', [
    ('URLError', BACKEND_ERROR_MESSAGE),
    ('TimeoutError', BACKEND_ERROR_MESSAGE),
    ('HTTPError', BACKEND_ERROR_MESSAGE),
    ('JSONDecodeError', BACKEND_RESPONSE_ERROR_MESSAGE),
    ('KeyError', BACKEND_RESPONSE_ERROR_MESSAGE),
    ('TypeError', BACKEND_RESPONSE_ERROR_MESSAGE),
])
def test_backend_error_guidance_matches_failure_kind(reason, expected):
    assert backend_error_message(reason) == expected
    if expected is BACKEND_RESPONSE_ERROR_MESSAGE:
        assert 'unavailable' not in expected


def test_pdf_pages_multiquery_and_temporal_context(tmp_path):
    (tmp_path / 'report.pdf').write_bytes(pdf_bytes(['Intro', BALANCE]))
    pipeline = FinRAGPipeline(tmp_path)
    calls = []
    class Generator:
        def generate(self, question, evidence):
            calls.append(evidence)
            return GenerationResult('Raw answer [1]', AnswerStatus.PRODUCED, 'Structural only')
    pipeline._generator = Generator()
    client = TestClient(create_app(lambda: pipeline, corpus_dir=tmp_path))
    for year in (2022, 2021):
        data = client.post('/query', json={'question': f'AMD quick ratio FY{year}?',
            'source': 'report.pdf', 'retrieval_mode': 'multiquery', 'generator': 'qwen'}).json()
        assert data['status'] == 'produced'
        assert data['period_check']['status'] == 'available'
        assert data['period_check']['references'][0]['year'] == year
        assert any(hit['page_number'] == 2 for hit in data['evidence'])
        assert [h['text'] for h in data['evidence']] == [h.chunk.text for h in calls[-1]]
    missing = client.post('/query', json={'question': 'AMD quick ratio FY2040?',
        'source': 'report.pdf', 'retrieval_mode': 'multiquery', 'generator': 'qwen'}).json()
    assert missing['status'] == 'insufficient_context'
    assert len(calls) == 2
    assert missing['evidence']
    assert client.post('/query', json={'question': 'Revenue?', 'retrieval_mode': 'multiquery'}).status_code == 422


def test_upload_into_empty_library_and_open_exact_original(tmp_path):
    client = TestClient(create_app(corpus_dir=tmp_path))
    assert client.get('/health').json()['documents'] == 0
    payload = pdf_bytes([BALANCE])
    response = client.post('/documents', headers={'X-FinRAG-Upload': '1'},
                           files={'file': ('report.pdf', payload, 'application/pdf')})
    assert response.status_code == 201
    assert client.get('/documents/report.pdf/file').content == payload
    assert client.post('/query', json={'question': 'Cash?', 'source': 'report.pdf'}).json()['evidence'][0]['page_number'] == 1


def test_symlink_download_and_upload_are_rejected(tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    secret = outside / 'secret.pdf'
    secret.write_bytes(pdf_bytes([BALANCE]))
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    (corpus / 'link.pdf').symlink_to(secret)
    client = TestClient(create_app(corpus_dir=corpus))
    assert client.get('/documents/link.pdf/file').status_code == 404
    res = client.post('/documents', headers={'X-FinRAG-Upload': '1'},
                      files={'file': ('link.pdf', secret.read_bytes(), 'application/pdf')})
    assert res.status_code == 409


def test_replaced_registered_file_symlink_is_not_served(tmp_path):
    (tmp_path / 'report.pdf').write_bytes(pdf_bytes([BALANCE]))
    client = TestClient(create_app(corpus_dir=tmp_path))
    client.get('/documents')
    (tmp_path / 'report.pdf').unlink()
    (tmp_path / 'report.pdf').symlink_to(tmp_path / 'secret.pdf')
    (tmp_path / 'secret.pdf').write_bytes(pdf_bytes(['private']))
    assert client.get('/documents/report.pdf/file').status_code == 404
