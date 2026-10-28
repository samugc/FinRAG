# Local HTTP API and React workspace

From the repository root, install `python -m pip install -e '.[dev]'`, then run
`uvicorn finrag.api:app --host 127.0.0.1 --port 8000` (one worker).
In `web/`, run `npm ci` and `npm run dev`; open http://127.0.0.1:5183.
Run `npm run build` and `npm run test:e2e` for frontend checks (Playwright Chromium required).
Ollama must be running with `qwen3:4b-instruct` for generated answers.

The corpus defaults to `data/raw` (`FINRAG_CORPUS`). Uploaded PDFs go to
`.local/uploads` (`FINRAG_UPLOADS`), separate from historical datasets.
An existing empty corpus is supported. Missing corpus paths return 503.

- `GET /health`: availability and document/chunk counts.
- `GET /documents`: filenames and chunk counts.
- `POST /documents`: multipart `file`, with `X-FinRAG-Upload: 1`. Requires a
  simple PDF filename, selectable text, at most 25 MB and 500 pages. Encrypted
  PDFs and duplicate names are rejected. Indexes refresh after upload.
- `GET /documents/{source}/file`: registered PDFs only; no traversal or symlink
  following, inline content with sandbox and nosniff headers.
- `POST /query`: see `/docs` for the complete typed contract.

Example query body:

```json
{"question":"What was AMD's quick ratio in FY2022?","source":"AMD_2022_10K.pdf","retriever":"minilm","generator":"qwen","retrieval_mode":"multiquery","max_pages":5,"chunks_per_page":2}
```

Direct retrieval retains TF-IDF/MiniLM, `k` and the extractive score cutoff.
Multi-Query requires one selected PDF and reuses the existing page-fusion and
chunk-selection functions. Direct PDF chunks use 25 words/5 overlap per physical
page; Multi-Query uses 80/15. These are API ingestion settings, not changes to
historical evaluation scripts or algorithms.

Responses contain original `answer`, P0 `status`, `reason`, optional backend
`error`, `timings`, `period_check`, and ordered `evidence`. Evidence includes
`citation_id`, document `source`, chunk ID, text, score and nullable zero-based
`page_index` / one-based physical `page_number`. Plain documents have no invented
pages. Additional reporting headers have `kind: reporting_context`. React keeps
these evidence snapshots attached to the original response. `calculation` is
reserved and always null; P1 is not implemented.

P0 states are `produced`, `model_abstention`, `invalid_output`, `no_evidence`,
`backend_error`, plus pipeline `insufficient_context`. A backend generation error
is a 200 response with that state and the retrieved evidence; malformed requests
return 422 and unavailable corpus/retrieval services return 503. For
`backend_error`, `error` is a user-facing Ollama setup hint while `reason`
retains the diagnostic exception type. The legacy `sufficient_evidence` boolean
means produced/structurally accepted, not factual correctness. Extractive
production means the score cutoff passed.

Period checks search complete pages of the selected evidence documents for
explicit fiscal periods or supported English statement dates, not bare numeric
amounts. Located headers can be added to model evidence. Missing requested years
in recognized reporting context yield `insufficient_context`, never a claim that
the PDF lacks the data. Unknown formats warn without blocking; mixed numeric-date
headings also warn. This is deliberately limited recognition, not general table
understanding: abbreviated/localized dates and unusual layouts may remain
unverified. Availability does not verify amount-to-column alignment, fiscal versus
calendar-year equivalence, units, arithmetic or citation entailment.

This is a local, single-process service without authentication, OCR, persistent
indexes or hostile-PDF isolation. Bind to loopback. The custom upload header is
not authentication. Limits apply after multipart parsing; they are not ingress
quotas. Uploaded documents and query evidence stay local; do not publish logs
containing private documents. Retrieval scores across methods are not comparable.
