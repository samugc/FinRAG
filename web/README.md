# FinRAG web workspace

A responsive React/Vite interface for the local FastAPI service, built around financial questions, source passages and retrieval metadata. The frontend is intentionally separate from the FinanceBench/RRF experiments.

## Run locally

Use **two terminals** from the project root.

Backend:

```bash
python -m pip install -e ".[dev]"
uvicorn finrag.api:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd web
npm ci
npm run dev
```

Open **http://127.0.0.1:5183**. Vite proxies `/api/*` to FastAPI at `127.0.0.1:8000` in development. You can also use `npm run build` to produce static files in `web/dist`; deploying this static build is not part of this change.

- `GET /health`: corpus loading and readiness
- `GET /documents`: document names and chunk counts; not full text
- `POST /query`: query, retriever, generator and top-k; answer with ranked evidence

Selecting a document in the left panel now **restricts retrieval to that document** through the API's optional `source` field. The document-specific retriever is fitted on first use and cached separately. Returning to All documents uses the original corpus-wide retriever. TF-IDF and MiniLM support direct retrieval; Multi-Query additionally uses the existing page-fusion pipeline for a selected PDF. `qwen` may need model downloads and can be slow on first use.

**Important:** Source similarity scores and the `sufficient_evidence` field do not establish factual support. The interface intentionally labels this limitation. The model may still hallucinate; verify all answers against the excerpts.

For browser interaction tests (mocked HTTP transport, real React application):

```bash
cd web
npx playwright install chromium
npm run test:e2e
```

The browser tests validate UI behavior and requests but **do not test actual MiniLM or Qwen inference**. Real-model tests require the installed models and a running local backend.

The UI requires a running local backend; there are no mock answers or fake performance figures. PDF uploads are stored locally; no hosted generation service is used. Do not publicly expose the API as-is.

## Retrieval comparison

The **Compare** action runs TF-IDF and MiniLM on the same question and document scope, displays both ranked evidence lists, and marks shared document chunks. It uses the real local retrieval API, with extractive generation so the model answers do not obscure the comparison. Shared hits are qualitative only: this is not Recall@k, MRR, or a calibrated model-score comparison. FinanceBench/RRF experiments remain separate.

## Response states and limitations

P0 distinguishes produced output, model abstention, invalid output, no evidence,
backend error and pipeline insufficient_context. Citations are structural checks,
not factual validation. There is no numeric-claim guard or deterministic calculator.
Each answer retains its own evidence snapshot and citation targets. Period checks
verify availability only; they cannot assign each amount to a year column.
See [API contract](../docs/api.md) for details and temporal-format limitations.
Run `npm test` for state/citation helper tests; browser tests cover interactions.

## Uploading PDF reports

Use **Upload PDF** in the left sidebar to add a text-based PDF (max 25 MB, 500 pages) to the local document corpus. Newly uploaded files appear immediately in the library and can be queried without restarting the backend. Uploads are stored in `FINRAG_UPLOADS` (defaults to `.local/uploads`, gitignored), separately from the corpus and historical datasets. Existing files are never overwritten; password-protected and image-only scans are rejected. This is a local-only feature: **do not expose the API publicly**. PDF text extraction is basic (no OCR or layout-aware parsing) and may perform poorly on tables or complex layouts.
