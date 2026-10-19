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
npm install
npm run dev
```

Open **http://127.0.0.1:5173**. Vite proxies `/api/*` to FastAPI at `127.0.0.1:8000` in development. You can also use `npm run build` to produce static files in `web/dist`; deploying this static build is not part of this change.

- `GET /health`: corpus loading and readiness
- `GET /documents`: document names and chunk counts; not full text
- `POST /query`: query, retriever, generator and top-k; answer with ranked evidence

Selecting a document in the left panel now **restricts retrieval to that document** through the API's optional `source` field. The document-specific retriever is fitted on first use and cached separately. Returning to All documents uses the original corpus-wide retriever. Only `tfidf` and `minilm` are implemented. RRF is not available yet. `qwen` may need model downloads and can be slow on first use.

**Important:** Source similarity scores and the `sufficient_evidence` field do not establish factual support. The interface intentionally labels this limitation. The model may still hallucinate; verify all answers against the excerpts.

For browser interaction tests (mocked HTTP transport, real React application):

```bash
cd web
npx playwright install chromium
npm run test:e2e
```

The browser tests validate UI behavior and requests but **do not test actual MiniLM or Qwen inference**. Real-model tests require the installed models and a running local backend.

The UI requires a running local backend; there are no mock answers or fake performance figures. It does not upload files, write to the corpus or call hosted models. Do not publicly expose the API as-is.

## Retrieval comparison

The **Compare** action runs TF-IDF and MiniLM on the same question and document scope, displays both ranked evidence lists, and marks shared document chunks. It uses the real local retrieval API, with extractive generation so the model answers do not obscure the comparison. Shared hits are qualitative only: this is not Recall@k, MRR, or a calibrated model-score comparison. FinanceBench/RRF experiments remain separate.

## Numeric hallucination guard

Qwen is no longer shown retrieval similarity scores in its prompt. Generated numeric claims absent from the retrieved *document text* are rejected as `INSUFFICIENT_EVIDENCE`, catching cases such as the unsupported dividend answer `0.0920`. This is a conservative guard, **not** verified factual grounding: a number may appear in the evidence while being unrelated to the question, and reformatted or calculated figures can be rejected. The extractive baseline remains unchanged.

## Uploading PDF reports

Use **Upload PDF** in the left sidebar to add a text-based PDF (max 25 MB, 500 pages) to the local document corpus. Newly uploaded files appear immediately in the library and can be queried without restarting the backend. Uploads are stored in `FINRAG_CORPUS` (defaults to `data/raw`) and PDF uploads to the default directory are gitignored. Existing files are never overwritten; password-protected and image-only scans are rejected. This is a local-only feature: **do not expose the API publicly**. PDF text extraction is basic (no OCR or layout-aware parsing) and may perform poorly on tables or complex layouts.
