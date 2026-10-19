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

The UI requires a running local backend; there are no mock answers or fake performance figures. It does not upload files, write to the corpus or call hosted models. Do not publicly expose the API as-is.
