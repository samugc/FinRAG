# HTTP API

The API wraps the same retrieval and generation components used by the CLI.
It does not change chunking, ranking, or generation evaluation.

From the repository root:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
uvicorn finrag.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the generated OpenAPI interface.

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What was Northstar operating profit?","retriever":"tfidf","generator":"extractive","k":3}'
```

The `/health` response reports document and chunk counts. It returns HTTP 503
if the configured corpus cannot be loaded. `/query` returns the answer,
an evidence-status flag, and ranked source chunks with their scores. Pydantic
rejects malformed query inputs with HTTP 422.

The default corpus is `data/raw`. Set `FINRAG_CORPUS` to an alternative path
before starting the server. Run from the repository root when using relative paths.

The TF-IDF retriever is the API default. MiniLM and local Qwen are optional
request selections. Retrievers and the generator are lazy-loaded and cached
within the process to avoid repeated fitting. The first MiniLM or Qwen query
may download model files and use substantial time and memory. The API does
not currently implement parallel GPU inference, persistent indexes, or
authentication: it is intended for local development. Do not expose it publicly.

**Important:** The extractive response is the highest-ranked chunk, not a
verified natural-language answer. The sufficient-evidence flag is based
on the existing cutoff or model output validation; it does not guarantee
factual support. Source scores from different retrieval methods are not
necessarily comparable. FinanceBench/RRF research remains separate from
this API branch.
