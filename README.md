# FinRAG

Local question answering over financial documents, with TF-IDF or MiniLM retrieval and an extractive baseline or local Qwen generation.

**The included data is small, synthetic and controlled: 4 documents, 12 chunks, 12 answerable questions and 6 unanswerable questions. It is not a representative financial QA benchmark.**

The Python pipeline implements document loading, word-based chunking with overlap, source metadata, cosine ranking, context construction and retrieval evaluation without a RAG framework. It uses scikit-learn for TF-IDF, Sentence Transformers for pretrained MiniLM embeddings and Transformers for Qwen inference; the models are not trained here.

```text
TXT / Markdown / PDF → 25-word chunks, 5-word overlap
                    → TF-IDF or MiniLM → cosine ranking → top-k chunks
                    → context with sources → extractive answer or local Qwen
```

Sparse retrieval computes cosine similarity over TF-IDF vectors. Dense retrieval uses `sentence-transformers/all-MiniLM-L6-v2`; normalized embeddings make their dot product equal to cosine similarity. Each result retains its filename, chunk id and score. Retrieval, context formatting and generation remain separate modules.

## Evaluation

Retrieval is evaluated independently of generation. Here, Recall@k means the fraction of questions with **at least one accepted evidence chunk** in the top k (also called hit rate). Questions can accept multiple chunk ids where overlap repeats evidence; this is not recall over every relevant chunk.

| Retriever | Questions | Recall@1 | Recall@3 |
| --- | --- | ---: | ---: |
| TF-IDF | All | 0.833 | 1.000 |
| TF-IDF | Lexical | 1.000 | 1.000 |
| TF-IDF | Paraphrase | 0.667 | 1.000 |
| MiniLM | All | 0.750 | 1.000 |
| MiniLM | Lexical | 0.833 | 1.000 |
| MiniLM | Paraphrase | 0.667 | 1.000 |

TF-IDF ranks better on this particular corpus. With only six lexical and six paraphrased questions, this does not establish a general advantage over dense retrieval. The comparison script regenerates `results/retrieval_metrics.json`.

Raw similarity did not reliably determine whether an answer existed in the corpus:

| Retriever | Lowest answerable top-1 score | Highest unanswerable top-1 score |
| --- | ---: | ---: |
| TF-IDF | 0.2107 | 0.3670 |
| MiniLM | 0.4865 | 0.7287 |

The score ranges overlap. Similarity is useful for ranking, but no single score cutoff separates these answerable and unanswerable questions.

## Run

Use Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python scripts/compare_retrievers.py
python scripts/analyze_confidence.py

finrag "What was Northstar's operating profit?" --retriever tfidf --generator extractive
finrag "What was Northstar's operating profit?" --retriever minilm --generator qwen
finrag "What dividend per share did Northstar distribute?" --retriever minilm --generator qwen
```

The default corpus is `data/raw` relative to the current directory. Use `--corpus PATH` from another directory or for your own documents. `--k` defaults to 3. The evaluation scripts always use the included corpus.

The first model run downloads weights from Hugging Face; no paid API or API key is required. After caching the weights, `HF_HUB_OFFLINE=1` disables Hub checks. Qwen uses Apple Silicon MPS when available and otherwise CPU; CPU generation can be slower. Models are loaded once per CLI invocation, and corpus embeddings are computed on each dense retriever fit.

## Generation checks and limitations

`Qwen/Qwen2.5-0.5B-Instruct` receives the retrieved evidence and an instruction to answer from it or return `INSUFFICIENT_EVIDENCE`. In the manual checks on Apple Silicon MPS:

| Question | Qwen output |
| --- | --- |
| What was Northstar's operating profit? | €58 million |
| What dividend per share did Northstar distribute? | INSUFFICIENT_EVIDENCE |

An explicit CPU check returned `€58 million` for profit but **`0.5434` for the unsupported dividend question**, copying a retrieval score. This passed output validation. Greedy decoding did not produce the same answer across these devices; the MPS abstention example is not a portable guarantee.

The CLI prints the source chunks and scores separately. The supported answer omitted the requested inline citation. Validation rejects empty or punctuation/citation-only output, invalid citation indices and explicit abstention. Passing validation does **not** establish factual support.

During development, one prompt returned only `[1]` for the unsupported question; a stricter prompt then rejected a supported question. The simpler prompt is retained. These manual examples do not demonstrate reliable abstention or hallucination control, and generation has not been evaluated systematically.

The extractive baseline returns the highest-ranked chunk, not a synthesized answer. Its optional `--min-score` is only a retrieval cutoff, not an answerability detector; unrelated questions can still receive text.

Word boundaries can split sentences and tables. PDF loading extracts embedded text without OCR, layout reconstruction or page-level citations; scanned PDFs may yield no text, and malformed PDFs can fail parsing. The hand-written relevance labels and tiny corpus limit the evaluation. There is no reranker or persistent index. Real financial reports and independently checked evidence labels would be needed before making broader quality claims.
