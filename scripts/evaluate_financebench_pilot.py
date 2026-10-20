
import argparse
import json
from pathlib import Path

from sentence_transformers import CrossEncoder

from finrag.bm25 import BM25Retriever
from finrag.dense import DenseRetriever
from finrag.documents import Document, chunk_document, load_pdf_pages
from finrag.evaluation import (
    all_evidence_rank,
    first_evidence_rank,
    rank_pages,
)
from finrag.fusion import reciprocal_rank_fusion
from finrag.query_planning import plan_queries
from finrag.rerank import rerank_pages
from finrag.sparse import SparseRetriever

# Configuration

parser = argparse.ArgumentParser(
    description="Evaluate FinRAG retrieval on FinanceBench."
)

parser.add_argument(
    "--split",
    choices=["pilot", "holdout"],
    default="pilot",
    help="Evaluation dataset",
)

args = parser.parse_args()

base = Path("data/external/financebench")

manifest_path = Path(
    f"data/eval/financebench_{args.split}.json"
)

pilot = json.loads(manifest_path.read_text())

questions = {
    q["financebench_id"]: q
    for line in (
        base / "financebench_open_source.jsonl"
    ).read_text().splitlines()
    if line.strip()
    for q in [json.loads(line)]
}


# Initialize retrieval models

retrievers = {
    "tfidf": SparseRetriever(),
    "bm25": BM25Retriever(),
    "minilm": DenseRetriever(),
}

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

results = []


# Evaluate each document

for doc_name in pilot["documents"]:

    pages = load_pdf_pages(
        base / f"{doc_name}.pdf"
    )

    chunks = []

    for page in pages:

        document = Document(
            source=f"{doc_name}:p{page.page_index}",
            text=page.text,
        )

        chunks.extend(
            chunk_document(
                document,
                chunk_size=80,
                overlap=15,
            )
        )

    items = [
        q for q in pilot["questions"]
        if q["doc_name"] == doc_name
    ]

    rankings_by_question = {}
    minilm_hits = {}

    # 1. Sparse and dense retrieval

    for method, retriever in retrievers.items():

        retriever.fit(chunks)

        for item in items:

            question_id = item["financebench_id"]
            query = questions[question_id]["question"]

            hits = retriever.search(
                query,
                k=len(chunks),
            )

            if method == "minilm":
                minilm_hits[question_id] = hits

            ranked_pages = rank_pages(hits)

            if method in ("bm25", "minilm"):
                rankings_by_question.setdefault(
                    question_id, {}
                )[method] = ranked_pages

            results.append({
                "question_id": question_id,
                "document": doc_name,
                "question_type": item["question_type"],
                "method": method,
                "rank": first_evidence_rank(
                    ranked_pages,
                    item["gold_pages"],
                ),
                "all_evidence_rank": all_evidence_rank(
                    ranked_pages,
                    item["gold_pages"],
                ),
            })

    # 2. Hybrid retrieval: BM25 + MiniLM

    for item in items:

        question_id = item["financebench_id"]
        rankings = rankings_by_question[question_id]

        fused_pages = reciprocal_rank_fusion([
            rankings["bm25"],
            rankings["minilm"],
        ])

        results.append({
            "question_id": question_id,
            "document": doc_name,
            "question_type": item["question_type"],
            "method": "hybrid_rrf",
            "rank": first_evidence_rank(
                fused_pages,
                item["gold_pages"],
            ),
            "all_evidence_rank": all_evidence_rank(
                fused_pages,
                item["gold_pages"],
            ),
        })

    # 3. Multi-Query Financial Retrieval

    for item in items:

        question_id = item["financebench_id"]
        query = questions[question_id]["question"]

        planned_queries = plan_queries(query)

        rankings = [
            rankings_by_question[question_id]["minilm"]
        ]

        for subquery in planned_queries[1:]:

            hits = retrievers["minilm"].search(
                subquery,
                k=len(chunks),
            )

            rankings.append(
                rank_pages(hits)
            )

        multiquery_pages = reciprocal_rank_fusion(
            rankings
        )

        results.append({
            "question_id": question_id,
            "document": doc_name,
            "question_type": item["question_type"],
            "method": "minilm_multiquery",
            "rank": first_evidence_rank(
                multiquery_pages,
                item["gold_pages"],
            ),
            "all_evidence_rank": all_evidence_rank(
                multiquery_pages,
                item["gold_pages"],
            ),
        })

    # 4. Cross-Encoder reranking

    for item in items:

        question_id = item["financebench_id"]
        query = questions[question_id]["question"]

        for n_chunks, method_name in [
            (1, "minilm_reranked"),
            (3, "minilm_reranked_3chunks"),
        ]:

            reranked_pages = rerank_pages(
                query,
                minilm_hits[question_id],
                cross_encoder,
                top_n=20,
                chunks_per_page=n_chunks,
            )

            results.append({
                "question_id": question_id,
                "document": doc_name,
                "question_type": item["question_type"],
                "method": method_name,
                "rank": first_evidence_rank(
                    reranked_pages,
                    item["gold_pages"],
                ),
                "all_evidence_rank": all_evidence_rank(
                    reranked_pages,
                    item["gold_pages"],
                ),
            })

    print(
        f"Evaluado: {doc_name} "
        f"({len(items)} preguntas, {len(chunks)} chunks)"
    )


# Evaluation metrics

def hit_rates(rows, rank_key="rank"):

    return {
        f"hit_at_{k}": sum(
            r[rank_key] is not None
            and r[rank_key] <= k
            for r in rows
        ) / len(rows)
        for k in (1, 3, 5)
    }


methods = [
    *retrievers,
    "hybrid_rrf",
    "minilm_reranked",
    "minilm_reranked_3chunks",
    "minilm_multiquery",
]

summary = {}

for method in methods:

    rows = [
        r for r in results
        if r["method"] == method
    ]

    summary[method] = {
        "overall": hit_rates(rows),

        "by_document": {
            doc: hit_rates([
                r for r in rows
                if r["document"] == doc
            ])
            for doc in pilot["documents"]
        },

        "all_evidence_overall": hit_rates(
            rows,
            "all_evidence_rank",
        ),

        "all_evidence_by_document": {
            doc: hit_rates(
                [
                    r for r in rows
                    if r["document"] == doc
                ],
                "all_evidence_rank",
            )
            for doc in pilot["documents"]
        },
    }


# Save results

output = Path(
    f"results/financebench_{args.split}.json"
)

output.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output.write_text(
    json.dumps({
        "metadata": {
            "split": args.split,
            "documents": pilot["documents"],
            "questions": len(pilot["questions"]),
            "chunk_size": 80,
            "overlap": 15,
            "evaluation": "within-document page Hit@k",
            "rrf_constant": 60,
            "rerank_top_n": 20,
            "rerank_chunks_per_page": [1, 3],
            "query_planning": "deterministic financial rules",
        },
        "results": results,
        "summary": summary,
    }, indent=2) + "\n"
)


# Print summary

print(f"\n=== FinanceBench: {args.split.upper()} ===")

print(
    f"Documentos: {len(pilot['documents'])} | "
    f"Preguntas: {len(pilot['questions'])}"
)

print("\n=== Retrieval Results ===")

for method, metrics in summary.items():

    any_hit = metrics["overall"]["hit_at_5"]
    all_hit = metrics["all_evidence_overall"]["hit_at_5"]

    print(
        f"{method:<27} "
        f"Any@5={any_hit:.1%}  "
        f"All@5={all_hit:.1%}"
    )

print(f"\nGuardado en {output}")
