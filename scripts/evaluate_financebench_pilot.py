
import json
from pathlib import Path

from finrag.documents import Document, chunk_document, load_pdf_pages
from finrag.sparse import SparseRetriever
from finrag.dense import DenseRetriever
from finrag.evaluation import rank_pages, first_evidence_rank, all_evidence_rank

base = Path("data/external/financebench")
pilot = json.loads(Path("data/eval/financebench_pilot.json").read_text())

questions = {
    q["financebench_id"]: q
    for line in (base / "financebench_open_source.jsonl").read_text().splitlines()
    if line.strip()
    for q in [json.loads(line)]
}

retrievers = {
    "tfidf": SparseRetriever(),
    "minilm": DenseRetriever(),
}

results = []

for doc_name in pilot["documents"]:
    pages = load_pdf_pages(base / f"{doc_name}.pdf")

    chunks = []
    for page in pages:
        document = Document(
            source=f"{doc_name}:p{page.page_index}",
            text=page.text,
        )
        chunks.extend(chunk_document(document, chunk_size=80, overlap=15))

    items = [q for q in pilot["questions"] if q["doc_name"] == doc_name]

    for method, retriever in retrievers.items():
        retriever.fit(chunks)

        for item in items:
            question_id = item["financebench_id"]
            query = questions[question_id]["question"]

            hits = retriever.search(query, k=len(chunks))
            ranked_pages = rank_pages(hits)
            rank = first_evidence_rank(ranked_pages, item["gold_pages"])
            all_rank = all_evidence_rank(ranked_pages, item["gold_pages"])

            results.append({
                "question_id": question_id,
                "document": doc_name,
                "question_type": item["question_type"],
                "method": method,
                "rank": rank,
                "all_evidence_rank": all_rank,
            })

    print(f"Evaluado: {doc_name} ({len(items)} preguntas, {len(chunks)} chunks)")


def hit_rates(rows, rank_key="rank"):
    return {
        f"hit_at_{k}": sum(
            r[rank_key] is not None and r[rank_key] <= k
            for r in rows
        ) / len(rows)
        for k in (1, 3, 5)
    }


summary = {}
for method in retrievers:
    rows = [r for r in results if r["method"] == method]
    summary[method] = {
        "overall": hit_rates(rows),
        "by_document": {
            doc: hit_rates([r for r in rows if r["document"] == doc])
            for doc in pilot["documents"]
        },
        "all_evidence_overall": hit_rates(rows, "all_evidence_rank"),
        "all_evidence_by_document": {
            doc: hit_rates(
                [r for r in rows if r["document"] == doc],
                "all_evidence_rank",
            )
            for doc in pilot["documents"]
        },
    }

output = Path("results/financebench_pilot.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({
    "metadata": {
        "documents": pilot["documents"],
        "questions": len(pilot["questions"]),
        "chunk_size": 80,
        "overlap": 15,
        "evaluation": "within-document page Hit@k",
    },
    "results": results,
    "summary": summary,
}, indent=2) + "\n")

print(json.dumps({m: s["overall"] for m, s in summary.items()}, indent=2))
print(f"Guardado en {output}")
