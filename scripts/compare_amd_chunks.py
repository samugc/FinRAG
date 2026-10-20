import json
from pathlib import Path

from finrag.dense import DenseRetriever
from finrag.documents import Document, chunk_document, load_pdf_pages
from finrag.evaluation import first_evidence_rank, rank_pages
from finrag.sparse import SparseRetriever

data_dir = Path("data/external/financebench")

questions = [
    json.loads(line)
    for line in (data_dir / "financebench_open_source.jsonl").read_text().splitlines()
    if line.strip()
]
questions = [q for q in questions if q["doc_name"] == "AMD_2022_10K"]

pages = load_pdf_pages(data_dir / "AMD_2022_10K.pdf")

chunks = []
for page in pages:
    document = Document(
        source=f"{page.doc_name}:p{page.page_index}",
        text=page.text,
    )
    chunks.extend(chunk_document(document, chunk_size=80, overlap=15))

results = {}

for name, retriever in [
    ("tfidf", SparseRetriever()),
    ("minilm", DenseRetriever()),
]:
    retriever.fit(chunks)
    positions = []

    print(f"\n--- {name} ---")

    for item in questions:
        hits = retriever.search(item["question"], k=len(chunks))

        ranked_pages = rank_pages(hits)

        gold = {
            e["evidence_page_num"] for e in item["evidence"]
        }

        position = first_evidence_rank(ranked_pages, list(gold))
        positions.append(position)

        print(f"Página {sorted(gold)} -> posición {position}")

    results[name] = {
        "positions": positions,
        **{
            f"hit_at_{k}": sum(
                p is not None and p <= k for p in positions
            ) / len(positions)
            for k in [1, 3, 5]
        },
    }

results["metadata"] = {
    "dataset": "FinanceBench",
    "document": "AMD_2022_10K",
    "questions": len(questions),
    "question_ids": [q["financebench_id"] for q in questions],
    "gold_pages": [
        sorted({e["evidence_page_num"] for e in q["evidence"]})
        for q in questions
    ],
    "chunk_size": 80,
    "overlap": 15,
    "chunks": len(chunks),
    "aggregation": "max chunk similarity per page",
}

output = Path("results/amd_chunks_comparison.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(results, indent=2) + "\n")

print("\n", json.dumps(results, indent=2))
