import json
from pathlib import Path

from finrag.dense import DenseRetriever
from finrag.documents import chunk_corpus, load_corpus
from finrag.evaluation import load_questions, recall_at_k
from finrag.sparse import SparseRetriever

ROOT = Path(__file__).resolve().parents[1]

docs = load_corpus(ROOT / "data/raw")
chunks = chunk_corpus(docs, chunk_size=25, overlap=5)
questions = load_questions(ROOT / "data/eval/questions.json")

groups = {
    "all": questions,
    "lexical": [q for q in questions if q["type"] == "lexical"],
    "paraphrase": [q for q in questions if q["type"] == "paraphrase"],
}

retrievers = {
    "TF-IDF": SparseRetriever(),
    "MiniLM": DenseRetriever(),
}

metrics = {"corpus": {
    "documents": len(docs), "chunks": len(chunks), "questions": len(questions),
    "chunk_size": 25, "overlap": 5,
    "note": "Small controlled synthetic corpus; not a representative financial benchmark.",
}}

for name, retriever in retrievers.items():
    scores = {}
    retriever.fit(chunks)

    print(f"\n{name}")

    for group_name, group in groups.items():
        r1 = recall_at_k(retriever, group, 1)
        r3 = recall_at_k(retriever, group, 3)

        scores[group_name] = {"recall_at_1": round(r1, 3), "recall_at_3": round(r3, 3)}
        print(
            f"{group_name:10} "
            f"Recall@1={r1:.3f} "
            f"Recall@3={r3:.3f}"
        )

    metrics[{"TF-IDF": "tfidf", "MiniLM": "minilm"}[name]] = scores

output = ROOT / "results/retrieval_metrics.json"
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
