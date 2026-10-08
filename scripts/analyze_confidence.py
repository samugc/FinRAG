from pathlib import Path

import json

from finrag.dense import DenseRetriever
from finrag.documents import chunk_corpus, load_corpus
from finrag.evaluation import load_questions
from finrag.sparse import SparseRetriever


ROOT = Path(__file__).resolve().parents[1]

documents = load_corpus(ROOT / "data/raw")
chunks = chunk_corpus(documents, chunk_size=25, overlap=5)

answerable = load_questions(ROOT / "data/eval/questions.json")

with open(ROOT / "data/eval/unanswerable.json", encoding="utf-8") as f:
    unanswerable = json.load(f)

retrievers = {
    "TF-IDF": SparseRetriever(),
    "MiniLM": DenseRetriever(),
}

for name, retriever in retrievers.items():
    retriever.fit(chunks)

    print(f"\n{name}")
    print("ANSWERABLE")

    answerable_scores = []

    for item in answerable:
        result = retriever.search(item["question"], k=1)[0]
        answerable_scores.append(result.score)
        print(f"{result.score:.4f}  {item['question']}")

    print(
        f"min answerable score: "
        f"{min(answerable_scores):.4f}"
    )

    print("\nUNANSWERABLE")

    unanswerable_scores = []

    for question in unanswerable:
        result = retriever.search(question, k=1)[0]
        unanswerable_scores.append(result.score)
        print(
            f"{result.score:.4f}  {question}\n"
            f"        -> {result.chunk.source} / "
            f"chunk {result.chunk.chunk_id}"
        )

    print(
        f"max unanswerable score: "
        f"{max(unanswerable_scores):.4f}"
    )
