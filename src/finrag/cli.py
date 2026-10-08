import argparse

from finrag.answer import extractive_answer
from finrag.context import format_context
from finrag.documents import chunk_corpus, load_corpus
from finrag.sparse import SparseRetriever


def main():
    parser = argparse.ArgumentParser(
        description="Query the local FinRAG corpus."
    )
    parser.add_argument("question")
    parser.add_argument(
        "--retriever",
        choices=["tfidf", "minilm"],
        default="minilm",
    )
    parser.add_argument(
        "--generator",
        choices=["extractive", "qwen"],
        default="extractive",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Extractive score cutoff; not an answerability test")
    parser.add_argument("--corpus", default="data/raw",
                        help="Directory of TXT, Markdown or text-based PDF files")
    args = parser.parse_args()

    if args.k <= 0:
        parser.error("--k must be positive")
    if not args.question.strip():
        parser.error("question must not be empty")

    try:
        documents = load_corpus(args.corpus)
    except (OSError, UnicodeError) as exc:
        parser.error(str(exc))
    chunks = chunk_corpus(documents, chunk_size=25, overlap=5)

    if not chunks:
        parser.error("corpus contains no extractable text")

    if args.retriever == "tfidf":
        retriever = SparseRetriever()
    else:
        from finrag.dense import DenseRetriever

        retriever = DenseRetriever()

    retriever.fit(chunks)
    results = retriever.search(args.question, k=args.k)

    if args.generator == "qwen":
        from finrag.generation import LocalGenerator

        generator = LocalGenerator()
        answer, sufficient = generator.answer(args.question, results)
    else:
        result = extractive_answer(
            results,
            min_score=args.min_score,
        )
        answer = result.text
        sufficient = result.sufficient_evidence

    print(f"\nQuestion: {args.question}")
    print(f"Retriever: {args.retriever}")
    print(f"Generator: {args.generator}")

    print("\nAnswer\n")
    print(answer)

    if not sufficient:
        print("\nEvidence status: insufficient")

    print("\nEvidence\n")
    print(format_context(results))


if __name__ == "__main__":
    main()
