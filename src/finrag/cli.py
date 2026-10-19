import argparse

from finrag.context import format_context
from finrag.pipeline import FinRAGPipeline


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
        result = FinRAGPipeline(args.corpus).query(
            args.question,
            retriever=args.retriever,
            generator=args.generator,
            k=args.k,
            min_score=args.min_score,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))

    print(f"\nQuestion: {result.question}")
    print(f"Retriever: {result.retriever}")
    print(f"Generator: {result.generator}")
    print("\nAnswer\n")
    print(result.answer)

    if not result.sufficient_evidence:
        print("\nEvidence status: insufficient")

    print("\nEvidence\n")
    print(format_context(result.evidence))


if __name__ == "__main__":
    main()
