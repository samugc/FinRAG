import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from finrag.context import format_context
from finrag.documents import Document, chunk_corpus, load_pdf_pages
from finrag.evidence import retrieve_evidence
from finrag.generation import SYSTEM_PROMPT, LocalGenerator


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the public FinanceBench evidence demo")
    parser.add_argument("--trace", type=Path, help="New local JSON file; includes public evidence and raw output")
    args = parser.parse_args(argv)
    if args.trace and args.trace.exists():
        parser.error("trace file already exists; choose a new path")
    from finrag.dense import DenseRetriever

    started = perf_counter()

    pdf = Path(__file__).resolve().parents[1] / "data/external/financebench/AMD_2022_10K.pdf"

    question = "What was AMD's quick ratio in FY2022?"

    pages = load_pdf_pages(pdf)

    documents = [
        Document(source=f"{page.doc_name}:p{page.page_index}", text=page.text)
        for page in pages
    ]
    chunks = chunk_corpus(documents, chunk_size=80, overlap=15)

    print(f"PDF: {len(pages)} pages, {len(chunks)} chunks")

    retriever = DenseRetriever()
    retriever.fit(chunks)

    hits = retrieve_evidence(
        question,
        retriever,
        max_pages=5,
        chunks_per_page=2,
    )

    context = format_context(hits)

    print("\n=== RETRIEVED EVIDENCE ===")

    for hit in hits:
        print(
            f"\n{hit.chunk.source} "
            f"| chunk={hit.chunk.chunk_id} "
            f"| similarity={hit.score:.3f}"
        )
        print(hit.chunk.text[:200])

    print("\n=== LLM CONTEXT PREVIEW ===")
    print(context[:1500])

    retrieval_seconds = perf_counter() - started

    print("\n=== FINRAG ANSWER ===")

    generator = LocalGenerator(model_name="qwen3:4b-instruct")

    result = generator.generate(question=question, results=hits)

    print(result.raw_answer or result.status.value)
    print(f"\nGeneration status: {result.status.value}")

    if args.trace:
        trace = {
            "created_at": datetime.now(UTC).isoformat(),
            "question": question,
            "configuration": {
                "model": generator.model_name,
                "endpoint": "http://127.0.0.1:11434/api/chat",
                "system_prompt": SYSTEM_PROMPT,
                "stream": False, "think": False,
                "options": {"temperature": 0}, "timeout_seconds": 180,
                "retriever": "sentence-transformers/all-MiniLM-L6-v2",
                "chunk_size": 80, "overlap": 15,
                "max_pages": 5, "chunks_per_page": 2,
            },
            "sources": [asdict(hit) for hit in hits],
            "context": context,
            "generation": asdict(result),
            "timings": {
                "preparation_and_retrieval_seconds": retrieval_seconds,
                "generation_seconds": result.elapsed_seconds,
                "total_seconds": perf_counter() - started,
            },
        }
        with args.trace.open("x", encoding="utf-8") as file:
            json.dump(trace, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print("Trace saved")


if __name__ == "__main__":
    main()
