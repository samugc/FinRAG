#!/usr/bin/env python3
"""Smoke-test a running local FinRAG API with real requests.

Does not test factual grounding or use mock responses.
Usage:
    python scripts/smoke_api.py
    python scripts/smoke_api.py --models
    python scripts/smoke_api.py --models --qwen
"""

import argparse
import json
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


QUESTION = "What was Northstar's operating profit in 2025?"
UNKNOWN = "What dividend per share did Northstar distribute?"


def request_json(base: str, route: str, body: dict | None, timeout: int):
    payload = None if body is None else json.dumps(body).encode("utf-8")
    headers = {} if body is None else {"Content-Type": "application/json"}
    request = Request(base + route, data=payload, headers=headers)
    start = time.monotonic()
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    return result, round(time.monotonic() - start, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--models", action="store_true", help="Also exercise MiniLM")
    parser.add_argument("--qwen", action="store_true", help="Also exercise local Qwen generation")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    base = args.base_url.rstrip("/")
    try:
        health, elapsed = request_json(base, "/health", None, args.timeout)
        print(f"Health: {health} ({elapsed}s)")
        if health.get("status") != "ok":
            print("API did not report healthy", file=sys.stderr)
            return 1
        documents, _ = request_json(base, "/documents", None, args.timeout)
        print("Available documents:", [d["source"] for d in documents["documents"]])
        if not documents.get("documents"):
            print("Corpus has no available documents", file=sys.stderr)
            return 1

        cases = [("tfidf", "extractive")]
        if args.models:
            cases.append(("minilm", "extractive"))
        if args.qwen:
            cases.extend([("tfidf", "qwen"), ("minilm", "qwen")])

        for retriever, generator in cases:
            for question in (QUESTION, UNKNOWN):
                body = {
                    "question": question,
                    "retriever": retriever,
                    "generator": generator,
                    "k": 3,
                }
                result, elapsed = request_json(base, "/query", body, args.timeout)
                evidence = result.get("evidence", [])
                if not isinstance(result.get("answer"), str) or not evidence:
                    print("Missing answer/evidence in API response", file=sys.stderr)
                    return 1
                if any(not {"source", "chunk_id", "text", "score"} <= hit.keys()
                       for hit in evidence):
                    print("Malformed evidence metadata", file=sys.stderr)
                    return 1

                print(f"\n[{retriever} / {generator}] {elapsed}s")
                print("Question:", question)
                print("Answer:", result["answer"])
                print("Evidence status (not verified):", result["sufficient_evidence"])
                print("Top source:", evidence[0]["source"], "chunk", evidence[0]["chunk_id"])
                print("Retrieved chunks:", len(evidence))

        print("\nSmoke checks completed. Review the answers manually.")
        print("A successful request DOES NOT prove factual grounding or correct abstention.")
        return 0
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
        print(f"API smoke test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
