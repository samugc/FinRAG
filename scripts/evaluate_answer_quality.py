#!/usr/bin/env python3
"""Minimal offline CLI for finrag.answer-quality evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main(argv: list[str] | None = None) -> int:
    _bootstrap()
    from finrag.answer_evaluation import (
        EvaluationError,
        load_assessments,
        load_gold,
        load_traces,
        evaluate_answer_quality,
        write_report,
    )

    parser = argparse.ArgumentParser(
        description="Evaluate FinRAG answer quality from gold, trace and assessment files."
    )
    parser.add_argument("--gold", required=True, help="Path to gold JSON or JSONL file")
    parser.add_argument("--traces", required=True, help="Path to traces JSON or JSONL file")
    parser.add_argument(
        "--assessments", required=True, help="Path to assessments JSON or JSONL file"
    )
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args(argv)

    try:
        gold = load_gold(args.gold)
        traces = load_traces(args.traces)
        assessments = load_assessments(args.assessments)
        report = evaluate_answer_quality(gold, traces, assessments)
        path = write_report(report, args.out)
    except (EvaluationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    aggregates = report["aggregates"]
    outcomes = aggregates["outcomes"]
    print(
        "Evaluated "
        f"{aggregates and report['input']['gold_questions']} case(s): "
        f"success={outcomes['success']} "
        f"failure={outcomes['failure']} "
        f"unassessed={outcomes['unassessed']} "
        f"invalid={outcomes['invalid']}"
    )
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
chmod +x /Users/samu/Desktop/FinRAG-eval-engine/scripts/evaluate_answer_quality.py