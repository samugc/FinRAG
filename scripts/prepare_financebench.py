"""Create a reproducible FinanceBench development manifest from official JSONL."""

import argparse
import json
from pathlib import Path

from finrag.financebench import build_pilot, load_questions, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions", type=Path,
        default=Path("data/external/financebench/financebench_open_source.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/eval/financebench_pilot.json"),
    )
    args = parser.parse_args()

    try:
        questions = load_questions(args.questions)
        pilot = build_pilot(questions)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pilot, indent=2) + "\n", encoding="utf-8")
    print(summarize(questions, pilot))
    print(f"Manifest: {args.output}")


if __name__ == "__main__":
    main()
