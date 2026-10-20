"""Prepare a fixed development subset from the public FinanceBench annotations."""

import json
from collections import Counter
from pathlib import Path

PILOT_DOCUMENTS = (
    "3M_2018_10K",
    "PEPSICO_2022_10K",
    "AMD_2022_10K",
    "BOEING_2022_10K",
    "ULTABEAUTY_2023Q4_EARNINGS",
    "3M_2023Q2_10Q",
)


def load_questions(path: str | Path) -> list[dict]:
    questions = []
    seen_ids = set()

    with Path(path).open(encoding="utf-8") as source:
        for lineno, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                question_id = entry["financebench_id"]
                evidence = entry["evidence"]
                if (
                    not isinstance(question_id, str)
                    or not question_id
                    or not isinstance(entry["doc_name"], str)
                    or not entry["doc_name"]
                    or not isinstance(entry["question"], str)
                    or not entry["question"].strip()
                    or not isinstance(evidence, list)
                    or not evidence
                    or not all(
                        isinstance(item, dict)
                        and type(item.get("evidence_page_num")) is int
                        and item["evidence_page_num"] >= 0
                        and isinstance(item.get("doc_name"), str)
                        and item["doc_name"]
                        for item in evidence
                    )
                ):
                    raise ValueError("invalid question or evidence metadata")
                if question_id in seen_ids:
                    raise ValueError(f"duplicate question ID {question_id}")
            except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid FinanceBench record at line {lineno}: {exc}") from exc
            questions.append(entry)
            seen_ids.add(question_id)

    if not questions:
        raise ValueError("FinanceBench JSONL is empty")
    return questions


def build_pilot(questions: list[dict], docs=PILOT_DOCUMENTS) -> dict:
    selected = [question for question in questions if question["doc_name"] in docs]
    missing = set(docs) - {question["doc_name"] for question in selected}
    if missing:
        raise ValueError(f"Pilot documents missing from dataset: {sorted(missing)}")

    selected.sort(key=lambda question: (docs.index(question["doc_name"]), question["financebench_id"]))
    return {
        "dataset": "FinanceBench public annotations",
        "purpose": "Fixed development pilot; not a representative test set",
        "source": "https://github.com/patronus-ai/financebench",
        "documents": list(docs),
        "questions": [
            {
                "financebench_id": question["financebench_id"],
                "doc_name": question["doc_name"],
                "question_type": question["question_type"],
                "gold_pages": sorted({ev["evidence_page_num"] for ev in question["evidence"]}),
            }
            for question in selected
        ],
    }


def summarize(questions: list[dict], pilot: dict) -> str:
    types = Counter(item["question_type"] for item in pilot["questions"])
    return "\n".join([
        f"FinanceBench: {len(questions)} questions / {len({q['doc_name'] for q in questions})} reports",
        f"Pilot: {len(pilot['questions'])} questions / {len(pilot['documents'])} reports",
        "Pilot question types: " + ", ".join(f"{name}={count}" for name, count in sorted(types.items())),
    ])
