import json

import pytest

from finrag.financebench import build_pilot, load_questions


def make_question(question_id="id1", doc="one", pages=(1, 3), question_type="metrics-generated"):
    return {
        "financebench_id": question_id,
        "doc_name": doc,
        "question": "What was revenue?",
        "question_type": question_type,
        "evidence": [{"doc_name": doc, "evidence_page_num": page} for page in pages],
    }


def test_loads_questions_and_rejects_invalid_metadata(tmp_path):
    path = tmp_path / "annotations.jsonl"
    path.write_text(json.dumps(make_question()) + "\n", encoding="utf-8")
    assert load_questions(path)[0]["financebench_id"] == "id1"

    for bad in (make_question(pages=(-1,)), make_question(pages=()), make_question(pages=("1",))):
        path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="line 1"):
            load_questions(path)


def test_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "annotations.jsonl"
    row = json.dumps(make_question())
    path.write_text(row + "\n" + row + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate question ID"):
        load_questions(path)


def test_builds_deterministic_manifest_without_answer_text():
    questions = [
        make_question("b", doc="second", pages=(4, 1, 4)),
        make_question("a", doc="first", pages=(2,)),
    ]
    manifest = build_pilot(questions, docs=("first", "second"))
    assert [q["financebench_id"] for q in manifest["questions"]] == ["a", "b"]
    assert manifest["questions"][1]["gold_pages"] == [1, 4]
    assert "answer" not in str(manifest)
    with pytest.raises(ValueError, match="missing"):
        build_pilot(questions, docs=("third",))
