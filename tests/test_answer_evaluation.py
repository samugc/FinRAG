import hashlib
import json
import os
from pathlib import Path

import pytest

from finrag.answer_evaluation import (
    SCHEMA_GOLD,
    SCHEMA_TRACE,
    SCHEMA_ASSESSMENT,
    EvaluationError,
    parse_gold,
    parse_traces,
    parse_assessments,
    evaluate_answer_quality,
    render_report,
    write_report,
    sha256_text,
    sha256_file,
    load_gold,
    load_traces,
    load_assessments,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("FINRAG_EVAL_DATA_ROOT", str(REPO_ROOT.parent / "FinRAG-eval-data")))
REAL_GOLD = DATA_ROOT / "data" / "eval" / "answer_golden_v1.json"
REAL_TRACES = DATA_ROOT / "tests" / "fixtures" / "answer_eval" / "traces.jsonl"
REAL_ASSESSMENTS = DATA_ROOT / "tests" / "fixtures" / "answer_eval" / "assessments.jsonl"
REAL_EXPECTED = DATA_ROOT / "tests" / "fixtures" / "answer_eval" / "expected_cases.json"


def sha(text):
    return sha256_text(text)


def canonical_gold():
    source_text = "Northstar reported revenue of €420 million in 2025."
    return {
        "schema_version": SCHEMA_GOLD,
        "dataset_id": "unit-gold",
        "corpus": [
            {
                "document_id": "sample_report",
                "source": "data/raw/sample_report.txt",
                "sha256": "0" * 64,
                "provenance": "synthetic",
            }
        ],
        "questions": [
            {
                "question_id": "q1",
                "question": "What was revenue in 2025?",
                "document_id": "sample_report",
                "answerability": "answerable",
                "expected_decision": "answer",
                "expected_answer": {
                    "status": "reviewed",
                    "text": "€420 million",
                    "evidence_ids": ["E1"],
                },
                "required_evidence": [
                    {
                        "evidence_id": "E1",
                        "document_id": "sample_report",
                        "chunk_id": 0,
                        "page_index": None,
                        "quote": "Northstar reported revenue of €420 million in 2025.",
                        "review_status": "reviewed",
                    }
                ],
                "review_status": "reviewed",
                "notes": None,
            }
        ],
    }


def canonical_trace(*, candidate_evidence=None, status="produced", raw_answer=None):
    text = "Northstar reported revenue of €420 million in 2025, compared with €390 million in 2024."
    if raw_answer is None:
        raw_answer = "Revenue was €420 million in 2025 [1]."
    return {
        "schema_version": SCHEMA_TRACE,
        "trace_id": "trace-q1",
        "run_id": "unit-run",
        "question_id": "q1",
        "gold_sha256": "0" * 64,
        "config": {"retriever": "minilm", "generator": "qwen", "k": 3},
        "baseline": {"git_commit": "0" * 40, "git_dirty": False},
        "result": {"status": status, "raw_answer": raw_answer, "reason": "Passed structural checks only"},
        "evidence": [
            {
                "citation_id": 1,
                "kind": "chunk",
                "document_id": "sample_report",
                "source": "data/raw/sample_report.txt",
                "chunk_id": 0,
                "page_index": None,
                "score": 0.95,
                "text": text,
                "text_sha256": sha(text),
            }
        ],
        "candidate_evidence": candidate_evidence,
        "error": None,
    }


def canonical_assessment():
    text = "Northstar reported revenue of €420 million in 2025, compared with €390 million in 2024."
    return {
        "schema_version": SCHEMA_ASSESSMENT,
        "assessment_id": "assess-q1",
        "trace_id": "trace-q1",
        "question_id": "q1",
        "reviewer_id": "unit",
        "reviewed_at": "2026-10-27",
        "context_sufficient": "sufficient",
        "decision": "answered",
        "answer_correct": "correct",
        "claims": [
            {"claim_id": "A1", "text": "Revenue was €420 million in 2025", "support": "supported"}
        ],
        "citations": [
            {
                "claim_id": "A1",
                "citation_id": 1,
                "evidence_sha256": sha(text),
                "support": "supports",
            }
        ],
        "secondary_labels": [],
        "notes": None,
    }


def run_canonical(candidate_evidence=None, status="produced", raw_answer=None, assessment=None):
    gold = parse_gold(canonical_gold())
    trace = parse_traces([canonical_trace(candidate_evidence=candidate_evidence, status=status, raw_answer=raw_answer)])
    assessments = parse_assessments([assessment if assessment is not None else canonical_assessment()])
    return evaluate_answer_quality(gold, trace, assessments)


def test_canonical_success():
    report = run_canonical()
    case = report["cases"][0]
    assert case["outcome"] == "success"
    assert case["failure_stage"] is None
    assert case["integrity"]["status"] == "ok"
    assert case["abstention_error"] == "none"
    assert case["coverage_complete_but_context_insufficient"] is False


def test_candidate_null_vs_empty():
    report_null = run_canonical(candidate_evidence=None)
    report_empty = run_canonical(candidate_evidence=[])
    assert report_null["cases"][0]["evidence"]["candidate_evidence"] == "unavailable"
    assert report_empty["cases"][0]["evidence"]["candidate_evidence"] == "available"


def test_backend_error_excluded_from_success():
    tr = canonical_trace(status="backend_error", raw_answer="", candidate_evidence=None)
    tr["result"]["reason"] = "TimeoutError"
    tr["error"] = "TimeoutError"
    gold = parse_gold(canonical_gold())
    trace = parse_traces([tr])
    report = evaluate_answer_quality(gold, trace, [])
    case = report["cases"][0]
    assert case["outcome"] != "success"
    assert case["trace_status"] == "backend_error"
    assert case["failure_stage"] == "backend"


def test_unassessed_not_success():
    gold = parse_gold(canonical_gold())
    trace = parse_traces([canonical_trace()])
    report = evaluate_answer_quality(gold, trace, [])
    case = report["cases"][0]
    assert case["outcome"] == "unassessed"
    assert case["outcome"] != "success"
    assert "missing_assessment" in case["unassessed_reasons"]


def test_zero_denominator_null():
    gold_data = canonical_gold()
    gold_data["questions"][0]["answerability"] = "unanswerable"
    gold_data["questions"][0]["expected_decision"] = "abstain"
    gold_data["questions"][0]["required_evidence"] = []
    gold_data["questions"][0]["expected_answer"] = {
        "status": "not_applicable",
        "text": None,
        "evidence_ids": [],
    }
    gold = parse_gold(gold_data)
    tr = canonical_trace(status="model_abstention", raw_answer="INSUFFICIENT_EVIDENCE", candidate_evidence=None)
    tr["evidence"] = []
    trace = parse_traces([tr])
    asm = canonical_assessment()
    asm["context_sufficient"] = "not_applicable"
    asm["decision"] = "abstained"
    asm["answer_correct"] = "not_applicable"
    asm["claims"] = []
    asm["citations"] = []
    assessments = parse_assessments([asm])
    report = evaluate_answer_quality(gold, trace, assessments)
    case = report["cases"][0]
    assert case["evidence"]["coverage"]["value"] is None
    assert case["evidence"]["coverage"]["status"] == "not_applicable"


def test_invalid_schema_and_duplicate_ids():
    with pytest.raises(EvaluationError):
        parse_gold({"schema_version": SCHEMA_GOLD, "dataset_id": "x", "corpus": [], "questions": []})
    bad_gold = canonical_gold()
    bad_gold["questions"].append(dict(bad_gold["questions"][0]))
    with pytest.raises(EvaluationError):
        parse_gold(bad_gold)
    bad_trace = canonical_trace()
    bad_trace["trace_id"] = "trace-q2"
    bad_trace["question_id"] = "q1"
    with pytest.raises(EvaluationError):
        parse_traces([canonical_trace(), bad_trace])
    bad_assess = canonical_assessment()
    bad_assess["assessment_id"] = "assess-q2"
    bad_assess["question_id"] = "q1"
    with pytest.raises(EvaluationError):
        parse_assessments([canonical_assessment(), bad_assess])


def test_deterministic_report():
    report1 = run_canonical()
    report2 = run_canonical()
    assert render_report(report1) == render_report(report2)


def test_portable_import():
    import finrag

    repo_root = Path(__file__).resolve().parents[1]
    assert Path(finrag.__file__).resolve().is_relative_to(repo_root)


@pytest.mark.skipif(not REAL_GOLD.exists(), reason="canonical data worktree unavailable")
def test_real_data_counts_and_hashes():
    gold = load_gold(REAL_GOLD)
    traces = load_traces(REAL_TRACES)
    assessments = load_assessments(REAL_ASSESSMENTS)
    assert len(gold) == 16
    assert len(traces) == 16
    assert len(assessments) == 14


@pytest.mark.skipif(not REAL_GOLD.exists(), reason="canonical data worktree unavailable")
def test_real_data_primary_stages():
    gold = load_gold(REAL_GOLD)
    traces = load_traces(REAL_TRACES)
    assessments = load_assessments(REAL_ASSESSMENTS)
    gold_sha = sha256_file(REAL_GOLD)
    report = evaluate_answer_quality(gold, traces, assessments, gold_sha256=gold_sha)
    expected = json.loads(REAL_EXPECTED.read_text())
    cases = {c["question_id"]: c for c in report["cases"]}
    mismatches = 0
    for ec in expected["cases"]:
        case = cases[ec["question_id"]]
        if case["failure_stage"] != ec["expected_primary"]:
            mismatches += 1
        # required evidence
        rexp = ec["assertions"]["required_evidence"]
        if rexp["status"] != "not_applicable":
            assert case["evidence"]["coverage"]["status"] == rexp["status"]
            assert case["evidence"]["covered_count"] == rexp["matched"]
            assert case["evidence"]["required_count"] == rexp["required"]
        # context
        if ec["assertions"]["context_sufficient"] is not None:
            assert case["context_sufficiency"] == ec["assertions"]["context_sufficient"]
        else:
            assert case["context_sufficiency"] is None
        # abstention error
        if ec["assertions"]["abstention_error"] != "not_assessed":
            assert case["abstention_error"] == ec["assertions"]["abstention_error"]
        # diagnostic
        assert case["coverage_complete_but_context_insufficient"] == ec["assertions"]["coverage_complete_but_context_insufficient"]
    assert mismatches == 0
    # Exact secondary-label comparison.
    secondary_mismatches = 0
    for ec in expected["cases"]:
        case = cases[ec["question_id"]]
        if set(case["secondary_labels"]) != set(ec["secondary_contains"]):
            secondary_mismatches += 1
    assert secondary_mismatches == 0
    assert report["aggregates"]["integrity_failures"]["numerator"] == 0


@pytest.mark.skipif(not REAL_GOLD.exists(), reason="canonical data worktree unavailable")
def test_real_data_metric_internal_consistency():
    gold = load_gold(REAL_GOLD)
    traces = load_traces(REAL_TRACES)
    assessments = load_assessments(REAL_ASSESSMENTS)
    gold_sha = sha256_file(REAL_GOLD)
    report = evaluate_answer_quality(gold, traces, assessments, gold_sha256=gold_sha)
    agg = report["aggregates"]
    # numerators <= denominators where applicable
    for key in ("trace_coverage", "assessment_coverage", "integrity_failures", "decision_accuracy",
                "false_abstention", "false_answer", "correct_abstention", "required_evidence_coverage",
                "context_sufficiency", "answer_correctness", "claim_faithfulness", "citation_structural_validity",
                "citation_support"):
        m = agg[key]
        assert m["numerator"] <= m["denominator"]
        if m["denominator"] == 0:
            assert m["value"] is None
    # unassessed cases are not successes
    unassessed = [c for c in report["cases"] if c["outcome"] == "unassessed"]
    for c in unassessed:
        assert c["outcome"] != "success"
    # backend errors excluded from decision accuracy denominator
    backend_cases = [c for c in report["cases"] if c["trace_status"] == "backend_error"]
    assert len(backend_cases) == 1
    assert backend_cases[0]["observed_decision"] is None
    # decision metrics are structural and explicitly exclude backend/non-decisions
    observed_decisions = [c for c in report["cases"] if c["observed_decision"] in {"answered", "abstained"}]
    assert agg["decision_accuracy"]["denominator"] == len(observed_decisions)
    expected_answer_observed = [c for c in report["cases"] if c["expected_decision"] == "answer" and c["observed_decision"] in {"answered", "abstained"}]
    assert agg["false_abstention"]["denominator"] == len(expected_answer_observed)
    expected_abstain_observed = [c for c in report["cases"] if c["expected_decision"] == "abstain" and c["observed_decision"] in {"answered", "abstained"}]
    assert agg["false_answer"]["denominator"] == len(expected_abstain_observed)
    assert agg["assessment_coverage"]["numerator"] == 14
    assert agg["assessment_coverage"]["denominator"] == 16


@pytest.mark.skipif(not REAL_GOLD.exists(), reason="canonical data worktree unavailable")
def test_real_data_determinism(tmp_path):
    gold = load_gold(REAL_GOLD)
    traces = load_traces(REAL_TRACES)
    assessments = load_assessments(REAL_ASSESSMENTS)
    gold_sha = sha256_file(REAL_GOLD)
    report1 = evaluate_answer_quality(gold, traces, assessments, gold_sha256=gold_sha)
    report2 = evaluate_answer_quality(gold, traces, assessments, gold_sha256=gold_sha)
    assert render_report(report1) == render_report(report2)
    # byte-identical file output
    p1 = write_report(report1, tmp_path / "a")
    p2 = write_report(report2, tmp_path / "b")
    assert p1.read_bytes() == p2.read_bytes()


@pytest.mark.skipif(not REAL_GOLD.exists(), reason="canonical data worktree unavailable")
def test_cli_real_data(tmp_path):
    import subprocess
    import sys

    out = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_answer_quality.py"),
            "--gold", str(REAL_GOLD),
            "--traces", str(REAL_TRACES),
            "--assessments", str(REAL_ASSESSMENTS),
            "--out", str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "report.json").exists()

def test_incorrect_evidence_text_hash():
    tr = canonical_trace()
    tr["evidence"][0]["text_sha256"] = sha256_text("wrong")
    with pytest.raises(EvaluationError):
        parse_traces([tr])


def test_stale_gold_sha256():
    gold = parse_gold(canonical_gold())
    trace = parse_traces([canonical_trace()])
    assessments = parse_assessments([canonical_assessment()])
    report = evaluate_answer_quality(gold, trace, assessments, gold_sha256="f" * 64)
    case = report["cases"][0]
    assert case["integrity"]["status"] == "failed"
    assert any("gold_sha256 mismatch" in e for e in case["integrity"]["errors"])


def test_retrieval_vs_context_attribution():
    tr = canonical_trace(candidate_evidence=[{"document_id": "sample_report", "chunk_id": 0, "page_index": None}])
    # candidate contains required evidence, but context evidence is empty
    tr["evidence"] = []
    tr["result"]["status"] = "no_evidence"
    tr["result"]["raw_answer"] = "INSUFFICIENT_EVIDENCE"
    pass
    gold = parse_gold(canonical_gold())
    trace = parse_traces([tr])
    assessments = parse_assessments([canonical_assessment()]) if False else []
    # unassessed is acceptable; attribution should be context_assembly
    report = evaluate_answer_quality(gold, trace, [])
    case = report["cases"][0]
    assert case["evidence"]["candidate_evidence"] == "available"
    assert case["failure_stage"] == "context_assembly"


def test_malformed_and_out_of_range_qwen_citations():
    gold = parse_gold(canonical_gold())
    for raw in ["Revenue €420 million [", "Revenue €420 million [9]"]:
        tr = canonical_trace(raw_answer=raw)
        trace = parse_traces([tr])
        asm = canonical_assessment()
        asm["claims"] = []
        asm["citations"] = []
        assessments = parse_assessments([asm])
        report = evaluate_answer_quality(gold, trace, assessments)
        case = report["cases"][0]
        assert case["citations"]["structural_validity"] == "invalid"
        assert case["failure_stage"] == "citation"


def test_unassessed_claims():
    asm = canonical_assessment()
    asm["claims"] = [
        {"claim_id": "A1", "text": "Revenue was €420 million in 2025", "support": "unassessed"}
    ]
    gold = parse_gold(canonical_gold())
    trace = parse_traces([canonical_trace()])
    assessments = parse_assessments([asm])
    report = evaluate_answer_quality(gold, trace, assessments)
    case = report["cases"][0]
    assert case["claims"]["unassessed_count"] == 1
    assert case["outcome"] != "success"


def test_invalid_cross_entity_references():
    gold = parse_gold(canonical_gold())
    tr = canonical_trace()
    tr["question_id"] = "missing-question"
    trace = parse_traces([tr])
    with pytest.raises(EvaluationError):
        evaluate_answer_quality(gold, trace, [])
    asm = canonical_assessment()
    asm["trace_id"] = "missing-trace"
    asm["question_id"] = "q1"
    assessments = parse_assessments([asm])
    with pytest.raises(EvaluationError):
        evaluate_answer_quality(gold, parse_traces([canonical_trace()]), assessments)


def test_extractive_answer_without_qwen_citations():
    tr = canonical_trace()
    tr["config"]["generator"] = "extractive"
    tr["result"]["raw_answer"] = "Northstar reported revenue of €420 million in 2025."
    pass
    tr["evidence"][0]["citation_id"] = 1
    gold = parse_gold(canonical_gold())
    trace = parse_traces([tr])
    asm = canonical_assessment()
    asm["citations"] = []
    asm["claims"] = [
        {"claim_id": "A1", "text": "Northstar reported revenue of €420 million in 2025", "support": "supported"}
    ]
    assessments = parse_assessments([asm])
    report = evaluate_answer_quality(gold, trace, assessments)
    case = report["cases"][0]
    assert case["citations"]["requirement"] == "not_applicable"
    assert case["outcome"] == "success"


def test_incorrect_corpus_source_hash(tmp_path):
    root = tmp_path / "repo"
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "sample_report.txt").write_text("Northstar reported revenue of €420 million in 2025.", encoding="utf-8")
    gold_data = canonical_gold()
    gold_data["corpus"][0]["sha256"] = "f" * 64
    gold_path = root / "gold.json"
    gold_path.write_text(json.dumps(gold_data), encoding="utf-8")
    import os
    old_cwd = os.getcwd()
    os.chdir(root)
    try:
        with pytest.raises(EvaluationError):
            load_gold(gold_path)
    finally:
        os.chdir(old_cwd)


def test_byte_identical_serialization_canonical():
    report1 = run_canonical()
    report2 = run_canonical()
    assert render_report(report1) == render_report(report2)
