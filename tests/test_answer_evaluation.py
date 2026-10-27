import json
from pathlib import Path

import pytest

from finrag.answer_evaluation import (
    EvaluationError,
    GoldDocument,
    GoldQuestion,
    RequiredEvidence,
    ExpectedAnswer,
    Trace,
    Assessment,
    ClaimAssessment,
    CitationAssessment,
    EvidenceSnapshot,
    parse_gold,
    parse_traces,
    parse_assessments,
    evaluate_answer_quality,
    render_report,
    sha256_text,
    load_gold,
    load_traces,
    load_assessments,
    write_report,
    SCHEMA_GOLD,
    SCHEMA_TRACE,
    SCHEMA_ASSESSMENT,
)


def sha(text):
    return sha256_text(text)


def gold_doc(source="report.txt", text="Revenue was 42 million.", doc_id="doc1"):
    return {
        "document_id": doc_id,
        "source": source,
        "sha256": sha(text),
        "content": text,
    }


def required_ev(
    evidence_id="e1",
    doc_id="doc1",
    source="report.txt",
    text="Revenue was 42 million.",
    evidence_sha=None,
):
    return {
        "evidence_id": evidence_id,
        "document_id": doc_id,
        "source": source,
        "text": text,
        "sha256": evidence_sha or sha(text),
    }


def gold_question(
    qid="q1",
    question="What was revenue?",
    answerability="answerable",
    expected_decision="answer",
    required=None,
    expected_answer=None,
    review_status="reviewed",
    docs=None,
):
    if docs is None:
        docs = [gold_doc()]
    if required is None:
        required = [required_ev()] if answerability == "answerable" else []
    if expected_answer is None and answerability == "answerable":
        expected_answer = {"kind": "numeric", "value": 42, "text": "42 million"}
    if answerability == "unanswerable":
        expected_answer = None
    return {
        "question_id": qid,
        "question": question,
        "documents": docs,
        "answerability": answerability,
        "expected_decision": expected_decision,
        "required_evidence": required,
        "expected_answer": expected_answer,
        "review_status": review_status,
    }


def trace_evidence(
    evidence_id="ev1",
    source="report.txt",
    text="Revenue was 42 million.",
    doc_id="doc1",
):
    return {
        "evidence_id": evidence_id,
        "source": source,
        "text": text,
        "text_sha256": sha(text),
        "source_sha256": sha("Revenue was 42 million."),
        "document_id": doc_id,
    }


def trace_obj(
    qid="q1",
    status="produced",
    raw_answer="Revenue was 42 million [1].",
    answer_mode="generated",
    evidence=None,
    citation_ids=None,
    candidate_evidence="__default__",
    assessment_needed=True,
):
    if evidence is None:
        evidence = [trace_evidence()]
    if citation_ids is None:
        citation_ids = [1] if answer_mode == "generated" and status == "produced" else []
    trace = {
        "schema_version": SCHEMA_TRACE,
        "trace_id": f"t-{qid}",
        "question_id": qid,
        "answer_mode": answer_mode,
        "status": status,
        "raw_answer": raw_answer,
        "evidence": evidence,
        "citation_ids": citation_ids,
        "execution": {"backend": "ollama", "model": "qwen3:4b-instruct", "duration_ms": 12},
    }
    if candidate_evidence == "__default__":
        trace["candidate_evidence"] = [dict(e) for e in evidence]
    elif candidate_evidence is not None:
        trace["candidate_evidence"] = candidate_evidence
    return trace


def assessment_obj(
    qid="q1",
    context="sufficient",
    correctness="correct",
    claims=None,
    citations=None,
    review_status="reviewed",
    secondary=None,
):
    if claims is None:
        claims = [
            {
                "claim_id": "c1",
                "text": "Revenue was 42 million",
                "label": "supported",
                "evidence_sha256": [sha("Revenue was 42 million.")],
            }
        ]
    if citations is None:
        citations = [
            {
                "citation_id": 1,
                "label": "supported",
                "evidence_sha256": sha("Revenue was 42 million."),
            }
        ]
    return {
        "schema_version": SCHEMA_ASSESSMENT,
        "assessment_id": f"a-{qid}",
        "question_id": qid,
        "review_status": review_status,
        "context_sufficiency": context,
        "answer_correctness": correctness,
        "claims": claims,
        "citations": citations,
        "secondary_errors": secondary or [],
    }


def run(gold=None, traces=None, assessments=None):
    if gold is None:
        gold = [gold_question()]
    if traces is None:
        traces = [trace_obj()]
    if assessments is None:
        assessments = [assessment_obj()]
    g = parse_gold({"schema_version": SCHEMA_GOLD, "questions": gold})
    t = parse_traces([_strip_schema(tr) for tr in traces])
    a = parse_assessments([_strip_schema(asm) for asm in assessments])
    return evaluate_answer_quality(g, t, a)


def _strip_schema(record):
    record = dict(record)
    record.pop("schema_version", None)
    return record


def test_correct_grounded_answer_success():
    report = run()
    case = report["cases"][0]
    assert case["outcome"] == "success"
    assert case["failure_stage"] is None
    assert case["integrity"]["status"] == "ok"
    agg = report["aggregates"]
    assert agg["required_evidence_coverage"]["value"] == 1.0
    assert agg["context_sufficiency"]["value"] == 1.0
    assert agg["answer_correctness"]["value"] == 1.0
    assert agg["claim_faithfulness"]["value"] == 1.0
    assert agg["citation_structural_validity"]["value"] == 1.0
    assert agg["citation_support"]["value"] == 1.0


def test_incorrect_numeric_answer():
    tr = trace_obj(raw_answer="Revenue was 41 million [1].")
    asm = assessment_obj(correctness="incorrect", claims=[], citations=[])
    report = run(traces=[tr], assessments=[asm])
    case = report["cases"][0]
    assert case["outcome"] == "failure"
    assert case["failure_stage"] == "generation"
    assert "answer_incorrect" in case["failure_reasons"]
    assert report["aggregates"]["answer_correctness"]["value"] == 0.0


def test_unsupported_claim():
    asm = assessment_obj(
        claims=[
            {
                "claim_id": "c1",
                "text": "Revenue was 42 million",
                "label": "supported",
                "evidence_sha256": [sha("Revenue was 42 million.")],
            },
            {
                "claim_id": "c2",
                "text": "Revenue grew 10%",
                "label": "unsupported",
            },
        ]
    )
    tr = trace_obj(raw_answer="Revenue was 42 million. Revenue grew 10% [1].")
    report = run(traces=[tr], assessments=[asm])
    case = report["cases"][0]
    assert case["outcome"] == "failure"
    assert case["failure_stage"] == "generation"
    assert "unsupported_claim" in case["failure_reasons"]
    assert case["claims"]["unsupported_count"] == 1
    assert report["aggregates"]["claim_faithfulness"]["value"] == 0.5


def test_contradicted_claim():
    asm = assessment_obj(
        claims=[
            {
                "claim_id": "c1",
                "text": "Revenue was 42 million",
                "label": "contradicted",
            }
        ]
    )
    report = run(assessments=[asm])
    case = report["cases"][0]
    assert case["outcome"] == "failure"
    assert "contradicted_claim" in case["failure_reasons"]
    assert case["claims"]["contradicted_count"] == 1
    assert report["aggregates"]["contradicted_claims"]["numerator"] == 1


def test_malformed_citations():
    for raw in ["Revenue 42 [", "Revenue 42 [one]", "Revenue 42 []", "Revenue 42 [1,]", "Revenue 42 [1-2]"]:
        tr = trace_obj(raw_answer=raw)
        asm = assessment_obj(claims=[], citations=[])
        report = run(traces=[tr], assessments=[asm])
        case = report["cases"][0]
        assert case["outcome"] == "failure"
        assert case["failure_stage"] == "citation"
        assert case["citations"]["structural_validity"] == "invalid"


def test_missing_citations():
    tr = trace_obj(raw_answer="Revenue was 42 million.")
    asm = assessment_obj(claims=[], citations=[])
    report = run(traces=[tr], assessments=[asm])
    case = report["cases"][0]
    assert case["outcome"] == "failure"
    assert case["failure_stage"] == "citation"
    assert "missing_citations" in case["citations"]["syntax_errors"]


def test_out_of_range_citations():
    tr = trace_obj(raw_answer="Revenue was 42 million [3].")
    asm = assessment_obj(claims=[], citations=[])
    report = run(traces=[tr], assessments=[asm])
    case = report["cases"][0]
    assert case["outcome"] == "failure"
    assert case["failure_stage"] == "citation"
    assert "out_of_range" in case["citations"]["syntax_errors"]


def test_irrelevant_citation():
    asm = assessment_obj(
        citations=[
            {
                "citation_id": 1,
                "label": "irrelevant",
                "evidence_sha256": sha("Revenue was 42 million."),
            }
        ]
    )
    report = run(assessments=[asm])
    case = report["cases"][0]
    assert case["outcome"] == "failure"
    assert case["failure_stage"] == "citation"
    assert "citation_irrelevant" in case["failure_reasons"]
    assert report["aggregates"]["citation_support"]["value"] == 0.0


def test_false_abstention():
    tr = trace_obj(status="model_abstention", raw_answer="INSUFFICIENT_EVIDENCE", citation_ids=[])
    asm = assessment_obj(context="insufficient", correctness="not_applicable", claims=[], citations=[])
    report = run(traces=[tr], assessments=[asm])
    case = report["cases"][0]
    assert case["decision"]["false_abstention"] is True
    assert case["outcome"] == "failure"
    assert case["failure_stage"] == "generation"
    assert "false_abstention" in case["failure_reasons"]


def test_correct_abstention():
    g = gold_question(
        answerability="unanswerable",
        expected_decision="abstain",
        required=[],
        expected_answer=None,
    )
    tr = trace_obj(
        status="model_abstention",
        raw_answer="INSUFFICIENT_EVIDENCE",
        citation_ids=[],
        evidence=[],
        candidate_evidence=[],
    )
    asm = assessment_obj(
        context="insufficient",
        correctness="not_applicable",
        claims=[],
        citations=[],
    )
    report = run(gold=[g], traces=[tr], assessments=[asm])
    case = report["cases"][0]
    assert case["outcome"] == "success"
    assert case["failure_stage"] is None
    assert case["decision"]["correct_abstention"] is True


def test_false_answer():
    g = gold_question(answerability="unanswerable", expected_decision="abstain", required=[], expected_answer=None)
    tr = trace_obj(status="produced", raw_answer="Revenue was 42 million [1].")
    asm = assessment_obj(context="insufficient", correctness="incorrect", claims=[], citations=[])
    # fix gold doc still needed; keep evidence
    report = run(gold=[g], traces=[tr], assessments=[asm])
    case = report["cases"][0]
    assert case["decision"]["false_answer"] is True
    assert case["outcome"] == "failure"
    assert case["failure_stage"] == "generation"


def test_missing_retrieval_evidence():
    tr = trace_obj(evidence=[], citation_ids=[], candidate_evidence=[])
    tr["raw_answer"] = "INSUFFICIENT_EVIDENCE"
    tr["status"] = "no_evidence"
    asm = assessment_obj(context="insufficient", correctness="not_applicable", claims=[], citations=[])
    # evidence list empty but trace schema requires evidence? empty allowed
    report = run(traces=[tr], assessments=[asm])
    case = report["cases"][0]
    assert case["evidence"]["candidate_evidence"] == "available"
    assert case["evidence"]["covered_count"] == 0
    assert case["failure_stage"] == "retrieval"
    assert "retrieval_required_evidence_missing" in case["failure_reasons"]


def test_context_assembly_drop():
    ev = trace_evidence()
    candidate = [dict(ev)]
    tr = trace_obj(evidence=[], citation_ids=[], candidate_evidence=candidate)
    tr["raw_answer"] = "INSUFFICIENT_EVIDENCE"
    tr["status"] = "no_evidence"
    asm = assessment_obj(context="insufficient", correctness="not_applicable", claims=[], citations=[])
    report = run(traces=[tr], assessments=[asm])
    case = report["cases"][0]
    assert case["evidence"]["candidate_evidence"] == "available"
    assert case["failure_stage"] == "context_assembly"
    assert "context_assembly_required_evidence_dropped" in case["failure_reasons"]


def test_candidate_information_unavailable():
    tr = trace_obj(evidence=[], citation_ids=[], candidate_evidence=None)
    tr["raw_answer"] = "INSUFFICIENT_EVIDENCE"
    tr["status"] = "no_evidence"
    asm = assessment_obj(context="insufficient", correctness="not_applicable", claims=[], citations=[])
    report = run(traces=[tr], assessments=[asm])
    case = report["cases"][0]
    assert case["evidence"]["candidate_evidence"] == "unavailable"
    assert case["failure_stage"] != "retrieval"
    assert case["failure_stage"] == "unknown"


def test_backend_error():
    tr = trace_obj(status="backend_error", raw_answer="", citation_ids=[], evidence=[], candidate_evidence=None)
    asm = assessment_obj(context="unknown", correctness="unknown", claims=[], citations=[], review_status="unreviewed")
    report = run(traces=[tr], assessments=[asm])
    case = report["cases"][0]
    assert case["outcome"] == "failure"
    assert case["failure_stage"] == "backend"
    assert case["trace_status"] == "backend_error"


def test_missing_assessments():
    report = run(assessments=[])
    case = report["cases"][0]
    assert case["outcome"] == "unassessed"
    assert "missing_assessment" in case["unassessed_reasons"]
    assert case["outcome"] != "success"
    agg = report["aggregates"]
    assert agg["answer_correctness"]["denominator"] == 0
    assert agg["answer_correctness"]["value"] is None


def test_hash_mismatch():
    ev = trace_evidence()
    ev["text_sha256"] = sha("wrong")
    tr = trace_obj(evidence=[ev])
    report = run(traces=[tr])
    case = report["cases"][0]
    assert case["outcome"] == "invalid"
    assert case["integrity"]["status"] == "failed"
    assert any("text SHA-256 mismatch" in e for e in case["integrity"]["errors"])
    assert report["aggregates"]["integrity_failures"]["numerator"] == 1


def test_source_hash_mismatch():
    ev = trace_evidence()
    ev["source_sha256"] = sha("wrong source")
    tr = trace_obj(evidence=[ev])
    report = run(traces=[tr])
    case = report["cases"][0]
    assert case["outcome"] == "invalid"
    assert any("source SHA-256 mismatch" in e for e in case["integrity"]["errors"])


def test_invalid_schema_and_duplicate_ids():
    with pytest.raises(EvaluationError):
        parse_gold({"schema_version": SCHEMA_GOLD, "questions": []})
    with pytest.raises(EvaluationError):
        parse_gold({"schema_version": SCHEMA_GOLD, "questions": [gold_question(), gold_question()]})
    tr = trace_obj()
    tr2 = dict(tr)
    tr2["trace_id"] = "t-q2"
    # duplicate question_id
    recs = [parse_traces([_strip_schema(tr)])[0], parse_traces([_strip_schema(tr2)])[0]]
    # force duplicate qid
    recs[1] = Trace(**{**recs[1].__dict__, "question_id": "q1"})
    with pytest.raises(EvaluationError):
        parse_traces([_strip_schema(tr), {**_strip_schema(tr2), "question_id": "q1"}])
    # invalid assessment schema
    with pytest.raises(EvaluationError):
        parse_assessments([{"schema_version": SCHEMA_ASSESSMENT, "question_id": "q1"}])
    # trace invalid status
    bad = _strip_schema(trace_obj())
    bad["status"] = "nope"
    with pytest.raises(EvaluationError):
        parse_traces([bad])


def test_deterministic_report_bytes():
    report1 = run()
    report2 = run()
    assert render_report(report1) == render_report(report2)
    # reversed input order should not change report
    tr = trace_obj()
    asm = assessment_obj()
    g = gold_question()
    g2 = gold_question(qid="q2")
    t2 = trace_obj(qid="q2")
    a2 = assessment_obj(qid="q2")
    golds = [g, g2]
    traces = [tr, t2]
    assess = [asm, a2]
    # shuffle
    report_a = evaluate_answer_quality(
        parse_gold({"schema_version": SCHEMA_GOLD, "questions": golds}),
        parse_traces([_strip_schema(x) for x in traces]),
        parse_assessments([_strip_schema(x) for x in assess]),
    )
    report_b = evaluate_answer_quality(
        parse_gold({"schema_version": SCHEMA_GOLD, "questions": list(reversed(golds))}),
        parse_traces([_strip_schema(x) for x in reversed(traces)]),
        parse_assessments([_strip_schema(x) for x in reversed(assess)]),
    )
    assert render_report(report_a) == render_report(report_b)


def test_extractive_output_handled_separately():
    tr = trace_obj(
        answer_mode="extractive",
        raw_answer="Revenue was 42 million.",
        citation_ids=[],
    )
    asm = assessment_obj(citations=[])
    report = run(traces=[tr], assessments=[asm])
    case = report["cases"][0]
    assert case["outcome"] == "success"
    assert case["citations"]["requirement"] == "not_applicable"
    assert case["citations"]["structural_validity"] == "not_applicable"


def test_coverage_does_not_prove_context_sufficiency():
    asm = assessment_obj(context="insufficient")
    report = run(assessments=[asm])
    case = report["cases"][0]
    assert case["evidence"]["coverage"]["value"] == 1.0
    assert case["context_sufficiency"] == "insufficient"
    assert case["outcome"] == "failure"
    agg = report["aggregates"]
    assert agg["required_evidence_coverage"]["value"] == 1.0
    assert agg["context_sufficiency"]["value"] == 0.0


def test_candidate_missing_not_retrieval_failure():
    tr = trace_obj(evidence=[], citation_ids=[], candidate_evidence=None)
    tr["raw_answer"] = "INSUFFICIENT_EVIDENCE"
    tr["status"] = "no_evidence"
    asm = assessment_obj(context="insufficient", correctness="not_applicable", claims=[], citations=[])
    report = run(traces=[tr], assessments=[asm])
    case = report["cases"][0]
    assert case["evidence"]["candidate_evidence"] == "unavailable"
    assert case["failure_stage"] != "retrieval"


def test_loader_json_and_jsonl(tmp_path):
    gold = [gold_question()]
    trace = trace_obj()
    asm = assessment_obj()
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps({"schema_version": SCHEMA_GOLD, "questions": gold}))
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text(json.dumps(_strip_schema(trace)) + "\n")
    # headerless trace needs schema_version per record
    trace_rec = dict(_strip_schema(trace))
    trace_rec["schema_version"] = SCHEMA_TRACE
    trace_path.write_text(json.dumps(trace_rec) + "\n")
    asm_path = tmp_path / "assessments.jsonl"
    asm_rec = dict(_strip_schema(asm))
    asm_rec["schema_version"] = SCHEMA_ASSESSMENT
    asm_path.write_text(json.dumps(asm_rec) + "\n")
    g = load_gold(gold_path)
    t = load_traces(trace_path)
    a = load_assessments(asm_path)
    report = evaluate_answer_quality(g, t, a)
    assert report["cases"][0]["outcome"] == "success"


def test_write_report_and_cli(tmp_path):
    report = run()
    path = write_report(report, tmp_path)
    assert path.name == "answer_quality_report.json"
    assert path.read_text().endswith("\n")
    # byte identical rewrite
    before = path.read_bytes()
    write_report(report, tmp_path)
    assert path.read_bytes() == before
    # CLI
    import subprocess, sys
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"schema_version": SCHEMA_GOLD, "questions": [gold_question()]}))
    tr = tmp_path / "traces.jsonl"
    tr.write_text(json.dumps({**_strip_schema(trace_obj()), "schema_version": SCHEMA_TRACE}) + "\n")
    asm = tmp_path / "assessments.jsonl"
    asm.write_text(json.dumps({**_strip_schema(assessment_obj()), "schema_version": SCHEMA_ASSESSMENT}) + "\n")
    out = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "evaluate_answer_quality.py"),
         "--gold", str(gold), "--traces", str(tr), "--assessments", str(asm), "--out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "answer_quality_report.json").exists()


def test_finrag_file_resolves_in_worktree():
    import finrag

    assert "FinRAG-eval-engine" in str(finrag.__file__)