"""Deterministic, offline evaluation of RAG answer quality.

The evaluator keeps three entities strictly separate:

``finrag.answer_gold.v1``
    Reviewed ground truth: questions, document provenance, answerability,
    expected decisions, required evidence, expected answers and review status.

``finrag.answer_trace.v1``
    Immutable execution snapshots: exact generated output, ordered evidence,
    citation IDs, SHA-256 hashes, execution metadata, statuses and optional
    candidate evidence.

``finrag.answer_assessment.v1``
    Reviewed semantic judgements: context sufficiency, decision, answer
    correctness, claim support, citation support and secondary labels.

Only standard-library dependencies are used.  Semantic truth is never
inferred from lexical overlap: reviewed assessment labels are the only source
of semantic truth.  Structural checks are computed independently.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA_GOLD = "finrag.answer_gold.v1"
SCHEMA_TRACE = "finrag.answer_trace.v1"
SCHEMA_ASSESSMENT = "finrag.answer_assessment.v1"
SCHEMA_REPORT = "finrag.answer_evaluation_report.v1"

REPORT_FILENAME = "report.json"

ANSWER_STATUSES = frozenset({
    "produced",
    "model_abstention",
    "invalid_output",
    "no_evidence",
    "insufficient_context",
    "backend_error",
})
ABSTENTION_STATUSES = frozenset({
    "model_abstention",
    "no_evidence",
    "insufficient_context",
})
GENERATOR_VALUES = frozenset({"qwen", "extractive"})
FAILURE_STAGES = (
    "retrieval",
    "context_assembly",
    "generation",
    "citation",
    "backend",
    "unknown",
)
_HEX64 = re.compile(r"[0-9a-fA-F]{64}\Z")

CONTEXT_LABELS = frozenset({"sufficient", "insufficient", "not_applicable"})
ANSWER_CORRECT_LABELS = frozenset({"correct", "incorrect", "not_applicable"})
CLAIM_SUPPORT_LABELS = frozenset({
    "supported",
    "unsupported",
    "contradicted",
    "unassessed",
})
CITATION_SUPPORT_LABELS = frozenset({
    "supports",
    "does_not_support",
    "irrelevant",
    "unassessed",
})
REVIEW_STATUSES = frozenset({"reviewed", "pending"})
ANSWERABILITY_VALUES = frozenset({"answerable", "unanswerable"})
DECISION_VALUES = frozenset({"answer", "abstain"})
ASSESSMENT_DECISION_VALUES = frozenset({"answered", "abstained"})
GOLD_REVIEW_STATUSES = frozenset({"reviewed", "pending"})
CITATION_ORDER = (
    "no_substantive_content",
    "malformed_brackets",
    "missing_citations",
    "unsupported_syntax",
    "out_of_range",
)


class EvaluationError(ValueError):
    """Raised for invalid schemas, references or integrity failures."""


@dataclass(frozen=True)
class GoldDocument:
    document_id: str
    source: str
    sha256: str
    provenance: str


@dataclass(frozen=True)
class RequiredEvidence:
    evidence_id: str
    document_id: str
    chunk_id: int
    page_index: int | None
    quote: str
    review_status: str


@dataclass(frozen=True)
class ExpectedAnswer:
    status: str
    text: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class GoldQuestion:
    question_id: str
    question: str
    document_id: str
    answerability: str
    expected_decision: str
    expected_answer: ExpectedAnswer
    required_evidence: tuple[RequiredEvidence, ...]
    review_status: str
    notes: str | None


@dataclass(frozen=True)
class CandidateEvidence:
    document_id: str
    chunk_id: int
    page_index: int | None


@dataclass(frozen=True)
class TraceEvidence:
    citation_id: int
    kind: str
    document_id: str
    source: str
    chunk_id: int
    page_index: int | None
    score: float
    text: str
    text_sha256: str


@dataclass(frozen=True)
class Trace:
    trace_id: str
    run_id: str
    question_id: str
    gold_sha256: str
    config: dict[str, Any]
    baseline: dict[str, Any]
    status: str
    raw_answer: str
    reason: str
    evidence: tuple[TraceEvidence, ...]
    candidate_evidence: tuple[CandidateEvidence, ...] | None
    error: dict[str, Any] | None


@dataclass(frozen=True)
class ClaimAssessment:
    claim_id: str
    text: str
    support: str


@dataclass(frozen=True)
class CitationAssessment:
    claim_id: str
    citation_id: int
    evidence_sha256: str
    support: str


@dataclass(frozen=True)
class Assessment:
    assessment_id: str
    trace_id: str
    question_id: str
    reviewer_id: str
    reviewed_at: str
    context_sufficient: str
    decision: str
    answer_correct: str
    claims: tuple[ClaimAssessment, ...]
    citations: tuple[CitationAssessment, ...]
    secondary_labels: tuple[str, ...]
    notes: str | None


def _error(context: str, message: str) -> None:
    raise EvaluationError(f"{context}: {message}")


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _error(context, "must be an object")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        _error(context, "must be a list")
    return value


def _nonempty_str(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _error(context, "must be a non-empty string")
    return value


def _optional_str(value: Any, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        _error(context, "must be a string or null")
    return value


def _hex64(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        _error(context, "must be a 64-character hexadecimal SHA-256 digest")
    return value.lower()


def _int(value: Any, context: str, *, nonnegative: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _error(context, "must be an integer")
    if nonnegative and value < 0:
        _error(context, "must be non-negative")
    return value


def _optional_int(value: Any, context: str) -> int | None:
    if value is None:
        return None
    return _int(value, context)


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(context, "must be a number")
    if not math.isfinite(value):
        _error(context, "must be finite")
    return float(value)


def _check_keys(
    value: dict[str, Any],
    required: set[str],
    optional: set[str],
    context: str,
) -> None:
    allowed = required | optional
    unknown = sorted(set(value) - allowed)
    if unknown:
        _error(context, f"unexpected field(s): {', '.join(unknown)}")
    missing = sorted(required - set(value))
    if missing:
        _error(context, f"missing required field(s): {', '.join(missing)}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _text_present(needle: str, haystack: str) -> bool:
    if needle in haystack:
        return True
    needle_norm = needle.strip().rstrip(" .,!?:;").casefold()
    haystack_norm = haystack.strip().rstrip(" .,!?:;").casefold()
    return needle_norm in haystack_norm or needle_norm in haystack.casefold()


def _has_number(text: str) -> bool:
    return any(char.isdigit() for char in text) or any(
        symbol in text for symbol in ("€", "$", "£", "%")
    )


def _secondary_labels(
    *,
    gold,
    trace,
    assessment,
    observed,
    missing_required_ids,
    candidate_available,
    candidate_missing_ids,
    context_label,
    claim_unsupported,
    claim_contradicted,
    claim_items,
    structural_validity,
    syntax_errors,
    citation_does_not_support_ids,
    citation_irrelevant_ids,
    false_abstention,
    false_answer,
    integrity_failed,
) -> list[str]:
    labels: set[str] = set()
    if assessment is not None:
        labels.update(assessment.secondary_labels)

    if integrity_failed:
        pass

    if trace is not None and trace.status == "backend_error":
        labels.add("backend_error")

    if missing_required_ids:
        labels.add("missing_required_evidence")
        if candidate_available and candidate_missing_ids:
            labels.add("candidate_evidence_absent")
        if candidate_available and not candidate_missing_ids:
            labels.add("candidate_present_not_in_context")

    if assessment is not None and false_abstention:
        labels.add("incorrect_abstention")
    if assessment is not None and false_answer:
        labels.add("false_answer")

    if assessment is None and not (trace is not None and trace.status == "backend_error"):
        labels.add("unassessed_label")

    if structural_validity == "invalid":
        labels.add("invalid_citation_syntax")
        if "malformed_brackets" in syntax_errors or "unsupported_syntax" in syntax_errors:
            labels.add("malformed_citation")
        if "missing_citations" in syntax_errors:
            labels.add("missing_citation")
        if "out_of_range" in syntax_errors:
            labels.add("out_of_range_citation")

    if citation_does_not_support_ids or citation_irrelevant_ids:
        labels.add("citation_not_supported")
    if citation_irrelevant_ids:
        labels.add("irrelevant_citation")

    if claim_unsupported > 0:
        labels.add("unsupported_claim")
        for item in claim_items:
            if item.get("support") == "unsupported" and _has_number(item.get("text", "")):
                labels.add("unsupported_numeric")
    if claim_contradicted > 0:
        labels.add("contradicted_claim")
        for item in claim_items:
            if item.get("support") == "contradicted" and _has_number(item.get("text", "")):
                labels.add("contradicted_numeric")

    return sorted(labels)


def _resolve_repo_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "data" / "raw").is_dir():
            return candidate
    return Path.cwd().resolve()


def _resolve_source_path(source: str, repo_root: Path) -> Path:
    path = Path(source)
    if path.is_absolute():
        return path
    if (Path.cwd() / path).exists():
        return Path.cwd() / path
    return repo_root / path


def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvaluationError(f"{path}: {exc}") from exc
    if not text.strip():
        _error(str(path), "file is empty")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _load_jsonl(text, path)
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        result: list[dict[str, Any]] = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                _error(str(path), f"entry {index} must be an object")
            result.append(item)
        return result
    _error(str(path), "top-level JSON value must be an object or list")
    return []


def _load_jsonl(text: str, path: Path) -> list[dict[str, Any]]:
    lines = text.splitlines()
    records: list[dict[str, Any]] = []
    header_version: str | None = None
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            _error(str(path), f"line {line_number}: invalid JSON ({exc.msg})")
        if not isinstance(obj, dict):
            _error(str(path), f"line {line_number}: must be a JSON object")
        version = obj.get("schema_version")
        if (
            header_version is None
            and version is not None
            and set(obj) <= {"schema_version", "metadata"}
        ):
            header_version = str(version)
            continue
        if version is not None and header_version is not None and version != header_version:
            _error(str(path), f"line {line_number}: schema_version mismatch")
        if version is None:
            if header_version is None:
                _error(
                    str(path),
                    f"line {line_number}: missing schema_version "
                    "(add a header line or version field)",
                )
            obj = dict(obj)
            obj["schema_version"] = header_version
        records.append(obj)
    if not records:
        _error(str(path), "JSONL contains no records")
    return records


def parse_gold(document: Any) -> list[GoldQuestion]:
    if not isinstance(document, dict):
        _error("gold", "document must be an object")
    _check_keys(
        document,
        {"schema_version", "dataset_id", "corpus", "questions"},
        set(),
        "gold",
    )
    if document["schema_version"] != SCHEMA_GOLD:
        _error("gold", f"schema_version must be {SCHEMA_GOLD!r}")
    _nonempty_str(document["dataset_id"], "gold.dataset_id")
    corpus_raw = _list(document["corpus"], "gold.corpus")
    if not corpus_raw:
        _error("gold", "corpus must be non-empty")
    corpus: dict[str, GoldDocument] = {}
    seen_doc_ids: set[str] = set()
    seen_sources: set[str] = set()
    for index, raw_doc in enumerate(corpus_raw):
        context = f"gold.corpus[{index}]"
        doc = _mapping(raw_doc, context)
        _check_keys(
            doc,
            {"document_id", "source", "sha256", "provenance"},
            set(),
            context,
        )
        document_id = _nonempty_str(doc["document_id"], f"{context}.document_id")
        if document_id in seen_doc_ids:
            _error(context, f"duplicate document_id {document_id!r}")
        seen_doc_ids.add(document_id)
        source = _nonempty_str(doc["source"], f"{context}.source")
        if source in seen_sources:
            _error(context, f"duplicate source {source!r}")
        seen_sources.add(source)
        sha = _hex64(doc["sha256"], f"{context}.sha256")
        provenance = _nonempty_str(doc["provenance"], f"{context}.provenance")
        corpus[document_id] = GoldDocument(
            document_id=document_id,
            source=source,
            sha256=sha,
            provenance=provenance,
        )
    questions_raw = _list(document["questions"], "gold.questions")
    if not questions_raw:
        _error("gold", "questions must be non-empty")
    questions: list[GoldQuestion] = []
    seen_question_ids: set[str] = set()
    for index, raw_question in enumerate(questions_raw):
        context = f"gold.questions[{index}]"
        q = _mapping(raw_question, context)
        _check_keys(
            q,
            {
                "question_id",
                "question",
                "document_id",
                "answerability",
                "expected_decision",
                "expected_answer",
                "required_evidence",
                "review_status",
            },
            {"notes"},
            context,
        )
        question_id = _nonempty_str(q["question_id"], f"{context}.question_id")
        if question_id in seen_question_ids:
            _error(context, f"duplicate question_id {question_id!r}")
        seen_question_ids.add(question_id)
        question = _nonempty_str(q["question"], f"{context}.question")
        document_id = _nonempty_str(q["document_id"], f"{context}.document_id")
        if document_id not in corpus:
            _error(context, f"unknown document_id {document_id!r}")
        answerability = q["answerability"]
        if answerability not in ANSWERABILITY_VALUES:
            _error(context, "invalid answerability")
        expected_decision = q["expected_decision"]
        if expected_decision not in DECISION_VALUES:
            _error(context, "invalid expected_decision")
        if answerability == "answerable" and expected_decision != "answer":
            _error(context, "answerable questions must expect decision 'answer'")
        if answerability == "unanswerable" and expected_decision != "abstain":
            _error(context, "unanswerable questions must expect decision 'abstain'")
        expected_answer_raw = _mapping(
            q["expected_answer"], f"{context}.expected_answer"
        )
        _check_keys(
            expected_answer_raw,
            {"status", "text", "evidence_ids"},
            set(),
            f"{context}.expected_answer",
        )
        expected_status = _nonempty_str(
            expected_answer_raw["status"], f"{context}.expected_answer.status"
        )
        expected_text_raw = expected_answer_raw["text"]
        if expected_text_raw is None:
            expected_text = None
        else:
            expected_text = _nonempty_str(
                expected_text_raw, f"{context}.expected_answer.text"
            )
        evidence_ids_raw = _list(
            expected_answer_raw["evidence_ids"],
            f"{context}.expected_answer.evidence_ids",
        )
        expected_evidence_ids = tuple(
            _nonempty_str(item, f"{context}.expected_answer.evidence_ids")
            for item in evidence_ids_raw
        )
        expected_answer = ExpectedAnswer(
            status=expected_status,
            text=expected_text,
            evidence_ids=expected_evidence_ids,
        )
        required_raw = _list(q["required_evidence"], f"{context}.required_evidence")
        required: list[RequiredEvidence] = []
        seen_required_ids: set[str] = set()
        for ev_index, raw_ev in enumerate(required_raw):
            ev_context = f"{context}.required_evidence[{ev_index}]"
            ev = _mapping(raw_ev, ev_context)
            _check_keys(
                ev,
                {"evidence_id", "document_id", "chunk_id", "page_index", "quote", "review_status"},
                set(),
                ev_context,
            )
            evidence_id = _nonempty_str(ev["evidence_id"], f"{ev_context}.evidence_id")
            if evidence_id in seen_required_ids:
                _error(ev_context, f"duplicate evidence_id {evidence_id!r}")
            seen_required_ids.add(evidence_id)
            ev_document_id = _nonempty_str(ev["document_id"], f"{ev_context}.document_id")
            if ev_document_id not in corpus:
                _error(ev_context, f"unknown document_id {ev_document_id!r}")
            chunk_id = _int(ev["chunk_id"], f"{ev_context}.chunk_id")
            page_index = _optional_int(ev["page_index"], f"{ev_context}.page_index")
            quote = _nonempty_str(ev["quote"], f"{ev_context}.quote")
            review_status = ev["review_status"]
            if review_status not in GOLD_REVIEW_STATUSES:
                _error(ev_context, "invalid review_status")
            required.append(
                RequiredEvidence(
                    evidence_id=evidence_id,
                    document_id=ev_document_id,
                    chunk_id=chunk_id,
                    page_index=page_index,
                    quote=quote,
                    review_status=review_status,
                )
            )
        review_status = q["review_status"]
        if review_status not in GOLD_REVIEW_STATUSES:
            _error(context, "invalid review_status")
        notes = _optional_str(q.get("notes"), f"{context}.notes")
        questions.append(
            GoldQuestion(
                question_id=question_id,
                question=question,
                document_id=document_id,
                answerability=answerability,
                expected_decision=expected_decision,
                expected_answer=expected_answer,
                required_evidence=tuple(required),
                review_status=review_status,
                notes=notes,
            )
        )
    return questions


def _parse_trace_evidence(record: Any, context: str) -> TraceEvidence:
    item = _mapping(record, context)
    _check_keys(
        item,
        {
            "citation_id",
            "kind",
            "document_id",
            "source",
            "chunk_id",
            "page_index",
            "score",
            "text",
            "text_sha256",
        },
        set(),
        context,
    )
    citation_id = _int(item["citation_id"], f"{context}.citation_id")
    if citation_id < 1:
        _error(context, "citation_id must be positive")
    kind = _nonempty_str(item["kind"], f"{context}.kind")
    document_id = _nonempty_str(item["document_id"], f"{context}.document_id")
    source = _nonempty_str(item["source"], f"{context}.source")
    chunk_id = _int(item["chunk_id"], f"{context}.chunk_id")
    page_index = _optional_int(item["page_index"], f"{context}.page_index")
    score = _number(item["score"], f"{context}.score")
    text = _nonempty_str(item["text"], f"{context}.text")
    text_sha = _hex64(item["text_sha256"], f"{context}.text_sha256")
    if sha256_text(text) != text_sha:
        _error(context, "text SHA-256 does not match text_sha256")
    return TraceEvidence(
        citation_id=citation_id,
        kind=kind,
        document_id=document_id,
        source=source,
        chunk_id=chunk_id,
        page_index=page_index,
        score=score,
        text=text,
        text_sha256=text_sha,
    )


def _parse_candidate(record: Any, context: str) -> CandidateEvidence:
    item = _mapping(record, context)
    _check_keys(item, {"document_id", "chunk_id", "page_index"}, set(), context)
    document_id = _nonempty_str(item["document_id"], f"{context}.document_id")
    chunk_id = _int(item["chunk_id"], f"{context}.chunk_id")
    page_index = _optional_int(item["page_index"], f"{context}.page_index")
    return CandidateEvidence(
        document_id=document_id,
        chunk_id=chunk_id,
        page_index=page_index,
    )


def _parse_trace(record: Any) -> Trace:
    context = "trace"
    item = _mapping(record, context)
    _check_keys(
        item,
        {
            "schema_version",
            "trace_id",
            "run_id",
            "question_id",
            "gold_sha256",
            "config",
            "baseline",
            "result",
            "evidence",
            "candidate_evidence",
            "error",
        },
        set(),
        context,
    )
    if item["schema_version"] != SCHEMA_TRACE:
        _error(context, f"schema_version must be {SCHEMA_TRACE!r}")
    trace_id = _nonempty_str(item["trace_id"], "trace.trace_id")
    run_id = _nonempty_str(item["run_id"], "trace.run_id")
    question_id = _nonempty_str(item["question_id"], "trace.question_id")
    gold_sha256 = _hex64(item["gold_sha256"], "trace.gold_sha256")
    config = _mapping(item["config"], "trace.config")
    baseline = _mapping(item["baseline"], "trace.baseline")
    result = _mapping(item["result"], "trace.result")
    _check_keys(result, {"status", "raw_answer", "reason"}, set(), "trace.result")
    status = result["status"]
    if status not in ANSWER_STATUSES:
        _error("trace.result", "invalid status")
    raw_answer = result["raw_answer"]
    if not isinstance(raw_answer, str):
        _error("trace.result", "raw_answer must be a string")
    reason = result["reason"]
    if not isinstance(reason, str):
        _error("trace.result", "reason must be a string")
    evidence_raw = _list(item["evidence"], "trace.evidence")
    evidence: list[TraceEvidence] = []
    seen_citation_ids: set[int] = set()
    for index, raw_ev in enumerate(evidence_raw):
        ev = _parse_trace_evidence(raw_ev, f"trace.evidence[{index}]")
        if ev.citation_id in seen_citation_ids:
            _error(
                f"trace.evidence[{index}]",
                f"duplicate citation_id {ev.citation_id!r}",
            )
        seen_citation_ids.add(ev.citation_id)
        evidence.append(ev)
    citation_ids = sorted(ev.citation_id for ev in evidence)
    if citation_ids != list(range(1, len(evidence) + 1)):
        _error("trace.evidence", "citation_id values must be contiguous from 1")
    candidate_raw = item["candidate_evidence"]
    if candidate_raw is None:
        candidate_evidence = None
    else:
        candidate_list = _list(candidate_raw, "trace.candidate_evidence")
        candidate_evidence = []
        for index, raw_candidate in enumerate(candidate_list):
            candidate_evidence.append(
                _parse_candidate(raw_candidate, f"trace.candidate_evidence[{index}]")
            )
        candidate_evidence = tuple(candidate_evidence)
    error = item["error"]
    if error is None:
        error = None
    elif isinstance(error, str):
        error = {"message": error}
    else:
        error = _mapping(error, "trace.error")
    return Trace(
        trace_id=trace_id,
        run_id=run_id,
        question_id=question_id,
        gold_sha256=gold_sha256,
        config=config,
        baseline=baseline,
        status=status,
        raw_answer=raw_answer,
        reason=reason,
        evidence=tuple(evidence),
        candidate_evidence=candidate_evidence,
        error=error,
    )


def parse_traces(records: Any) -> list[Trace]:
    if not isinstance(records, list):
        _error("traces", "must be a list")
    traces: list[Trace] = []
    seen_trace_ids: set[str] = set()
    seen_question_ids: set[str] = set()
    for record in records:
        trace = _parse_trace(record)
        if trace.trace_id in seen_trace_ids:
            _error("traces", f"duplicate trace_id {trace.trace_id!r}")
        if trace.question_id in seen_question_ids:
            _error("traces", f"duplicate question_id {trace.question_id!r}")
        seen_trace_ids.add(trace.trace_id)
        seen_question_ids.add(trace.question_id)
        traces.append(trace)
    return traces


def _parse_claim(record: Any, context: str) -> ClaimAssessment:
    item = _mapping(record, context)
    _check_keys(item, {"claim_id", "text", "support"}, set(), context)
    claim_id = _nonempty_str(item["claim_id"], f"{context}.claim_id")
    text = _nonempty_str(item["text"], f"{context}.text")
    support = item["support"]
    if support not in CLAIM_SUPPORT_LABELS:
        _error(context, "invalid claim support")
    return ClaimAssessment(claim_id=claim_id, text=text, support=support)


def _parse_citation(record: Any, context: str) -> CitationAssessment:
    item = _mapping(record, context)
    _check_keys(
        item,
        {"claim_id", "citation_id", "evidence_sha256", "support"},
        set(),
        context,
    )
    claim_id = _nonempty_str(item["claim_id"], f"{context}.claim_id")
    citation_id = _int(item["citation_id"], f"{context}.citation_id")
    if citation_id < 1:
        _error(context, "citation_id must be positive")
    evidence_sha = _hex64(item["evidence_sha256"], f"{context}.evidence_sha256")
    support = item["support"]
    if support not in CITATION_SUPPORT_LABELS:
        _error(context, "invalid citation support")
    return CitationAssessment(
        claim_id=claim_id,
        citation_id=citation_id,
        evidence_sha256=evidence_sha,
        support=support,
    )


def parse_assessments(records: Any) -> list[Assessment]:
    if not isinstance(records, list):
        _error("assessments", "must be a list")
    assessments: list[Assessment] = []
    seen_assessment_ids: set[str] = set()
    seen_trace_ids: set[str] = set()
    seen_question_ids: set[str] = set()
    for index, record in enumerate(records):
        context = f"assessment {index}"
        item = _mapping(record, context)
        _check_keys(
            item,
            {
                "schema_version",
                "assessment_id",
                "trace_id",
                "question_id",
                "reviewer_id",
                "reviewed_at",
                "context_sufficient",
                "decision",
                "answer_correct",
                "claims",
                "citations",
                "secondary_labels",
                "notes",
            },
            set(),
            context,
        )
        if item["schema_version"] != SCHEMA_ASSESSMENT:
            _error(context, f"schema_version must be {SCHEMA_ASSESSMENT!r}")
        assessment_id = _nonempty_str(item["assessment_id"], f"{context}.assessment_id")
        if assessment_id in seen_assessment_ids:
            _error(context, f"duplicate assessment_id {assessment_id!r}")
        seen_assessment_ids.add(assessment_id)
        trace_id = _nonempty_str(item["trace_id"], f"{context}.trace_id")
        if trace_id in seen_trace_ids:
            _error(context, f"duplicate trace_id {trace_id!r}")
        seen_trace_ids.add(trace_id)
        question_id = _nonempty_str(item["question_id"], f"{context}.question_id")
        if question_id in seen_question_ids:
            _error(context, f"duplicate question_id {question_id!r}")
        seen_question_ids.add(question_id)
        reviewer_id = _nonempty_str(item["reviewer_id"], f"{context}.reviewer_id")
        reviewed_at = _nonempty_str(item["reviewed_at"], f"{context}.reviewed_at")
        context_sufficient = item["context_sufficient"]
        if context_sufficient not in CONTEXT_LABELS:
            _error(context, "invalid context_sufficient")
        decision = item["decision"]
        if decision not in ASSESSMENT_DECISION_VALUES:
            _error(context, "invalid decision")
        answer_correct = item["answer_correct"]
        if answer_correct not in ANSWER_CORRECT_LABELS:
            _error(context, "invalid answer_correct")
        claims_raw = _list(item["claims"], f"{context}.claims")
        claims: list[ClaimAssessment] = []
        seen_claim_ids: set[str] = set()
        for claim_index, raw_claim in enumerate(claims_raw):
            claim = _parse_claim(raw_claim, f"{context}.claims[{claim_index}]")
            if claim.claim_id in seen_claim_ids:
                _error(
                    f"{context}.claims[{claim_index}]",
                    f"duplicate claim_id {claim.claim_id!r}",
                )
            seen_claim_ids.add(claim.claim_id)
            claims.append(claim)
        citations_raw = _list(item["citations"], f"{context}.citations")
        citations: list[CitationAssessment] = []
        seen_pairs: set[tuple[str, int]] = set()
        for citation_index, raw_citation in enumerate(citations_raw):
            citation = _parse_citation(
                raw_citation, f"{context}.citations[{citation_index}]"
            )
            pair = (citation.claim_id, citation.citation_id)
            if pair in seen_pairs:
                _error(
                    f"{context}.citations[{citation_index}]",
                    f"duplicate claim/citation pair {pair!r}",
                )
            seen_pairs.add(pair)
            citations.append(citation)
        secondary_labels = _list(item["secondary_labels"], f"{context}.secondary_labels")
        secondary_tuple = tuple(
            _nonempty_str(label, f"{context}.secondary_labels")
            for label in secondary_labels
        )
        notes = _optional_str(item["notes"], f"{context}.notes")
        assessments.append(
            Assessment(
                assessment_id=assessment_id,
                trace_id=trace_id,
                question_id=question_id,
                reviewer_id=reviewer_id,
                reviewed_at=reviewed_at,
                context_sufficient=context_sufficient,
                decision=decision,
                answer_correct=answer_correct,
                claims=tuple(claims),
                citations=tuple(citations),
                secondary_labels=tuple(sorted(set(secondary_tuple))),
                notes=notes,
            )
        )
    return assessments


def load_gold(path: str | Path) -> list[GoldQuestion]:
    path = Path(path)
    raw = _load_json_or_jsonl(path)
    if len(raw) != 1 or not isinstance(raw[0], dict):
        _error(str(path), "gold file must contain one JSON object")
    document = raw[0]
    repo_root = _resolve_repo_root(path)
    # Verify corpus hashes and quotes when files are available.
    corpus = document.get("corpus")
    if isinstance(corpus, list):
        for entry in corpus:
            if not isinstance(entry, dict):
                continue
            source = entry.get("source")
            sha = entry.get("sha256")
            if isinstance(source, str) and isinstance(sha, str):
                resolved = _resolve_source_path(source, repo_root)
                if resolved.exists():
                    actual = sha256_file(resolved)
                    if actual != sha.lower():
                        _error(str(path), f"corpus hash mismatch for {source}")
    questions = document.get("questions")
    if isinstance(questions, list):
        for q in questions:
            if not isinstance(q, dict):
                continue
            required = q.get("required_evidence")
            if isinstance(required, list):
                # Map document_id to source for quote verification.
                doc_sources = {}
                if isinstance(corpus, list):
                    for entry in corpus:
                        if isinstance(entry, dict):
                            doc_sources[entry.get("document_id")] = entry.get("source")
                for ev in required:
                    if not isinstance(ev, dict):
                        continue
                    doc_id = ev.get("document_id")
                    quote = ev.get("quote")
                    source = doc_sources.get(doc_id)
                    if isinstance(source, str) and isinstance(quote, str):
                        resolved = _resolve_source_path(source, repo_root)
                        if resolved.exists():
                            text = resolved.read_text(encoding="utf-8")
                            if quote not in text:
                                _error(str(path), f"required evidence quote not found in {source}")
    return parse_gold(document)


def load_traces(path: str | Path) -> list[Trace]:
    records = _load_json_or_jsonl(Path(path))
    # _load_json_or_jsonl returns [payload] for a JSON object; unwrap if needed.
    if len(records) == 1 and isinstance(records[0], dict) and "traces" in records[0]:
        records = records[0]["traces"]
    if len(records) == 1 and isinstance(records[0], dict) and "schema_version" in records[0] and "trace_id" not in records[0]:
        # header-only object; shouldn't happen
        pass
    # Support JSON wrapper with "traces".
    if isinstance(records, dict):
        records = records.get("traces", [])
    return parse_traces(records)


def load_assessments(path: str | Path) -> list[Assessment]:
    records = _load_json_or_jsonl(Path(path))
    if len(records) == 1 and isinstance(records[0], dict) and "assessments" in records[0]:
        records = records[0]["assessments"]
    if isinstance(records, dict):
        records = records.get("assessments", [])
    return parse_assessments(records)


def metric(numerator: int, denominator: int) -> dict[str, Any]:
    if denominator == 0:
        return {
            "numerator": numerator,
            "denominator": denominator,
            "value": None,
            "status": "not_applicable",
        }
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": round(numerator / denominator, 6),
        "status": "assessed",
    }


def _observed_decision(status: str) -> str | None:
    if status == "produced":
        return "answered"
    if status == "invalid_output":
        return "answered"
    if status in ABSTENTION_STATUSES:
        return "abstained"
    return None


def _parse_citations(text: str, evidence_count: int) -> tuple[list[int], list[str]]:
    errors: set[str] = set()
    groups = re.findall(r"\[([^\[\]]*)\]", text)
    content = re.sub(r"\[[^\[\]]*\]", "", text)
    if not any(char.isalnum() for char in content):
        errors.add("no_substantive_content")
    if "[" in content or "]" in content:
        errors.add("malformed_brackets")
    if not groups:
        errors.add("missing_citations")
    ids: list[int] = []
    for group in groups:
        if not re.fullmatch(r"\s*[0-9]+(?:\s*,\s*[0-9]+)*\s*", group):
            errors.add("unsupported_syntax")
            continue
        for token in group.split(","):
            number = int(token.strip())
            if number < 1 or number > evidence_count:
                errors.add("out_of_range")
            else:
                ids.append(number)
    ordered = [error for error in CITATION_ORDER if error in errors]
    return ids, ordered


def _covered_required_ids(
    required: tuple[RequiredEvidence, ...],
    evidence: tuple[TraceEvidence, ...],
) -> set[str]:
    covered: set[str] = set()
    for req in required:
        for ev in evidence:
            if ev.document_id == req.document_id and ev.chunk_id == req.chunk_id:
                covered.add(req.evidence_id)
                break
    return covered


def _covered_candidate_ids(
    required: tuple[RequiredEvidence, ...],
    candidate: tuple[CandidateEvidence, ...] | None,
) -> set[str]:
    if candidate is None:
        return set()
    covered: set[str] = set()
    for req in required:
        for cand in candidate:
            if cand.document_id == req.document_id and cand.chunk_id == req.chunk_id:
                covered.add(req.evidence_id)
                break
    return covered


def _integrity_errors(
    gold: GoldQuestion,
    trace: Trace,
    assessment: Assessment | None,
    gold_sha256: str | None,
) -> list[str]:
    errors: list[str] = []

    if gold_sha256 is not None and trace.gold_sha256 != gold_sha256:
        errors.append("gold_sha256 mismatch")

    docs_by_id = None  # not needed; corpus not passed here

    for evidence in trace.evidence:
        if sha256_text(evidence.text) != evidence.text_sha256:
            errors.append(
                f"evidence {evidence.citation_id}: text SHA-256 mismatch"
            )

    if assessment is not None:
        trace_evidence_by_id = {ev.citation_id: ev for ev in trace.evidence}
        claim_ids = {claim.claim_id for claim in assessment.claims}
        if trace.status in {"produced", "invalid_output"}:
            for claim in assessment.claims:
                if not _text_present(claim.text, trace.raw_answer):
                    errors.append(f"claim {claim.claim_id}: text not found in generated output")
        for citation in assessment.citations:
            if citation.claim_id not in claim_ids:
                errors.append(
                    f"citation ({citation.claim_id}, {citation.citation_id}): unknown claim_id"
                )
            evidence = trace_evidence_by_id.get(citation.citation_id)
            if evidence is None:
                errors.append(
                    f"citation {citation.citation_id}: citation_id not in trace evidence"
                )
            else:
                if citation.evidence_sha256 != evidence.text_sha256:
                    errors.append(
                        f"citation {citation.citation_id}: evidence snapshot mismatch"
                    )
            # If structural syntax valid, citation should be cited in answer.
        cited_ids = set(_parse_citations(trace.raw_answer, len(trace.evidence))[0])
        if cited_ids and _parse_citations(trace.raw_answer, len(trace.evidence))[1] == []:
            for citation in assessment.citations:
                if citation.citation_id not in cited_ids:
                    errors.append(
                        f"citation {citation.citation_id}: not cited in generated output"
                    )

    errors = sorted(set(errors))
    return errors


def _evaluate_case(
    gold: GoldQuestion,
    trace: Trace | None,
    assessment: Assessment | None,
    gold_sha256: str | None,
) -> dict[str, Any]:
    integrity = [] if trace is None else _integrity_errors(gold, trace, assessment, gold_sha256)
    integrity_failed = bool(integrity)

    trace_id = trace.trace_id if trace is not None else None
    assessment_id = assessment.assessment_id if assessment is not None else None
    trace_status = trace.status if trace is not None else None
    generator_type = None
    if trace is not None:
        generator_type = "extractive" if trace.config.get("generator") == "extractive" else "generated"
    observed = _observed_decision(trace_status) if trace_status is not None else None

    gold_reviewed = gold.review_status == "reviewed"
    assessment_present = assessment is not None
    trace_exists = trace is not None

    # Decision metrics.
    decision_correct: bool | None = None
    false_abstention: bool | None = None
    false_answer: bool | None = None
    correct_abstention: bool | None = None
    if gold_reviewed and trace_exists and observed in {"answered", "abstained"}:
        if gold.expected_decision == "answer":
            decision_correct = observed == "answered"
            false_abstention = observed == "abstained"
        else:
            decision_correct = observed == "abstained"
            false_answer = observed == "answered"
            correct_abstention = observed == "abstained"

    # Evidence coverage.
    required_count = len(gold.required_evidence)
    required_ids = {req.evidence_id for req in gold.required_evidence}
    covered_ids: set[str] = set()
    candidate_covered_ids: set[str] = set()
    candidate_available = False
    missing_required_ids: list[str] = []
    candidate_missing_ids: list[str] = []
    if trace_exists:
        covered_ids = _covered_required_ids(gold.required_evidence, trace.evidence)
        missing_required_ids = sorted(required_ids - covered_ids)
        if trace.candidate_evidence is not None:
            candidate_available = True
            candidate_covered_ids = _covered_candidate_ids(
                gold.required_evidence, trace.candidate_evidence
            )
            candidate_missing_ids = sorted(required_ids - candidate_covered_ids)

    if required_count == 0:
        coverage_metric = metric(0, 0)
        coverage_metric["status"] = "not_applicable"
        candidate_coverage_metric = metric(0, 0)
        candidate_coverage_metric["status"] = "not_applicable"
    else:
        coverage_metric = metric(len(covered_ids), required_count) if trace_exists else None
        candidate_coverage_metric = (
            metric(len(candidate_covered_ids), required_count)
            if trace_exists and candidate_available
            else None
        )

    # Assessment labels.
    context_label = None
    correctness_label = None
    if assessment is not None:
        context_label = assessment.context_sufficient
        correctness_label = assessment.answer_correct
        if gold.expected_decision != "answer" and context_label == "not_applicable":
            pass

    # Claims.
    claim_items: list[dict[str, Any]] = []
    claim_reviewed = 0
    claim_supported = 0
    claim_unsupported = 0
    claim_contradicted = 0
    claim_unassessed = 0
    if assessment is not None:
        for claim in assessment.claims:
            claim_items.append(
                {
                    "claim_id": claim.claim_id,
                    "text": claim.text,
                    "support": claim.support,
                }
            )
            if claim.support == "supported":
                claim_reviewed += 1
                claim_supported += 1
            elif claim.support == "unsupported":
                claim_reviewed += 1
                claim_unsupported += 1
            elif claim.support == "contradicted":
                claim_reviewed += 1
                claim_contradicted += 1
            else:
                claim_unassessed += 1

    # Citations.
    generated = generator_type == "generated"
    citation_requirement = "required" if generated else "not_applicable"
    cited_ids: list[int] = []
    syntax_errors: list[str] = []
    structural_validity = "not_applicable"
    if generated and trace_exists and trace.status in {"produced", "invalid_output"}:
        cited_ids, syntax_errors = _parse_citations(
            trace.raw_answer, len(trace.evidence)
        )
        structural_validity = "valid" if not syntax_errors else "invalid"
    elif generated and trace_exists:
        structural_validity = "not_applicable"
    unique_cited = sorted(set(cited_ids))

    # Citation support.
    citation_reviewed = 0
    citation_supported = 0
    citation_does_not_support_ids: list[int] = []
    citation_irrelevant_ids: list[int] = []
    citation_unassessed_ids: list[int] = []
    citation_pairs: Any = "not_applicable"
    if generated and assessment is not None and structural_validity == "valid":
        by_pair = {(c.claim_id, c.citation_id): c for c in assessment.citations}
        reviewed_count = len(assessment.citations)
        supported_count = sum(
            1 for c in assessment.citations if c.support == "supports"
        )
        citation_pairs = {"reviewed": reviewed_count, "supported": supported_count}
        trace_evidence_by_id = {ev.citation_id: ev for ev in trace.evidence}
        for citation in assessment.citations:
            evidence = trace_evidence_by_id.get(citation.citation_id)
            if citation.support == "supports":
                citation_reviewed += 1
                citation_supported += 1
            elif citation.support == "does_not_support":
                citation_reviewed += 1
                citation_does_not_support_ids.append(citation.citation_id)
            elif citation.support == "irrelevant":
                citation_reviewed += 1
                citation_irrelevant_ids.append(citation.citation_id)
            else:
                citation_unassessed_ids.append(citation.citation_id)
    elif generated and assessment is not None and structural_validity == "invalid":
        citation_pairs = "not_applicable"
    elif not generated:
        citation_pairs = "not_applicable"

    # Citation structural metric denominator handled in aggregates.

    # Failure detection.
    failure_reasons: list[str] = []
    unassessed_reasons: list[str] = []
    stage_candidates: list[str] = []
    evidence_stage: str | None = None

    if integrity_failed:
        failure_reasons.append("integrity_failure")

    if not trace_exists:
        unassessed_reasons.append("missing_trace")
    if not gold_reviewed:
        unassessed_reasons.append("gold_unreviewed")
    if assessment is None:
        unassessed_reasons.append("missing_assessment")

    if trace_exists and trace.status == "backend_error":
        failure_reasons.append("backend_error")
        stage_candidates.append("backend")

    # Required evidence stage attribution.
    if trace_exists and missing_required_ids:
        if candidate_available:
            if candidate_missing_ids:
                evidence_stage = "retrieval"
                failure_reasons.append("retrieval_required_evidence_missing")
            else:
                evidence_stage = "context_assembly"
                failure_reasons.append("context_assembly_required_evidence_dropped")
        else:
            evidence_stage = "unknown"
            failure_reasons.append("required_evidence_missing_unattributed")
        stage_candidates.append(evidence_stage)

    # Context sufficiency.
    if gold.expected_decision == "answer" and context_label == "insufficient":
        failure_reasons.append("context_insufficient")
        if evidence_stage is None:
            stage_candidates.append("unknown")
    if context_label is None and assessment is not None:
        unassessed_reasons.append("context_sufficiency_unassessed")
    if assessment is None:
        pass
    elif assessment.context_sufficient == "unknown":
        unassessed_reasons.append("context_sufficiency_unassessed")

    # Decision failures.
    if assessment is not None:
        if gold.expected_decision == "answer" and observed == "abstained":
            failure_reasons.append("false_abstention")
            stage_candidates.append("generation")
        if gold.expected_decision == "abstain" and observed == "answered":
            failure_reasons.append("false_answer")
            stage_candidates.append("generation")

    # Status-based failures.
    if trace_exists and trace.status == "invalid_output":
        if syntax_errors:
            failure_reasons.append("invalid_citation_structure")
            stage_candidates.append("citation")
        else:
            failure_reasons.append("invalid_output")
            stage_candidates.append("generation")

    # Answer correctness.
    if gold.expected_decision == "answer" and correctness_label == "incorrect":
        failure_reasons.append("answer_incorrect")
        stage_candidates.append("generation")

    # Claims.
    if claim_unsupported > 0:
        failure_reasons.append("unsupported_claim")
        stage_candidates.append("generation")
    if claim_contradicted > 0:
        failure_reasons.append("contradicted_claim")
        stage_candidates.append("generation")

    # Citation structural.
    if generated and structural_validity == "invalid":
        failure_reasons.append("citation_structural_invalid")
        stage_candidates.append("citation")

    # Citation support.
    citation_support_failed = False
    if generated and structural_validity == "valid":
        if citation_does_not_support_ids:
            citation_support_failed = True
            failure_reasons.append("citation_does_not_support")
        if citation_irrelevant_ids:
            citation_support_failed = True
            failure_reasons.append("citation_irrelevant")
    if citation_support_failed:
        stage_candidates.append("citation")

    # Unassessed reasons.
    if gold.expected_decision == "answer":
        if correctness_label is None and assessment is not None:
            unassessed_reasons.append("answer_correctness_unassessed")
        if assessment is not None and not assessment.claims:
            unassessed_reasons.append("claims_unassessed")
        if claim_unassessed > 0:
            unassessed_reasons.append("claims_unassessed")
        if generated and structural_validity == "valid" and citation_unassessed_ids:
            unassessed_reasons.append("citation_support_unassessed")
        if generated and structural_validity == "valid" and unique_cited and citation_reviewed == 0 and assessment is not None:
            unassessed_reasons.append("citation_support_unassessed")
    if assessment is None:
        pass

    failure_reasons = sorted(set(failure_reasons))
    unassessed_reasons = sorted(set(unassessed_reasons))

    prefer_unknown = context_label == "insufficient" and not missing_required_ids
    # Stage selection.
    stage: str | None = None
    if integrity_failed:
        stage = "unknown"
    elif failure_reasons:
        if "backend" in stage_candidates:
            stage = "backend"
        elif "retrieval" in stage_candidates:
            stage = "retrieval"
        elif "context_assembly" in stage_candidates:
            stage = "context_assembly"
        elif evidence_stage == "unknown":
            stage = "unknown"
        elif prefer_unknown:
            stage = "unknown"
        elif "generation" in stage_candidates:
            stage = "generation"
        elif "citation" in stage_candidates:
            stage = "citation"
        elif "unknown" in stage_candidates:
            stage = "unknown"
        else:
            stage = "unknown"
    else:
        stage = None

    # Success checks.
    success = False
    if (
        gold_reviewed
        and trace_exists
        and assessment is not None
        and not integrity_failed
        and not failure_reasons
        and not unassessed_reasons
    ):
        if gold.expected_decision == "answer":
            if (
                trace.status == "produced"
                and observed == "answered"
                and decision_correct is True
                and not missing_required_ids
                and context_label == "sufficient"
                and correctness_label == "correct"
                and claim_reviewed > 0
                and claim_unassessed == 0
                and claim_unsupported == 0
                and claim_contradicted == 0
            ):
                if generator_type == "extractive":
                    success = True
                else:
                    if (
                        structural_validity == "valid"
                        and len(unique_cited) > 0
                        and citation_reviewed == len(assessment.citations)
                        and citation_supported == len(assessment.citations)
                        and not citation_does_not_support_ids
                        and not citation_irrelevant_ids
                        and not citation_unassessed_ids
                    ):
                        success = True
        else:
            if (
                trace.status in ABSTENTION_STATUSES
                and observed == "abstained"
                and decision_correct is True
                and context_label in {"sufficient", "insufficient", "not_applicable"}
            ):
                success = True

    if success:
        outcome = "success"
        stage = None
    elif integrity_failed:
        outcome = "invalid"
    elif failure_reasons:
        outcome = "failure"
    elif unassessed_reasons:
        outcome = "unassessed"
    else:
        outcome = "unassessed"
        if not unassessed_reasons:
            unassessed_reasons = ["success_criteria_not_met"]
            unassessed_reasons = sorted(set(unassessed_reasons))

    if outcome in {"unassessed", "invalid"} and stage is None:
        stage = "unknown"
    if outcome == "success":
        stage = None

    abstention_error = "not_assessed"
    if assessment is not None:
        if false_abstention:
            abstention_error = "false_abstention"
        elif false_answer:
            abstention_error = "false_answer"
        else:
            abstention_error = "none"

    coverage_complete = required_count > 0 and len(covered_ids) == required_count
    coverage_complete_but_context_insufficient = bool(
        coverage_complete and context_label == "insufficient"
    )

    case = {
        "question_id": gold.question_id,
        "trace_id": trace_id,
        "assessment_id": assessment_id,
        "gold_review_status": gold.review_status,
        "answerability": gold.answerability,
        "expected_decision": gold.expected_decision,
        "observed_decision": observed,
        "trace_status": trace_status,
        "generator_type": generator_type,
        "outcome": outcome,
        "failure_stage": stage,
        "failure_reasons": failure_reasons,
        "unassessed_reasons": unassessed_reasons,
        "integrity": {
            "status": "failed" if integrity_failed else "ok",
            "errors": integrity,
        },
        "decision": {
            "expected": gold.expected_decision,
            "observed": observed,
            "correct": decision_correct,
            "false_abstention": false_abstention,
            "false_answer": false_answer,
            "correct_abstention": correct_abstention,
        },
        "evidence": {
            "required_count": required_count,
            "covered_count": len(covered_ids),
            "coverage": coverage_metric,
            "missing_required_evidence_ids": missing_required_ids,
            "candidate_evidence": "available" if candidate_available else "unavailable",
            "candidate_required_coverage": candidate_coverage_metric,
            "missing_from_candidate_evidence_ids": candidate_missing_ids,
        },
        "context_sufficiency": context_label,
        "answer_correctness": correctness_label,
        "claims": None if assessment is None else {
            "items": claim_items,
            "reviewed_count": claim_reviewed,
            "supported_count": claim_supported,
            "unsupported_count": claim_unsupported,
            "contradicted_count": claim_contradicted,
            "unassessed_count": claim_unassessed,
            "faithfulness": metric(claim_supported, claim_reviewed),
        },
        "citations": {
            "requirement": citation_requirement,
            "structural_validity": structural_validity,
            "syntax_errors": syntax_errors,
            "cited_ids": cited_ids,
            "reviewed_count": citation_reviewed,
            "supported_count": citation_supported,
            "does_not_support_ids": citation_does_not_support_ids,
            "irrelevant_ids": citation_irrelevant_ids,
            "unassessed_ids": citation_unassessed_ids,
            "support": metric(citation_supported, citation_reviewed) if citation_reviewed or (generated and assessment is not None and structural_validity == "valid") else metric(0, 0),
            "citation_pairs": citation_pairs,
        },
        "secondary_labels": _secondary_labels(
        gold=gold,
        trace=trace,
        assessment=assessment,
        observed=observed,
        missing_required_ids=missing_required_ids,
        candidate_available=candidate_available,
        candidate_missing_ids=candidate_missing_ids,
        context_label=context_label,
        claim_unsupported=claim_unsupported,
        claim_contradicted=claim_contradicted,
        claim_items=claim_items,
        structural_validity=structural_validity,
        syntax_errors=syntax_errors,
        citation_does_not_support_ids=citation_does_not_support_ids,
        citation_irrelevant_ids=citation_irrelevant_ids,
        false_abstention=false_abstention,
        false_answer=false_answer,
        integrity_failed=integrity_failed,
    ),
        "coverage_complete_but_context_insufficient": coverage_complete_but_context_insufficient,
        "abstention_error": abstention_error,
    }
    # Fix support metric status for zero denominators.
    if case["citations"]["reviewed_count"] == 0:
        case["citations"]["support"] = metric(0, 0)
        case["citations"]["support"]["status"] = "not_applicable"
    return case


def evaluate_answer_quality(
    gold_questions: list[GoldQuestion],
    traces: list[Trace],
    assessments: list[Assessment],
    gold_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(gold_questions, list) or not all(isinstance(item, GoldQuestion) for item in gold_questions):
        raise EvaluationError("gold_questions must be a list of GoldQuestion")
    if not isinstance(traces, list) or not all(isinstance(item, Trace) for item in traces):
        raise EvaluationError("traces must be a list of Trace")
    if not isinstance(assessments, list) or not all(isinstance(item, Assessment) for item in assessments):
        raise EvaluationError("assessments must be a list of Assessment")
    if not gold_questions:
        raise EvaluationError("gold questions must be non-empty")

    gold_by_id = {}
    for gold in gold_questions:
        if gold.question_id in gold_by_id:
            raise EvaluationError(f"duplicate gold question_id {gold.question_id!r}")
        gold_by_id[gold.question_id] = gold

    trace_by_question = {}
    trace_by_id = {}
    for trace in traces:
        if trace.trace_id in trace_by_id:
            raise EvaluationError(f"duplicate trace_id {trace.trace_id!r}")
        trace_by_id[trace.trace_id] = trace
        if trace.question_id in trace_by_question:
            raise EvaluationError(f"duplicate trace question_id {trace.question_id!r}")
        if trace.question_id not in gold_by_id:
            raise EvaluationError(f"trace references unknown question_id {trace.question_id!r}")
        trace_by_question[trace.question_id] = trace

    assessment_by_question = {}
    assessment_by_trace = {}
    for assessment in assessments:
        if assessment.assessment_id in assessment_by_trace and False:
            pass
        if assessment.question_id in assessment_by_question:
            raise EvaluationError(f"duplicate assessment question_id {assessment.question_id!r}")
        if assessment.question_id not in gold_by_id:
            raise EvaluationError(f"assessment references unknown question_id {assessment.question_id!r}")
        if assessment.trace_id not in trace_by_id:
            raise EvaluationError(f"assessment references unknown trace_id {assessment.trace_id!r}")
        assessment_by_question[assessment.question_id] = assessment
        assessment_by_trace[assessment.trace_id] = assessment

    cases = []
    for gold in sorted(gold_questions, key=lambda item: item.question_id):
        trace = trace_by_question.get(gold.question_id)
        assessment = assessment_by_question.get(gold.question_id)
        cases.append(_evaluate_case(gold, trace, assessment, gold_sha256))

    total = len(cases)
    outcome_counts = {"success": 0, "failure": 0, "unassessed": 0, "invalid": 0}
    for case in cases:
        outcome_counts[case["outcome"]] += 1

    trace_coverage_num = sum(1 for case in cases if case["trace_id"] is not None)
    assessment_coverage_num = sum(1 for case in cases if case["assessment_id"] is not None)

    integrity_den = sum(1 for case in cases if case["trace_id"] is not None)
    integrity_num = sum(1 for case in cases if case["integrity"]["status"] == "failed")

    decision_num = 0
    decision_den = 0
    false_abstention_num = 0
    false_abstention_den = 0
    false_answer_num = 0
    false_answer_den = 0
    correct_abstention_num = 0
    correct_abstention_den = 0
    for case in cases:
        if case["gold_review_status"] != "reviewed":
            continue
        if case["trace_id"] is None:
            continue
        if case["integrity"]["status"] == "failed":
            continue
        observed = case["observed_decision"]
        if observed in {"answered", "abstained"}:
            decision_den += 1
            if case["decision"]["correct"] is True:
                decision_num += 1
        if case["expected_decision"] == "answer" and observed in {"answered", "abstained"}:
            false_abstention_den += 1
            if case["decision"]["false_abstention"] is True:
                false_abstention_num += 1
        if case["expected_decision"] == "abstain" and observed in {"answered", "abstained"}:
            false_answer_den += 1
            if case["decision"]["false_answer"] is True:
                false_answer_num += 1
            correct_abstention_den += 1
            if case["decision"]["correct_abstention"] is True:
                correct_abstention_num += 1

    req_cov_num = 0
    req_cov_den = 0
    candidate_cov_num = 0
    candidate_cov_den = 0
    for case in cases:
        if case["gold_review_status"] != "reviewed":
            continue
        if case["trace_id"] is None:
            continue
        if case["integrity"]["status"] == "failed":
            continue
        req_cov_num += case["evidence"]["covered_count"]
        req_cov_den += case["evidence"]["required_count"]
        if case["evidence"]["candidate_evidence"] == "available":
            required_count = case["evidence"]["required_count"]
            missing_candidate = len(case["evidence"]["missing_from_candidate_evidence_ids"])
            candidate_cov_num += required_count - missing_candidate
            candidate_cov_den += required_count

    context_num = 0
    context_den = 0
    for case in cases:
        if case["gold_review_status"] != "reviewed":
            continue
        if case["assessment_id"] is None:
            continue
        if case["integrity"]["status"] == "failed":
            continue
        if case["context_sufficiency"] in {"sufficient", "insufficient"}:
            context_den += 1
            if case["context_sufficiency"] == "sufficient":
                context_num += 1

    correctness_num = 0
    correctness_den = 0
    for case in cases:
        if case["gold_review_status"] != "reviewed":
            continue
        if case["assessment_id"] is None:
            continue
        if case["integrity"]["status"] == "failed":
            continue
        if case["answer_correctness"] in {"correct", "incorrect"}:
            correctness_den += 1
            if case["answer_correctness"] == "correct":
                correctness_num += 1

    claim_supported_total = 0
    claim_reviewed_total = 0
    claim_unsupported_total = 0
    claim_contradicted_total = 0
    claim_unassessed_total = 0
    for case in cases:
        if case["gold_review_status"] != "reviewed":
            continue
        if case["trace_id"] is None:
            continue
        if case["integrity"]["status"] == "failed":
            continue
        claims = case["claims"]
        if isinstance(claims, dict):
            claim_supported_total += claims["supported_count"]
            claim_reviewed_total += claims["reviewed_count"]
            claim_unsupported_total += claims["unsupported_count"]
            claim_contradicted_total += claims["contradicted_count"]
            claim_unassessed_total += claims["unassessed_count"]

    citation_struct_num = 0
    citation_struct_den = 0
    for case in cases:
        if case["gold_review_status"] != "reviewed":
            continue
        if case["trace_id"] is None:
            continue
        if case["integrity"]["status"] == "failed":
            continue
        citations = case["citations"]
        if citations["requirement"] == "required" and citations["structural_validity"] in {"valid", "invalid"}:
            citation_struct_den += 1
            if citations["structural_validity"] == "valid":
                citation_struct_num += 1

    citation_support_num = 0
    citation_support_den = 0
    citation_unassessed_total = 0
    citation_cited_total = 0
    for case in cases:
        if case["gold_review_status"] != "reviewed":
            continue
        if case["trace_id"] is None:
            continue
        if case["integrity"]["status"] == "failed":
            continue
        citations = case["citations"]
        if citations["requirement"] != "required":
            continue
        if citations["structural_validity"] != "valid":
            continue
        citation_cited_total += len(set(citations["cited_ids"]))
        citation_support_num += citations["supported_count"]
        citation_support_den += citations["reviewed_count"]
        citation_unassessed_total += len(citations["unassessed_ids"])

    failure_stages = {stage: 0 for stage in FAILURE_STAGES}
    for case in cases:
        if case["outcome"] == "failure":
            stage = case["failure_stage"]
            if stage in failure_stages:
                failure_stages[stage] += 1
            else:
                failure_stages["unknown"] += 1

    secondary_counts: dict[str, int] = {}
    for case in cases:
        if case["gold_review_status"] != "reviewed":
            continue
        if case["trace_id"] is None:
            continue
        if case["integrity"]["status"] == "failed":
            continue
        for label in case["secondary_labels"]:
            secondary_counts[label] = secondary_counts.get(label, 0) + 1
    secondary_counts = dict(sorted(secondary_counts.items()))

    aggregates = {
        "outcomes": outcome_counts,
        "trace_coverage": metric(trace_coverage_num, total),
        "assessment_coverage": metric(assessment_coverage_num, total),
        "integrity_failures": metric(integrity_num, integrity_den),
        "decision_accuracy": metric(decision_num, decision_den),
        "false_abstention": metric(false_abstention_num, false_abstention_den),
        "false_answer": metric(false_answer_num, false_answer_den),
        "correct_abstention": metric(correct_abstention_num, correct_abstention_den),
        "required_evidence_coverage": metric(req_cov_num, req_cov_den),
        "candidate_evidence_coverage": metric(candidate_cov_num, candidate_cov_den),
        "context_sufficiency": metric(context_num, context_den),
        "answer_correctness": metric(correctness_num, correctness_den),
        "claim_faithfulness": metric(claim_supported_total, claim_reviewed_total),
        "unsupported_claims": metric(claim_unsupported_total, claim_reviewed_total),
        "contradicted_claims": metric(claim_contradicted_total, claim_reviewed_total),
        "claim_counts": {
            "supported": claim_supported_total,
            "unsupported": claim_unsupported_total,
            "contradicted": claim_contradicted_total,
            "unassessed": claim_unassessed_total,
            "reviewed": claim_reviewed_total,
        },
        "citation_structural_validity": metric(citation_struct_num, citation_struct_den),
        "citation_support": metric(citation_support_num, citation_support_den),
        "citation_support_unassessed": metric(citation_unassessed_total, citation_cited_total),
        "citation_counts": {
            "cited": citation_cited_total,
            "reviewed": citation_support_den,
            "supported": citation_support_num,
            "unassessed": citation_unassessed_total,
        },
        "failure_stages": failure_stages,
        "secondary_labels": secondary_counts,
    }

    report = {
        "schema_version": SCHEMA_REPORT,
        "contracts": {
            "gold": SCHEMA_GOLD,
            "trace": SCHEMA_TRACE,
            "assessment": SCHEMA_ASSESSMENT,
        },
        "input": {
            "gold_questions": len(gold_questions),
            "traces": len(traces),
            "assessments": len(assessments),
            "gold_sha256": gold_sha256,
        },
        "notes": [
            "required_evidence_coverage measures annotated required evidence presence, not retrieval Recall@k.",
            "Coverage does not prove reviewed context sufficiency.",
            "Valid citation syntax does not prove factual support.",
            "Missing candidate evidence does not prove retrieval failure.",
            "Semantic truth comes only from reviewed assessment labels, not lexical overlap.",
            "Citations are evaluated against their exact evidence snapshots.",
            "Primary failure categories are mutually exclusive per case.",
            "Zero denominators are reported as null, not 0%.",
            "Unassessed labels are reported as unknown, not success.",
            "Extractive answers are evaluated without Qwen citation requirements.",
        ],
        "cases": cases,
        "aggregates": aggregates,
    }
    return report


def render_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def write_report(report: dict[str, Any], out_dir: str | Path, filename: str = REPORT_FILENAME) -> Path:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    rendered = render_report(report)
    temporary = directory / (filename + ".tmp")
    temporary.write_text(rendered, encoding="utf-8", newline="\n")
    temporary.replace(path)
    return path
