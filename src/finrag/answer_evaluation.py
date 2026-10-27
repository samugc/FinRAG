"""Deterministic, offline evaluation of RAG answer quality.

The evaluator keeps three entities strictly separate:

``finrag.answer_gold.v1``
    Reviewed ground truth: questions, document provenance, answerability,
    expected decisions, required evidence snapshots, expected answers and
    review status.

``finrag.answer_trace.v1``
    Immutable execution snapshots: exact generated output, ordered evidence,
    citation IDs, SHA-256 hashes, execution metadata, statuses and optional
    candidate evidence.

``finrag.answer_assessment.v1``
    Reviewed semantic judgements: context sufficiency, answer correctness,
    claim support, citation support and secondary error labels.

Only standard-library dependencies are used.  Semantic truth is never
inferred from lexical overlap: the reviewed assessment labels are the only
source of semantic truth.  Structural checks (hashes, references, citation
syntax, evidence coverage) are computed independently.
"""

from __future__ import annotations

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

REPORT_FILENAME = "answer_quality_report.json"

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
ANSWER_MODES = frozenset({"generated", "extractive"})
FAILURE_STAGES = (
    "retrieval",
    "context_assembly",
    "generation",
    "citation",
    "backend",
    "unknown",
)
_STAGE_PRIORITY = {
    "backend": 0,
    "retrieval": 1,
    "context_assembly": 2,
    "generation": 3,
    "citation": 4,
    "unknown": 5,
}
_HEX64 = re.compile(r"[0-9a-fA-F]{64}\Z")

CONTEXT_LABELS = frozenset({"sufficient", "insufficient", "unknown"})
ANSWER_CORRECTNESS_LABELS = frozenset({
    "correct",
    "incorrect",
    "unknown",
    "not_applicable",
})
CLAIM_LABELS = frozenset({
    "supported",
    "unsupported",
    "contradicted",
    "unknown",
})
CITATION_LABELS = frozenset({
    "supported",
    "unsupported",
    "irrelevant",
    "contradicted",
    "unknown",
})
REVIEW_STATUSES = frozenset({"reviewed", "unreviewed"})
GOLD_REVIEW_STATUSES = frozenset({"reviewed", "pending"})
ANSWERABILITY_VALUES = frozenset({"answerable", "unanswerable"})
DECISION_VALUES = frozenset({"answer", "abstain"})
ANSWER_KINDS = frozenset({"numeric", "text", "boolean"})
CITATION_ORDER = (
    "no_substantive_content",
    "malformed_brackets",
    "missing_citations",
    "unsupported_syntax",
    "out_of_range",
)


class EvaluationError(ValueError):
    """Raised for invalid schemas, references, hashes or integrity failures."""


@dataclass(frozen=True)
class GoldDocument:
    document_id: str
    source: str
    sha256: str


@dataclass(frozen=True)
class RequiredEvidence:
    evidence_id: str
    document_id: str
    source: str
    text: str
    sha256: str


@dataclass(frozen=True)
class ExpectedAnswer:
    kind: str
    value: Any
    text: str | None
    tolerance: float | None


@dataclass(frozen=True)
class GoldQuestion:
    question_id: str
    question: str
    documents: tuple[GoldDocument, ...]
    answerability: str
    expected_decision: str
    required_evidence: tuple[RequiredEvidence, ...]
    expected_answer: ExpectedAnswer | None
    review_status: str


@dataclass(frozen=True)
class EvidenceSnapshot:
    evidence_id: str
    source: str
    text: str
    text_sha256: str
    source_sha256: str
    document_id: str | None


@dataclass(frozen=True)
class Trace:
    trace_id: str
    question_id: str
    answer_mode: str
    status: str
    raw_answer: str
    evidence: tuple[EvidenceSnapshot, ...]
    citation_ids: tuple[int, ...]
    execution: dict[str, Any]
    candidate_evidence: tuple[EvidenceSnapshot, ...] | None
    error: dict[str, Any] | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ClaimAssessment:
    claim_id: str
    text: str
    label: str
    evidence_sha256: tuple[str, ...]


@dataclass(frozen=True)
class CitationAssessment:
    citation_id: int
    label: str
    evidence_sha256: str
    evidence_id: str | None


@dataclass(frozen=True)
class Assessment:
    assessment_id: str | None
    question_id: str
    review_status: str
    context_sufficiency: str
    answer_correctness: str
    claims: tuple[ClaimAssessment, ...] | None
    citations: tuple[CitationAssessment, ...] | None
    secondary_errors: tuple[str, ...]
    metadata: dict[str, Any]


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
    if not isinstance(value, str) or not value.strip():
        _error(context, "must be a non-empty string or null")
    return value


def _hex64(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        _error(context, "must be a 64-character hexadecimal SHA-256 digest")
    return value.lower()


def _str_list(value: Any, context: str) -> tuple[str, ...]:
    items = _list(value, context)
    result: list[str] = []
    for index, item in enumerate(items):
        result.append(_nonempty_str(item, f"{context}[{index}]"))
    return tuple(result)


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


def _strip_schema_version(
    record: dict[str, Any],
    expected: str,
    context: str,
) -> dict[str, Any]:
    copy = dict(record)
    version = copy.pop("schema_version", None)
    if version is not None and version != expected:
        _error(context, f"schema_version must be {expected!r}")
    return copy


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
    return []  # unreachable


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
    if header_version is not None:
        for obj in records:
            if obj.get("schema_version") != header_version:
                _error(str(path), "inconsistent schema_version in JSONL records")
    return records


def _load_dataset(
    path: Path,
    expected_schema: str,
    collection_keys: tuple[str, ...],
    label: str,
) -> list[dict[str, Any]]:
    raw = _load_json_or_jsonl(path)
    if len(raw) == 1 and isinstance(raw[0], dict):
        first = raw[0]
        present = [key for key in collection_keys if key in first]
        if len(present) > 1:
            _error(str(path), f"multiple collection fields: {', '.join(present)}")
        if present:
            allowed_top = {"schema_version", *collection_keys, "metadata"}
            _check_keys(first, {"schema_version"}, allowed_top - {"schema_version"}, str(path))
            version = first.get("schema_version")
            if version != expected_schema:
                _error(
                    str(path),
                    f"schema_version must be {expected_schema!r}, got {version!r}",
                )
            collection = first[present[0]]
            if not isinstance(collection, list):
                _error(str(path), f"{present[0]} must be a list")
            records = collection
            if any(not isinstance(item, dict) for item in records):
                _error(str(path), f"{present[0]} must contain objects")
            return [
                _strip_schema_version(item, expected_schema, f"{label} entry")
                for item in records
            ]
        if "schema_version" in first:
            version = first["schema_version"]
            if version != expected_schema:
                _error(
                    str(path),
                    f"schema_version must be {expected_schema!r}, got {version!r}",
                )
            return [_strip_schema_version(first, expected_schema, label)]
    versioned = all(
        isinstance(item, dict) and item.get("schema_version") is not None
        for item in raw
    )
    if not versioned:
        _error(
            str(path),
            "records must include schema_version or be wrapped in an object",
        )
    records = []
    for item in raw:
        if not isinstance(item, dict):
            _error(str(path), "entry must be an object")
        version = item.get("schema_version")
        if version != expected_schema:
            _error(
                str(path),
                f"schema_version must be {expected_schema!r}, got {version!r}",
            )
        records.append(_strip_schema_version(item, expected_schema, label))
    return records


def sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_gold(document: Any) -> list[GoldQuestion]:
    if isinstance(document, dict):
        if "schema_version" in document and any(
            key in document for key in ("questions", "cases", "items")
        ):
            present = [
                key
                for key in ("questions", "cases", "items")
                if key in document
            ]
            if len(present) > 1:
                _error("gold", f"multiple collection fields: {', '.join(present)}")
            version = document.get("schema_version")
            if version != SCHEMA_GOLD:
                _error(
                    "gold",
                    f"schema_version must be {SCHEMA_GOLD!r}, got {version!r}",
                )
            records = document[present[0]]
            if not isinstance(records, list):
                _error("gold", f"{present[0]} must be a list")
        else:
            records = [document]
    elif isinstance(document, list):
        records = document
    else:
        _error("gold", "document must be an object or list")
    if not records:
        _error("gold", "questions must be non-empty")
    questions: list[GoldQuestion] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        context = f"gold question {index}"
        item = _strip_schema_version(
            _mapping(record, context), SCHEMA_GOLD, context
        )
        _check_keys(
            item,
            {"question_id", "question", "documents", "answerability", "expected_decision", "required_evidence", "review_status"},
            {"expected_answer"},
            context,
        )
        question_id = _nonempty_str(item["question_id"], f"{context}.question_id")
        if question_id in seen_ids:
            _error(context, f"duplicate question_id {question_id!r}")
        seen_ids.add(question_id)
        question = _nonempty_str(item["question"], f"{context}.question")
        documents_raw = _list(item["documents"], f"{context}.documents")
        if not documents_raw:
            _error(context, "documents must be non-empty")
        documents: list[GoldDocument] = []
        seen_doc_ids: set[str] = set()
        seen_sources: set[str] = set()
        for doc_index, raw_doc in enumerate(documents_raw):
            doc_context = f"{context}.documents[{doc_index}]"
            doc = _mapping(raw_doc, doc_context)
            _check_keys(
                doc,
                {"document_id", "source", "sha256"},
                {"content"},
                doc_context,
            )
            document_id = _nonempty_str(
                doc["document_id"], f"{doc_context}.document_id"
            )
            if document_id in seen_doc_ids:
                _error(doc_context, f"duplicate document_id {document_id!r}")
            seen_doc_ids.add(document_id)
            source = _nonempty_str(doc["source"], f"{doc_context}.source")
            if source in seen_sources:
                _error(doc_context, f"duplicate source {source!r}")
            seen_sources.add(source)
            sha = _hex64(doc["sha256"], f"{doc_context}.sha256")
            content = None
            if "content" in doc and doc["content"] is not None:
                content = _nonempty_str(doc["content"], f"{doc_context}.content")
                if sha256_text(content) != sha:
                    _error(doc_context, "content SHA-256 does not match sha256")
            documents.append(
                GoldDocument(document_id=document_id, source=source, sha256=sha)
            )
        answerability = item["answerability"]
        if answerability not in ANSWERABILITY_VALUES:
            _error(context, "invalid answerability")
        expected_decision = item["expected_decision"]
        if expected_decision not in DECISION_VALUES:
            _error(context, "invalid expected_decision")
        if answerability == "answerable" and expected_decision != "answer":
            _error(context, "answerable questions must expect decision 'answer'")
        if answerability == "unanswerable" and expected_decision != "abstain":
            _error(context, "unanswerable questions must expect decision 'abstain'")
        required_raw = _list(
            item["required_evidence"], f"{context}.required_evidence"
        )
        required: list[RequiredEvidence] = []
        seen_required_ids: set[str] = set()
        for ev_index, raw_ev in enumerate(required_raw):
            ev_context = f"{context}.required_evidence[{ev_index}]"
            ev = _mapping(raw_ev, ev_context)
            _check_keys(
                ev,
                {"evidence_id", "document_id", "source", "text", "sha256"},
                set(),
                ev_context,
            )
            evidence_id = _nonempty_str(
                ev["evidence_id"], f"{ev_context}.evidence_id"
            )
            if evidence_id in seen_required_ids:
                _error(ev_context, f"duplicate evidence_id {evidence_id!r}")
            seen_required_ids.add(evidence_id)
            document_id = _nonempty_str(
                ev["document_id"], f"{ev_context}.document_id"
            )
            source = _nonempty_str(ev["source"], f"{ev_context}.source")
            matching_docs = [
                doc for doc in documents if doc.document_id == document_id
            ]
            if not matching_docs:
                _error(ev_context, f"document_id {document_id!r} is not declared")
            if matching_docs[0].source != source:
                _error(ev_context, "source does not match declared document")
            text = _nonempty_str(ev["text"], f"{ev_context}.text")
            sha = _hex64(ev["sha256"], f"{ev_context}.sha256")
            if sha256_text(text) != sha:
                _error(ev_context, "text SHA-256 does not match sha256")
            required.append(
                RequiredEvidence(
                    evidence_id=evidence_id,
                    document_id=document_id,
                    source=source,
                    text=text,
                    sha256=sha,
                )
            )
        review_status = item["review_status"]
        if review_status not in GOLD_REVIEW_STATUSES:
            _error(context, "invalid review_status")
        expected_answer = None
        if "expected_answer" in item and item["expected_answer"] is not None:
            answer_context = f"{context}.expected_answer"
            answer_value = _mapping(item["expected_answer"], answer_context)
            _check_keys(
                answer_value,
                {"kind", "value"},
                {"text", "tolerance"},
                answer_context,
            )
            kind = answer_value["kind"]
            if kind not in ANSWER_KINDS:
                _error(answer_context, "invalid expected answer kind")
            value = answer_value["value"]
            if kind == "numeric":
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    _error(answer_context, "numeric value must be a number")
                if not math.isfinite(value):
                    _error(answer_context, "numeric value must be finite")
                tolerance = answer_value.get("tolerance")
                if tolerance is not None:
                    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
                        _error(answer_context, "tolerance must be a number")
                    if tolerance < 0 or not math.isfinite(tolerance):
                        _error(answer_context, "tolerance must be non-negative and finite")
            elif kind == "boolean":
                if not isinstance(value, bool):
                    _error(answer_context, "boolean value must be a boolean")
                tolerance = None
            else:
                if not isinstance(value, str):
                    _error(answer_context, "text value must be a string")
                tolerance = None
            text_value = answer_value.get("text")
            if text_value is not None:
                text_value = _nonempty_str(text_value, f"{answer_context}.text")
            if answerability == "unanswerable":
                _error(context, "unanswerable questions cannot have expected_answer")
            expected_answer = ExpectedAnswer(
                kind=kind,
                value=value,
                text=text_value,
                tolerance=tolerance,
            )
        if answerability == "answerable" and expected_answer is None:
            _error(context, "answerable questions require expected_answer")
        questions.append(
            GoldQuestion(
                question_id=question_id,
                question=question,
                documents=tuple(documents),
                answerability=answerability,
                expected_decision=expected_decision,
                required_evidence=tuple(required),
                expected_answer=expected_answer,
                review_status=review_status,
            )
        )
    return questions


def _parse_evidence(record: Any, context: str, index: int | None = None) -> EvidenceSnapshot:
    item = _mapping(record, context)
    optional_rank = "rank" in item
    allowed_optional = {"document_id", "rank"}
    _check_keys(
        item,
        {"evidence_id", "source", "text", "text_sha256", "source_sha256"},
        allowed_optional,
        context,
    )
    evidence_id = _nonempty_str(item["evidence_id"], f"{context}.evidence_id")
    source = _nonempty_str(item["source"], f"{context}.source")
    text = _nonempty_str(item["text"], f"{context}.text")
    text_sha = _hex64(item["text_sha256"], f"{context}.text_sha256")
    source_sha = _hex64(item["source_sha256"], f"{context}.source_sha256")
    document_id = _optional_str(item.get("document_id"), f"{context}.document_id")
    if optional_rank:
        rank = item["rank"]
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            _error(context, "rank must be a positive integer")
        if index is not None and rank != index + 1:
            _error(context, f"rank must equal {index + 1} in ordered evidence")
    return EvidenceSnapshot(
        evidence_id=evidence_id,
        source=source,
        text=text,
        text_sha256=text_sha,
        source_sha256=source_sha,
        document_id=document_id,
    )


def _parse_trace(record: Any) -> Trace:
    context = "trace"
    item = _strip_schema_version(
        _mapping(record, context), SCHEMA_TRACE, context
    )
    _check_keys(
        item,
        {
            "trace_id",
            "question_id",
            "answer_mode",
            "status",
            "raw_answer",
            "evidence",
            "citation_ids",
            "execution",
        },
        {"candidate_evidence", "error", "metadata"},
        context,
    )
    trace_id = _nonempty_str(item["trace_id"], "trace.trace_id")
    question_id = _nonempty_str(item["question_id"], "trace.question_id")
    answer_mode = item["answer_mode"]
    if answer_mode not in ANSWER_MODES:
        _error(context, "invalid answer_mode")
    status = item["status"]
    if status not in ANSWER_STATUSES:
        _error(context, "invalid status")
    raw_answer = item["raw_answer"]
    if not isinstance(raw_answer, str):
        _error(context, "raw_answer must be a string")
    evidence_raw = _list(item["evidence"], "trace.evidence")
    evidence: list[EvidenceSnapshot] = []
    seen_ids: set[str] = set()
    for index, raw_ev in enumerate(evidence_raw):
        ev = _parse_evidence(
            raw_ev, f"trace.evidence[{index}]", index=index
        )
        if ev.evidence_id in seen_ids:
            _error(
                f"trace.evidence[{index}]",
                f"duplicate evidence_id {ev.evidence_id!r}",
            )
        seen_ids.add(ev.evidence_id)
        evidence.append(ev)
    citation_ids_raw = _list(item["citation_ids"], "trace.citation_ids")
    citation_ids: list[int] = []
    for index, value in enumerate(citation_ids_raw):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            _error(
                f"trace.citation_ids[{index}]",
                "citation IDs must be positive integers",
            )
        citation_ids.append(value)
    execution = _mapping(item["execution"], "trace.execution")
    _check_keys(
        execution,
        {"backend"},
        {"model", "duration_ms", "started_at", "retriever", "generator", "retrieval_mode", "k", "max_pages", "chunks_per_page", "metadata"},
        "trace.execution",
    )
    backend = _nonempty_str(execution["backend"], "trace.execution.backend")
    execution = dict(execution)
    execution["backend"] = backend
    for key in ("model", "retriever", "generator", "started_at", "retrieval_mode"):
        if key in execution and execution[key] is not None:
            execution[key] = _nonempty_str(execution[key], f"trace.execution.{key}")
    for key in ("duration_ms",):
        if key in execution and execution[key] is not None:
            value = execution[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                _error(f"trace.execution.{key}", "must be a number or null")
            if value < 0 or not math.isfinite(value):
                _error(f"trace.execution.{key}", "must be non-negative and finite")
    for key in ("k", "max_pages", "chunks_per_page"):
        if key in execution and execution[key] is not None:
            value = execution[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                _error(f"trace.execution.{key}", "must be a positive integer or null")
    if "metadata" in execution and execution["metadata"] is not None:
        execution["metadata"] = _mapping(execution["metadata"], "trace.execution.metadata")
    candidate_raw = item.get("candidate_evidence", None)
    if candidate_raw is None:
        candidate_evidence = None
    else:
        candidate_list = _list(candidate_raw, "trace.candidate_evidence")
        candidate_evidence = []
        seen_candidate_ids: set[str] = set()
        for index, raw_ev in enumerate(candidate_list):
            ev = _parse_evidence(
                raw_ev, f"trace.candidate_evidence[{index}]"
            )
            if ev.evidence_id in seen_candidate_ids:
                _error(
                    f"trace.candidate_evidence[{index}]",
                    f"duplicate evidence_id {ev.evidence_id!r}",
                )
            seen_candidate_ids.add(ev.evidence_id)
            candidate_evidence.append(ev)
        candidate_evidence = tuple(candidate_evidence)
    error = item.get("error")
    if error is not None:
        error = _mapping(error, "trace.error")
    metadata = item.get("metadata", {})
    if metadata is None:
        metadata = {}
    metadata = _mapping(metadata, "trace.metadata")
    return Trace(
        trace_id=trace_id,
        question_id=question_id,
        answer_mode=answer_mode,
        status=status,
        raw_answer=raw_answer,
        evidence=tuple(evidence),
        citation_ids=tuple(citation_ids),
        execution=execution,
        candidate_evidence=candidate_evidence,
        error=error,
        metadata=metadata,
    )


def parse_traces(records: Any) -> list[Trace]:
    if not isinstance(records, list):
        _error("traces", "must be a list")
    traces: list[Trace] = []
    seen_trace_ids: set[str] = set()
    seen_question_ids: set[str] = set()
    for index, record in enumerate(records):
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
    _check_keys(
        item,
        {"claim_id", "text", "label"},
        {"evidence_sha256"},
        context,
    )
    claim_id = _nonempty_str(item["claim_id"], f"{context}.claim_id")
    text = _nonempty_str(item["text"], f"{context}.text")
    label = item["label"]
    if label not in CLAIM_LABELS:
        _error(context, "invalid claim label")
    evidence = item.get("evidence_sha256", [])
    if evidence is None:
        evidence = []
    evidence_list = _list(evidence, f"{context}.evidence_sha256")
    evidence_sha: list[str] = []
    for index, value in enumerate(evidence_list):
        evidence_sha.append(
            _hex64(value, f"{context}.evidence_sha256[{index}]")
        )
    return ClaimAssessment(
        claim_id=claim_id,
        text=text,
        label=label,
        evidence_sha256=tuple(evidence_sha),
    )


def _parse_citation_assessment(record: Any, context: str) -> CitationAssessment:
    item = _mapping(record, context)
    _check_keys(
        item,
        {"citation_id", "label", "evidence_sha256"},
        {"evidence_id"},
        context,
    )
    citation_id = item["citation_id"]
    if (
        isinstance(citation_id, bool)
        or not isinstance(citation_id, int)
        or citation_id < 1
    ):
        _error(context, "citation_id must be a positive integer")
    label = item["label"]
    if label not in CITATION_LABELS:
        _error(context, "invalid citation label")
    evidence_sha = _hex64(
        item["evidence_sha256"], f"{context}.evidence_sha256"
    )
    evidence_id = _optional_str(
        item.get("evidence_id"), f"{context}.evidence_id"
    )
    return CitationAssessment(
        citation_id=citation_id,
        label=label,
        evidence_sha256=evidence_sha,
        evidence_id=evidence_id,
    )


def parse_assessments(records: Any) -> list[Assessment]:
    if not isinstance(records, list):
        _error("assessments", "must be a list")
    assessments: list[Assessment] = []
    seen_assessment_ids: set[str] = set()
    seen_question_ids: set[str] = set()
    for index, record in enumerate(records):
        context = f"assessment {index}"
        item = _strip_schema_version(
            _mapping(record, context), SCHEMA_ASSESSMENT, context
        )
        _check_keys(
            item,
            {
                "question_id",
                "review_status",
                "context_sufficiency",
                "answer_correctness",
                "claims",
                "citations",
                "secondary_errors",
            },
            {"assessment_id", "metadata", "reviewer", "reviewed_at"},
            context,
        )
        question_id = _nonempty_str(
            item["question_id"], f"{context}.question_id"
        )
        review_status = item["review_status"]
        if review_status not in REVIEW_STATUSES:
            _error(context, "invalid review_status")
        context_sufficiency = item["context_sufficiency"]
        if context_sufficiency not in CONTEXT_LABELS:
            _error(context, "invalid context_sufficiency")
        answer_correctness = item["answer_correctness"]
        if answer_correctness not in ANSWER_CORRECTNESS_LABELS:
            _error(context, "invalid answer_correctness")
        claims_raw = item["claims"]
        if claims_raw is None:
            claims = None
        else:
            claims_list = _list(claims_raw, f"{context}.claims")
            claims = []
            seen_claim_ids: set[str] = set()
            for claim_index, raw_claim in enumerate(claims_list):
                claim = _parse_claim(
                    raw_claim, f"{context}.claims[{claim_index}]"
                )
                if claim.claim_id in seen_claim_ids:
                    _error(
                        f"{context}.claims[{claim_index}]",
                        f"duplicate claim_id {claim.claim_id!r}",
                    )
                seen_claim_ids.add(claim.claim_id)
                claims.append(claim)
            claims = tuple(claims)
        citations_raw = item["citations"]
        if citations_raw is None:
            citations = None
        else:
            citations_list = _list(citations_raw, f"{context}.citations")
            citations = []
            seen_citation_ids: set[int] = set()
            for citation_index, raw_citation in enumerate(citations_list):
                citation = _parse_citation_assessment(
                    raw_citation, f"{context}.citations[{citation_index}]"
                )
                if citation.citation_id in seen_citation_ids:
                    _error(
                        f"{context}.citations[{citation_index}]",
                        f"duplicate citation_id {citation.citation_id!r}",
                    )
                seen_citation_ids.add(citation.citation_id)
                citations.append(citation)
            citations = tuple(citations)
        secondary_errors = _str_list(
            item["secondary_errors"], f"{context}.secondary_errors"
        )
        assessment_id = _optional_str(
            item.get("assessment_id"), f"{context}.assessment_id"
        )
        if assessment_id is not None:
            if assessment_id in seen_assessment_ids:
                _error(context, f"duplicate assessment_id {assessment_id!r}")
            seen_assessment_ids.add(assessment_id)
        if question_id in seen_question_ids:
            _error(context, f"duplicate question_id {question_id!r}")
        seen_question_ids.add(question_id)
        metadata = item.get("metadata", {})
        if metadata is None:
            metadata = {}
        metadata = _mapping(metadata, f"{context}.metadata")
        assessments.append(
            Assessment(
                assessment_id=assessment_id,
                question_id=question_id,
                review_status=review_status,
                context_sufficiency=context_sufficiency,
                answer_correctness=answer_correctness,
                claims=claims,
                citations=citations,
                secondary_errors=tuple(sorted(set(secondary_errors))),
                metadata=metadata,
            )
        )
    return assessments


def load_gold(path: str | Path) -> list[GoldQuestion]:
    records = _load_dataset(Path(path), SCHEMA_GOLD, ("questions", "cases", "items"), "gold")
    return parse_gold(records)


def load_traces(path: str | Path) -> list[Trace]:
    records = _load_dataset(Path(path), SCHEMA_TRACE, ("traces", "items", "cases"), "trace")
    return parse_traces(records)


def load_assessments(path: str | Path) -> list[Assessment]:
    records = _load_dataset(Path(path), SCHEMA_ASSESSMENT, ("assessments", "items", "cases"), "assessment")
    return parse_assessments(records)


def metric(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": None if denominator == 0 else round(numerator / denominator, 6),
        "status": "assessed" if denominator > 0 else "unassessed",
    }


def _observed_decision(status: str) -> str:
    if status == "produced":
        return "answer"
    if status == "invalid_output":
        return "invalid"
    if status == "backend_error":
        return "error"
    return "abstain"


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


def _integrity_errors(
    gold: GoldQuestion,
    trace: Trace,
    assessment: Assessment | None,
) -> list[str]:
    errors: list[str] = []
    docs_by_source = {doc.source: doc for doc in gold.documents}
    required_by_id = {req.evidence_id: req for req in gold.required_evidence}
    required_hashes = {(req.source, req.sha256) for req in gold.required_evidence}

    for evidence in trace.evidence:
        recomputed = sha256_text(evidence.text)
        if recomputed != evidence.text_sha256:
            errors.append(
                f"evidence {evidence.evidence_id}: text SHA-256 mismatch"
            )
        doc = docs_by_source.get(evidence.source)
        if doc is None:
            errors.append(
                f"evidence {evidence.evidence_id}: source not declared in gold"
            )
        else:
            if evidence.source_sha256 != doc.sha256:
                errors.append(
                    f"evidence {evidence.evidence_id}: source SHA-256 mismatch"
                )
            if evidence.document_id is not None and evidence.document_id != doc.document_id:
                errors.append(
                    f"evidence {evidence.evidence_id}: document_id mismatch"
                )
        if evidence.evidence_id in required_by_id:
            req = required_by_id[evidence.evidence_id]
            if recomputed != req.sha256:
                errors.append(
                    f"evidence {evidence.evidence_id}: required evidence snapshot mismatch"
                )
            if evidence.source != req.source:
                errors.append(
                    f"evidence {evidence.evidence_id}: required evidence source mismatch"
                )

    if trace.candidate_evidence is not None:
        for evidence in trace.candidate_evidence:
            recomputed = sha256_text(evidence.text)
            if recomputed != evidence.text_sha256:
                errors.append(
                    f"candidate {evidence.evidence_id}: text SHA-256 mismatch"
                )
            doc = docs_by_source.get(evidence.source)
            if doc is None:
                errors.append(
                    f"candidate {evidence.evidence_id}: source not declared in gold"
                )
            else:
                if evidence.source_sha256 != doc.sha256:
                    errors.append(
                        f"candidate {evidence.evidence_id}: source SHA-256 mismatch"
                    )
                if evidence.document_id is not None and evidence.document_id != doc.document_id:
                    errors.append(
                        f"candidate {evidence.evidence_id}: document_id mismatch"
                    )
            if evidence.evidence_id in required_by_id:
                req = required_by_id[evidence.evidence_id]
                if recomputed != req.sha256:
                    errors.append(
                        f"candidate {evidence.evidence_id}: required evidence snapshot mismatch"
                    )

    if assessment is not None and assessment.review_status == "reviewed":
        if trace.status in {"produced", "invalid_output"}:
            for claim in assessment.claims or ():
                if claim.text not in trace.raw_answer:
                    errors.append(
                        f"claim {claim.claim_id}: text not found in generated output"
                    )
                for evidence_sha in claim.evidence_sha256:
                    known_hashes = {sha256_text(ev.text) for ev in trace.evidence}
                    known_hashes.update(req.sha256 for req in gold.required_evidence)
                    if evidence_sha not in known_hashes:
                        errors.append(
                            f"claim {claim.claim_id}: evidence snapshot not found"
                        )
        if trace.answer_mode == "generated" and trace.evidence:
            cited_ids = set(_parse_citations(trace.raw_answer, len(trace.evidence))[0])
            if trace.citation_ids and _parse_citations(trace.raw_answer, len(trace.evidence))[1] == []:
                parsed_ids = _parse_citations(trace.raw_answer, len(trace.evidence))[0]
                if parsed_ids != list(trace.citation_ids):
                    errors.append("citation_ids do not match generated output")
            for citation in assessment.citations or ():
                if citation.citation_id not in cited_ids and _parse_citations(trace.raw_answer, len(trace.evidence))[1] == []:
                    errors.append(
                        f"citation {citation.citation_id}: not cited in generated output"
                    )
                if citation.citation_id <= len(trace.evidence):
                    evidence = trace.evidence[citation.citation_id - 1]
                    if citation.evidence_sha256 != sha256_text(evidence.text):
                        errors.append(
                            f"citation {citation.citation_id}: evidence snapshot mismatch"
                        )
                    if citation.evidence_id is not None and citation.evidence_id != evidence.evidence_id:
                        errors.append(
                            f"citation {citation.citation_id}: evidence_id mismatch"
                        )
    errors = sorted(set(errors))
    return errors


def _covered_required_ids(
    required: tuple[RequiredEvidence, ...],
    evidence: tuple[EvidenceSnapshot, ...],
) -> set[str]:
    covered: set[str] = set()
    for req in required:
        for ev in evidence:
            if ev.source == req.source and sha256_text(ev.text) == req.sha256:
                covered.add(req.evidence_id)
                break
    return covered


def _evaluate_case(
    gold: GoldQuestion,
    trace: Trace | None,
    assessment: Assessment | None,
) -> dict[str, Any]:
    integrity = ["trace missing"] if trace is None else []
    if trace is not None:
        integrity = _integrity_errors(gold, trace, assessment)
    integrity_failed = bool(integrity)

    observed = None
    trace_status = None
    answer_mode = None
    trace_id = None
    assessment_id = assessment.assessment_id if assessment is not None else None
    if trace is not None:
        trace_id = trace.trace_id
        trace_status = trace.status
        answer_mode = trace.answer_mode
        observed = _observed_decision(trace.status)

    gold_reviewed = gold.review_status == "reviewed"
    assessment_reviewed = (
        assessment is not None and assessment.review_status == "reviewed"
    )
    trace_exists = trace is not None

    # Decision metrics.
    decision_correct: bool | None = None
    false_abstention: bool | None = None
    false_answer: bool | None = None
    correct_abstention: bool | None = None
    if gold_reviewed and trace_exists and observed in {"answer", "abstain"}:
        if gold.expected_decision == "answer":
            decision_correct = observed == "answer"
            false_abstention = observed == "abstain"
        else:
            decision_correct = observed == "abstain"
            false_answer = observed == "answer"
            correct_abstention = observed == "abstain"

    # Evidence coverage.
    required_count = len(gold.required_evidence)
    required_ids = {req.evidence_id for req in gold.required_evidence}
    covered_ids: set[str] = set()
    candidate_available = False
    candidate_covered_ids: set[str] = set()
    missing_required_ids: list[str] = []
    candidate_missing_ids: list[str] = []
    if trace_exists:
        covered_ids = _covered_required_ids(gold.required_evidence, trace.evidence)
        missing_required_ids = sorted(required_ids - covered_ids)
        if trace.candidate_evidence is not None:
            candidate_available = True
            candidate_covered_ids = _covered_required_ids(
                gold.required_evidence, trace.candidate_evidence
            )
            candidate_missing_ids = sorted(required_ids - candidate_covered_ids)

    coverage_metric = (
        metric(len(covered_ids), required_count)
        if trace_exists
        else None
    )
    candidate_coverage_metric = (
        metric(len(candidate_covered_ids), required_count)
        if trace_exists and candidate_available
        else None
    )

    # Assessment labels.
    context_label = None
    correctness_label = None
    if assessment_reviewed and assessment is not None:
        context_label = assessment.context_sufficiency
        if context_label == "unknown":
            context_label = None
        correctness_label = assessment.answer_correctness
        if correctness_label == "unknown":
            correctness_label = None
        if trace_exists and trace.status in ABSTENTION_STATUSES:
            if correctness_label in {"correct", "incorrect"}:
                correctness_label = None
        if trace_exists and trace.status == "produced" and correctness_label == "not_applicable":
            correctness_label = None

    # Claims.
    claim_items: list[dict[str, Any]] = []
    claim_assessed = 0
    claim_supported = 0
    claim_unsupported = 0
    claim_contradicted = 0
    claim_unassessed = 0
    if assessment_reviewed and assessment is not None and trace_exists and trace.status in {"produced", "invalid_output"}:
        for claim in sorted(assessment.claims or (), key=lambda item: item.claim_id):
            claim_items.append(
                {
                    "claim_id": claim.claim_id,
                    "text": claim.text,
                    "label": claim.label,
                }
            )
            if claim.label == "supported":
                claim_assessed += 1
                claim_supported += 1
            elif claim.label == "unsupported":
                claim_assessed += 1
                claim_unsupported += 1
            elif claim.label == "contradicted":
                claim_assessed += 1
                claim_contradicted += 1
            else:
                claim_unassessed += 1
    claim_faithfulness = metric(claim_supported, claim_assessed)

    # Citations.
    generated = answer_mode == "generated"
    citation_requirement = "required" if generated else "not_applicable"
    cited_ids: list[int] = []
    syntax_errors: list[str] = []
    structural_validity = "not_applicable"
    if generated and trace_exists and trace_status in {"produced", "invalid_output"}:
        cited_ids, syntax_errors = _parse_citations(
            trace.raw_answer, len(trace.evidence)
        )
        structural_validity = "valid" if not syntax_errors else "invalid"
    unique_cited = sorted(set(cited_ids))
    recorded_ids = list(trace.citation_ids) if trace_exists else []

    # Citation support.
    citation_assessed = 0
    citation_supported = 0
    citation_unsupported_ids: list[int] = []
    citation_irrelevant_ids: list[int] = []
    citation_contradicted_ids: list[int] = []
    citation_unassessed_ids: list[int] = []
    if (
        generated
        and structural_validity == "valid"
        and assessment_reviewed
        and assessment is not None
        and trace_exists
    ):
        by_id = {item.citation_id: item for item in (assessment.citations or ())}
        for citation_id in unique_cited:
            if citation_id not in by_id:
                citation_unassessed_ids.append(citation_id)
                continue
            entry = by_id[citation_id]
            if entry.label == "supported":
                citation_assessed += 1
                citation_supported += 1
            elif entry.label == "unsupported":
                citation_assessed += 1
                citation_unsupported_ids.append(citation_id)
            elif entry.label == "irrelevant":
                citation_assessed += 1
                citation_irrelevant_ids.append(citation_id)
            elif entry.label == "contradicted":
                citation_assessed += 1
                citation_contradicted_ids.append(citation_id)
            else:
                citation_unassessed_ids.append(citation_id)
    citation_support_metric = metric(citation_supported, citation_assessed)

    # Failure detection.
    failure_reasons: list[str] = []
    unassessed_reasons: list[str] = []
    stage_candidates: list[str] = []

    if integrity_failed:
        failure_reasons.append("integrity_failure")

    if not trace_exists:
        unassessed_reasons.append("missing_trace")

    if not gold_reviewed:
        unassessed_reasons.append("gold_unreviewed")

    if assessment is None:
        unassessed_reasons.append("missing_assessment")
    elif not assessment_reviewed:
        unassessed_reasons.append("assessment_unreviewed")

    if trace_exists and trace.status == "backend_error":
        failure_reasons.append("backend_error")
        stage_candidates.append("backend")

    # Required evidence stage attribution.
    evidence_stage: str | None = None
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
    if context_label is None and assessment_reviewed:
        unassessed_reasons.append("context_sufficiency_unassessed")
    if assessment is None or not assessment_reviewed:
        pass
    elif assessment_reviewed and assessment.context_sufficiency == "unknown":
        unassessed_reasons.append("context_sufficiency_unassessed")

    # Decision failures.
    if gold.expected_decision == "answer" and observed == "abstain":
        failure_reasons.append("false_abstention")
        stage_candidates.append("generation")
    if gold.expected_decision == "abstain" and observed == "answer":
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
    if (
        generated
        and structural_validity == "valid"
        and citation_unsupported_ids
    ):
        citation_support_failed = True
        failure_reasons.append("citation_unsupported")
    if (
        generated
        and structural_validity == "valid"
        and citation_irrelevant_ids
    ):
        citation_support_failed = True
        failure_reasons.append("citation_irrelevant")
    if (
        generated
        and structural_validity == "valid"
        and citation_contradicted_ids
    ):
        citation_support_failed = True
        failure_reasons.append("citation_contradicted")
    if citation_support_failed:
        stage_candidates.append("citation")

    # Unassessed reasons.
    if gold.expected_decision == "answer":
        if correctness_label is None and assessment_reviewed:
            unassessed_reasons.append("answer_correctness_unassessed")
        if claim_assessed == 0 and assessment_reviewed and trace_exists:
            unassessed_reasons.append("claims_unassessed")
        if claim_unassessed > 0:
            unassessed_reasons.append("claims_unassessed")
        if generated and structural_validity == "valid" and citation_unassessed_ids:
            unassessed_reasons.append("citation_support_unassessed")
        if generated and structural_validity == "valid" and not unique_cited:
            # missing citations would already be structural invalid
            pass
        if generated and structural_validity == "valid" and unique_cited and citation_assessed == 0 and assessment_reviewed:
            unassessed_reasons.append("citation_support_unassessed")
    if assessment is None or not assessment_reviewed:
        pass

    if gold.expected_decision == "answer" and assessment_reviewed and assessment is not None and assessment.claims is None:
        if "claims_unassessed" not in unassessed_reasons:
            unassessed_reasons.append("claims_unassessed")
    if gold.expected_decision == "answer" and assessment_reviewed and assessment is not None and assessment.citations is None and generated:
        if "citation_support_unassessed" not in unassessed_reasons and structural_validity == "valid" and unique_cited:
            unassessed_reasons.append("citation_support_unassessed")

    failure_reasons = sorted(set(failure_reasons))
    unassessed_reasons = sorted(set(unassessed_reasons))

    # Determine stage.
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
        and assessment_reviewed
        and not integrity_failed
        and not failure_reasons
        and not unassessed_reasons
    ):
        if gold.expected_decision == "answer":
            if (
                trace.status == "produced"
                and observed == "answer"
                and decision_correct is True
                and not missing_required_ids
                and context_label == "sufficient"
                and correctness_label == "correct"
                and claim_assessed > 0
                and claim_unassessed == 0
                and claim_unsupported == 0
                and claim_contradicted == 0
            ):
                if answer_mode == "extractive":
                    success = True
                else:
                    if (
                        structural_validity == "valid"
                        and len(unique_cited) > 0
                        and citation_assessed == len(unique_cited)
                        and not citation_unsupported_ids
                        and not citation_irrelevant_ids
                        and not citation_contradicted_ids
                        and not citation_unassessed_ids
                    ):
                        success = True
        else:
            if (
                trace.status in ABSTENTION_STATUSES
                and observed == "abstain"
                and decision_correct is True
                and context_label is not None
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
        # No explicit failure but success checks not met.
        outcome = "unassessed"
        if not unassessed_reasons:
            unassessed_reasons = ["success_criteria_not_met"]
            unassessed_reasons = sorted(set(unassessed_reasons))

    if outcome == "unassessed" and stage is None:
        stage = "unknown"
    if outcome == "success":
        stage = None

    # If no failure but unassessed, stage unknown.
    if outcome in {"unassessed", "invalid"}:
        if stage is None:
            stage = "unknown"

    claim_faithfulness_value = claim_faithfulness
    if claim_assessed == 0:
        claim_faithfulness_value = metric(0, 0)

    case = {
        "question_id": gold.question_id,
        "trace_id": trace_id,
        "assessment_id": assessment_id,
        "gold_review_status": gold.review_status,
        "answerability": gold.answerability,
        "expected_decision": gold.expected_decision,
        "observed_decision": observed,
        "trace_status": trace_status,
        "answer_mode": answer_mode,
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
        "claims": {
            "items": claim_items,
            "assessed_count": claim_assessed,
            "supported_count": claim_supported,
            "unsupported_count": claim_unsupported,
            "contradicted_count": claim_contradicted,
            "unassessed_count": claim_unassessed,
            "faithfulness": claim_faithfulness_value,
        },
        "citations": {
            "requirement": citation_requirement,
            "structural_validity": structural_validity,
            "syntax_errors": syntax_errors,
            "cited_ids": cited_ids,
            "recorded_ids": recorded_ids,
            "assessed_count": citation_assessed,
            "supported_count": citation_supported,
            "unsupported_ids": citation_unsupported_ids,
            "irrelevant_ids": citation_irrelevant_ids,
            "contradicted_ids": citation_contradicted_ids,
            "unassessed_ids": citation_unassessed_ids,
            "support": citation_support_metric,
        },
        "secondary_errors": list(assessment.secondary_errors) if assessment_reviewed and assessment is not None else [],
    }
    return case


def evaluate_answer_quality(
    gold_questions: list[GoldQuestion],
    traces: list[Trace],
    assessments: list[Assessment],
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
    for trace in traces:
        if trace.question_id in trace_by_question:
            raise EvaluationError(f"duplicate trace question_id {trace.question_id!r}")
        if trace.question_id not in gold_by_id:
            raise EvaluationError(f"trace references unknown question_id {trace.question_id!r}")
        trace_by_question[trace.question_id] = trace

    assessment_by_question = {}
    for assessment in assessments:
        if assessment.question_id in assessment_by_question:
            raise EvaluationError(f"duplicate assessment question_id {assessment.question_id!r}")
        if assessment.question_id not in gold_by_id:
            raise EvaluationError(f"assessment references unknown question_id {assessment.question_id!r}")
        assessment_by_question[assessment.question_id] = assessment

    cases = []
    for gold in sorted(gold_questions, key=lambda item: item.question_id):
        trace = trace_by_question.get(gold.question_id)
        assessment = assessment_by_question.get(gold.question_id)
        cases.append(_evaluate_case(gold, trace, assessment))

    # Aggregates.
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
        decision = case["decision"]
        observed = case["observed_decision"]
        if observed in {"answer", "abstain"}:
            decision_den += 1
            if decision["correct"] is True:
                decision_num += 1
        if case["expected_decision"] == "answer" and observed in {"answer", "abstain"}:
            false_abstention_den += 1
            if decision["false_abstention"] is True:
                false_abstention_num += 1
        if case["expected_decision"] == "abstain" and observed in {"answer", "abstain"}:
            false_answer_den += 1
            if decision["false_answer"] is True:
                false_answer_num += 1
            correct_abstention_den += 1
            if decision["correct_abstention"] is True:
                correct_abstention_num += 1

    # Required evidence coverage.
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

    # Context sufficiency.
    context_num = 0
    context_den = 0
    for case in cases:
        if case["gold_review_status"] != "reviewed":
            continue
        if case["assessment_id"] is None:
            continue
        if case["integrity"]["status"] == "failed":
            continue
        # Only reviewed assessments expose labels; context_sufficiency None if unreviewed.
        if case["context_sufficiency"] in {"sufficient", "insufficient"}:
            context_den += 1
            if case["context_sufficiency"] == "sufficient":
                context_num += 1

    # Answer correctness.
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

    # Claims.
    claim_supported_total = 0
    claim_assessed_total = 0
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
        claim_supported_total += claims["supported_count"]
        claim_assessed_total += claims["assessed_count"]
        claim_unsupported_total += claims["unsupported_count"]
        claim_contradicted_total += claims["contradicted_count"]
        claim_unassessed_total += claims["unassessed_count"]

    # Citations structural.
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

    # Citation support.
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
        unique_cited = sorted(set(citations["cited_ids"]))
        citation_cited_total += len(unique_cited)
        citation_support_num += citations["supported_count"]
        citation_support_den += citations["assessed_count"]
        citation_unassessed_total += len(citations["unassessed_ids"])

    # Failure stages.
    failure_stages = {stage: 0 for stage in FAILURE_STAGES}
    for case in cases:
        if case["outcome"] == "failure":
            stage = case["failure_stage"]
            if stage in failure_stages:
                failure_stages[stage] += 1
            else:
                failure_stages["unknown"] += 1

    # Secondary errors.
    secondary_counts: dict[str, int] = {}
    for case in cases:
        if case["gold_review_status"] != "reviewed":
            continue
        if case["trace_id"] is None:
            continue
        if case["integrity"]["status"] == "failed":
            continue
        for label in case["secondary_errors"]:
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
        "claim_faithfulness": metric(claim_supported_total, claim_assessed_total),
        "unsupported_claims": metric(claim_unsupported_total, claim_assessed_total),
        "contradicted_claims": metric(claim_contradicted_total, claim_assessed_total),
        "claim_counts": {
            "supported": claim_supported_total,
            "unsupported": claim_unsupported_total,
            "contradicted": claim_contradicted_total,
            "unassessed": claim_unassessed_total,
            "assessed": claim_assessed_total,
        },
        "citation_structural_validity": metric(citation_struct_num, citation_struct_den),
        "citation_support": metric(citation_support_num, citation_support_den),
        "citation_support_unassessed": metric(citation_unassessed_total, citation_cited_total),
        "citation_counts": {
            "cited": citation_cited_total,
            "assessed": citation_support_den,
            "supported": citation_support_num,
            "unassessed": citation_unassessed_total,
        },
        "failure_stages": failure_stages,
        "secondary_errors": secondary_counts,
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
