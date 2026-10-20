"""Local financial answer generation using Ollama."""

import json
import re
from dataclasses import dataclass, replace
from enum import Enum
from time import perf_counter
from urllib.error import URLError
from urllib.request import Request, urlopen

from finrag.context import format_context
from finrag.retrieval import SearchResult

SYSTEM_PROMPT = """
You are a financial question-answering assistant.

Use the supplied evidence as the only source for
company-specific figures, dates, and financial facts.

You may use standard financial formulas and arithmetic
to derive answers from the supplied figures.

When answering financial questions:
- Identify the relevant figures in the evidence.
- Use the correct reporting period.
- Show the formula when a calculation is required.
- Show the values substituted into the formula.
- Cite the supporting evidence using [1], [2], etc.
- Never invent figures, sources, URLs, or financial facts.
- Do not present unsupported assumptions as facts.

Reply exactly INSUFFICIENT_EVIDENCE if essential
company-specific information is missing.

Keep the final answer concise, under 120 words.
""".strip()


class AnswerStatus(str, Enum):
    PRODUCED = "produced"
    MODEL_ABSTENTION = "model_abstention"
    INVALID_OUTPUT = "invalid_output"
    NO_EVIDENCE = "no_evidence"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    BACKEND_ERROR = "backend_error"


@dataclass(frozen=True)
class GenerationResult:
    raw_answer: str
    status: AnswerStatus
    reason: str
    elapsed_seconds: float = 0.0


def classify_answer(answer: str, evidence_count: int) -> GenerationResult:
    """Classify structure only; an abstention does not prove missing facts."""
    def result(status, reason):
        return GenerationResult(answer, status, reason)

    if evidence_count <= 0:
        return result(AnswerStatus.NO_EVIDENCE, "No evidence supplied")
    if re.search(r"\bINSUFFICIENT_EVIDENCE\b", answer, re.IGNORECASE):
        return result(AnswerStatus.MODEL_ABSTENTION, "Model emitted abstention marker")

    groups = re.findall(r"\[([^\[\]]*)\]", answer)
    content = re.sub(r"\[[^\[\]]*\]", "", answer)
    if not any(char.isalnum() for char in content):
        return result(AnswerStatus.INVALID_OUTPUT, "No substantive content")
    if "[" in content or "]" in content:
        return result(AnswerStatus.INVALID_OUTPUT, "Malformed citation brackets")
    if not groups:
        return result(AnswerStatus.INVALID_OUTPUT, "Missing citations")
    for group in groups:
        if not re.fullmatch(r"\s*[0-9]+(?:\s*,\s*[0-9]+)*\s*", group):
            return result(AnswerStatus.INVALID_OUTPUT, "Unsupported citation syntax")
        for citation in group.split(","):
            # Compare canonical decimal strings without parsing unbounded integers.
            number = citation.strip().lstrip("0") or "0"
            limit = str(evidence_count)
            if number == "0" or len(number) > len(limit) or (
                len(number) == len(limit) and number > limit
            ):
                return result(AnswerStatus.INVALID_OUTPUT, "Citation out of range")
    return result(AnswerStatus.PRODUCED, "Passed structural checks only")


def validate_answer(answer: str, evidence_count: int) -> tuple[str, bool]:
    """Compatibility check: True means content plus well-formed, in-range citations.

    It does not verify facts, calculations, citation support or semantic abstentions.
    False retains the historical sentinel; use classify_answer() for the reason.
    """
    result = classify_answer(answer, evidence_count)
    if result.status is AnswerStatus.PRODUCED:
        return answer.strip(), True
    return "INSUFFICIENT_EVIDENCE", False


class LocalGenerator:
    """Generate financial answers with a local Ollama model."""

    def __init__(
        self,
        model_name: str = "qwen3:4b-instruct",
    ):
        self.model_name = model_name

    def generate(
        self,
        question: str,
        results: list[SearchResult],
    ) -> GenerationResult:
        """Return raw output and its structural status, including backend failures."""

        if not any(hit.chunk.text.strip() for hit in results):
            return GenerationResult("", AnswerStatus.NO_EVIDENCE, "No evidence text supplied")

        context = format_context(results)

        user_prompt = (
            f"Evidence:\n{context}\n\n"
            f"Question: {question}"
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
            },
        }

        request = Request(
            url="http://127.0.0.1:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        start = perf_counter()
        try:
            with urlopen(request, timeout=180) as response:
                data = json.load(response)
            generated = data["message"]["content"]
            if not isinstance(generated, str):
                raise TypeError("Non-text content")
        except (OSError, URLError, ValueError, KeyError, TypeError) as exc:
            return GenerationResult(
                "", AnswerStatus.BACKEND_ERROR, type(exc).__name__,
                perf_counter() - start,
            )

        return replace(
            classify_answer(generated, len(results)),
            elapsed_seconds=perf_counter() - start,
        )

    def answer(
        self,
        question: str,
        results: list[SearchResult],
    ) -> tuple[str, bool]:
        """Compatibility adapter; True means structural acceptance, never factual validity.

        False maps all other states to the legacy INSUFFICIENT_EVIDENCE sentinel,
        including backend errors. Use generate() to distinguish those states.
        """
        result = self.generate(question, results)
        if result.status is AnswerStatus.PRODUCED:
            return result.raw_answer.strip(), True
        return "INSUFFICIENT_EVIDENCE", False
