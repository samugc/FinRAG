import re

from finrag.retrieval import SearchResult


def validate_answer(answer: str, evidence_count: int) -> tuple[str, bool]:
    answer = answer.strip()
    citations = re.findall(r"\[(\d+)\]", answer)
    content = re.sub(r"\[\d+\]", "", answer)
    if (
        not evidence_count
        or not any(char.isalnum() for char in content)
        or re.fullmatch(r"INSUFFICIENT_EVIDENCE[.!]?", answer, re.IGNORECASE)
        or any(not 1 <= int(citation) <= evidence_count for citation in citations)
    ):
        return "INSUFFICIENT_EVIDENCE", False
    return answer, True



_NUMBER = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)")


def validate_numeric_evidence(
    answer: str,
    results: list[SearchResult],
) -> tuple[str, bool]:
    """Conservatively reject numeric claims absent from document text.

    This catches fabricated values and retrieval-score echoes. It is NOT
    a factuality verifier: copied but irrelevant numbers can still pass.
    """
    content = re.sub(r"\[\d+\]", "", answer)
    numbers = set(_NUMBER.findall(content))
    observed = {
        value
        for hit in results
        for value in _NUMBER.findall(hit.chunk.text)
    }
    if numbers - observed:
        return "INSUFFICIENT_EVIDENCE", False
    return answer, True


class LocalGenerator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
        )
        self.model.to(self.device)

    def answer(
        self,
        question: str,
        results: list[SearchResult],
    ) -> tuple[str, bool]:
        if not results:
            return "INSUFFICIENT_EVIDENCE", False

        # Never expose similarity scores as evidence for the generator:
        # the model may otherwise mistake a retrieval score for a financial value.
        context = "\n\n".join(
            f"[{index}] {hit.chunk.source} (chunk {hit.chunk.chunk_id})\n"
            f"{hit.chunk.text}"
            for index, hit in enumerate(results, start=1)
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied evidence. "
                    "Do not use outside knowledge. "
                    "If the evidence does not contain enough information, "
                    "reply exactly: INSUFFICIENT_EVIDENCE. "
                    "Otherwise answer briefly and cite evidence as [1], [2], etc."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Evidence:\n{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
        )

        generated = output[0][inputs["input_ids"].shape[1]:]

        answer = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        checked, valid = validate_answer(answer, len(results))
        if not valid:
            return checked, valid
        return validate_numeric_evidence(checked, results)
