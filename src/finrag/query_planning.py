
"""Financial query decomposition for retrieval."""

import re

FINANCIAL_CONCEPTS = {
    r"\bquick ratio\b": [
        "consolidated balance sheet cash short-term investments accounts receivable",
        "consolidated balance sheet total current liabilities",
    ],
    r"\bebitda\b": [
        "consolidated income statement operating income",
        "cash flow statement depreciation and amortization",
    ],
    r"\b(?:capex|capital expenditures?)\b": [
        "cash flow statement capital expenditures purchases of property plant equipment",
    ],
}


def plan_queries(question: str) -> list[str]:
    """Expand a financial question into evidence-oriented searches."""
    question = " ".join(question.split())

    if not question:
        raise ValueError("Question must not be empty")

    queries = [question]

    year = re.search(
        r"\b(?:FY\s*)?20\d{2}\b|\bFY\s*\d{2}\b",
        question,
        re.IGNORECASE,
    )
    year_text = year.group(0) if year else ""

    for pattern, concepts in FINANCIAL_CONCEPTS.items():
        if re.search(pattern, question, re.IGNORECASE):
            for concept in concepts:
                query = f"{year_text} {concept}".strip()
                queries.append(query)

    return list(dict.fromkeys(queries))
