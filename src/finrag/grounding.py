"""Reporting-period availability checks, not value/column extraction."""

import re
from datetime import datetime

_YEAR = r"(?:19|20)[0-9]{2}"
_MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_DATE = re.compile(rf"\b({_MONTH})\s+(\d{{1,2}}),?\s+({_YEAR})\b", re.IGNORECASE)
_STATEMENT = re.compile(r"(?:consolidated\s+)?(?:balance\s+sheets?|statements?\s+of\s+(?:income|operations|cash\s+flows|earnings|financial\s+position))", re.IGNORECASE)
_FISCAL = re.compile(rf"\b(?:fiscal\s+year|FY)\s*({_YEAR})\b|\b(?:year|years)\s+ended\s+({_MONTH}\s+\d{{1,2}},?\s+{_YEAR})", re.IGNORECASE)


def reporting_periods(text):
    """Recognize explicit reporting dates; plain four-digit numbers never qualify.

    Unknown layouts stay unknown. A positive result establishes that a period
    is named in reporting context, not that any given amount belongs to it.
    """
    found = []
    for statement in _STATEMENT.finditer(text):
        tail = text[statement.start():statement.end() + 240]
        # Restrict date recognition to the heading, before financial rows/units.
        heading = re.split(r"\(\s*in\s+|\bASSETS\b|\bCurrent assets\b", tail, maxsplit=1, flags=re.IGNORECASE)[0]
        for date in _DATE.finditer(heading):
            try:
                datetime.strptime(" ".join(date.groups()), "%B %d %Y")
            except ValueError:
                continue
            found.append({"year": int(date.group(3)), "quote": heading[:date.end()].strip(),
                          "basis": "statement_heading"})
    for fiscal in _FISCAL.finditer(text):
        year = fiscal.group(1)
        if year is None:
            date = _DATE.search(fiscal.group(2))
            if not date:
                continue
            try:
                datetime.strptime(" ".join(date.groups()), "%B %d %Y")
            except ValueError:
                continue
            year = date.group(3)
        found.append({"year": int(year), "quote": fiscal.group().strip(), "basis": "explicit_reporting_period"})
    return found


def check_periods(question, evidence, pages, documents):
    requested = sorted({int(y) for y in re.findall(rf"(?<!\d)({_YEAR})(?!\d)", question)})
    result = {"status": "not_requested", "requested_years": requested,
              "verified_years": [], "references": [], "warnings": []}
    if not requested or not evidence:
        return result
    sources = {hit.chunk.document_source or hit.chunk.source for hit in evidence}
    references = []
    unknown_heading = False
    for (source, page_index), text in pages.items():
        if source in sources:
            for match in _STATEMENT.finditer(text):
                heading = re.split(r"\(\s*in\s+|\bASSETS\b|\bCurrent assets\b",
                                   text[match.end():match.end() + 100], maxsplit=1, flags=re.IGNORECASE)[0]
                unknown_heading |= bool(re.search(
                    r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b|"
                    r"\b(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2}\b",
                    heading, re.IGNORECASE))
            references.extend({**p, "source": source, "page_index": page_index} for p in reporting_periods(text))
    pdf_sources = {source for source, _ in pages}
    for doc in documents:
        if doc.source in sources and doc.source not in pdf_sources:
            references.extend({**p, "source": doc.source, "page_index": None} for p in reporting_periods(doc.text))
    verified = sorted({p["year"] for p in references})
    result["verified_years"] = verified
    selected_pages = {(hit.chunk.document_source or hit.chunk.source, hit.chunk.page_index) for hit in evidence}
    for year in requested:
        for source in sorted(sources):
            candidates = [p for p in references if p["year"] == year and p["source"] == source]
            if candidates:
                result["references"].append(min(candidates, key=lambda p: (
                    p["basis"] != "statement_heading",
                    (p["source"], p["page_index"]) not in selected_pages,
                    p["page_index"] if p["page_index"] is not None else -1,
                )))
    if set(requested) <= set(verified):
        result["status"] = "available"
        result["warnings"] = ["Reporting period located; value-to-column correspondence is not verified."]
    elif verified and not unknown_heading:
        result["status"] = "insufficient_context"
        result["warnings"] = ["Requested period not verified in recognized reporting context; this does not prove absence from the PDF."]
    else:
        result["status"] = "unverified"
        result["warnings"] = ["Reporting-period format not recognized. Query was not blocked; a year-like amount is not period evidence."]
    return result
