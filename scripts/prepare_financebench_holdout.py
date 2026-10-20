
import json
from collections import defaultdict
from pathlib import Path

base = Path("data/external/financebench")
pilot = json.loads(
    Path("data/eval/financebench_pilot.json").read_text()
)

questions = [
    json.loads(line)
    for line in (base / "financebench_open_source.jsonl")
    .read_text().splitlines()
    if line.strip()
]

# Exclude companies already used during development
pilot_documents = set(pilot["documents"])
excluded_companies = {
    q["company"]
    for q in questions
    if q["doc_name"] in pilot_documents
}

by_document = defaultdict(list)

for q in questions:
    if q["company"] not in excluded_companies:
        by_document[q["doc_name"]].append(q)

# Fixed selection rule: most questions, then alphabetical
candidates = sorted(
    by_document,
    key=lambda name: (-len(by_document[name]), name),
)

selected = []
selected_companies = set()

for name in candidates:
    company = by_document[name][0]["company"]

    if company in selected_companies:
        continue

    selected.append(name)
    selected_companies.add(company)

    if len(selected) == 10:
        break

holdout_questions = []

for name in selected:
    for q in by_document[name]:
        holdout_questions.append({
            "financebench_id": q["financebench_id"],
            "doc_name": name,
            "question_type": q["question_type"],
            "gold_pages": sorted({
                e["evidence_page_num"]
                for e in q["evidence"]
            }),
        })

output = Path("data/eval/financebench_holdout.json")
output.write_text(
    json.dumps({
        "documents": selected,
        "questions": holdout_questions,
    }, indent=2) + "\n"
)

print("Documentos:", len(selected))
print("Preguntas:", len(holdout_questions))
print("Empresas excluidas:", sorted(excluded_companies))

print("\nPDF pendientes:")
for name in selected:
    path = base / f"{name}.pdf"
    if not path.exists():
        print(name)

print("\nGuardado:", output)
