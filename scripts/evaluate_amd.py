import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from finrag.documents import load_pdf_pages

data_dir = Path("data/external/financebench")

dataset = [
    json.loads(line)
    for line in (data_dir / "financebench_open_source.jsonl").read_text().splitlines()
    if line.strip()
]

questions = [
    item for item in dataset
    if item["doc_name"] == "AMD_2022_10K"
]

pages = load_pdf_pages(data_dir / "AMD_2022_10K.pdf")
results = {}

for name, stop_words in [
    ("original", None),
    ("no_stopwords", "english"),
]:
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(page.text for page in pages)

    positions = []

    for item in questions:
        q = vectorizer.transform([item["question"]])
        scores = (X @ q.T).toarray().ravel()
        ranking = scores.argsort()[::-1].tolist()

        gold = {
            e["evidence_page_num"] for e in item["evidence"]
        }

        position = min(ranking.index(p) + 1 for p in gold)
        positions.append(position)

    results[name] = {
        "positions": positions,
        **{
            f"hit_at_{k}": sum(p <= k for p in positions) / len(positions)
            for k in [1, 3, 5]
        },
    }

output = Path("results/amd_tfidf.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(results, indent=2) + "\n")

print(json.dumps(results, indent=2))
print(f"\nGuardado en {output}")