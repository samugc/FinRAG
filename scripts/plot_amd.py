import json
from pathlib import Path

import matplotlib.pyplot as plt

results = json.loads(
    Path("results/amd_tfidf.json").read_text()
)

metrics = ["hit_at_1", "hit_at_3", "hit_at_5"]
labels = ["Hit@1", "Hit@3", "Hit@5"]

x = list(range(len(metrics)))
width = 0.35

original = [results["original"][m] for m in metrics]
filtered = [results["no_stopwords"][m] for m in metrics]

fig, ax = plt.subplots(figsize=(8, 5))

bars1 = ax.bar(
    [i - width / 2 for i in x],
    original,
    width,
    label="TF-IDF original",
)

bars2 = ax.bar(
    [i + width / 2 for i in x],
    filtered,
    width,
    label="Sin stopwords",
)

ax.set_xticks(x, labels)
ax.set_ylim(0, 0.55)
ax.set_ylabel("Hit@k")
ax.set_title("FinanceBench: AMD 2022 (7 questions)")
ax.legend()
ax.grid(axis="y", alpha=0.2)

for bars in [bars1, bars2]:
    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=3,
    )

output = Path("results/figures/amd_tfidf.png")
output.parent.mkdir(parents=True, exist_ok=True)

fig.tight_layout()
fig.savefig(output, dpi=200)
plt.close(fig)

print(f"Figura guardada: {output}")

# TF-IDF vs MiniLM on AMD chunks
comparison = json.loads(
    Path("results/amd_chunks_comparison.json").read_text()
)

tfidf = [100 * comparison["tfidf"][m] for m in metrics]
minilm = [100 * comparison["minilm"][m] for m in metrics]

fig, ax = plt.subplots(figsize=(8, 5))

bars1 = ax.bar(
    [i - width / 2 for i in x],
    tfidf, width, label="TF-IDF"
)

bars2 = ax.bar(
    [i + width / 2 for i in x],
    minilm, width, label="MiniLM"
)

ax.set_xticks(x, labels)
ax.set_ylim(0, 85)
ax.set_ylabel("Hit@k (%)")
ax.set_title("FinanceBench - AMD 2022 (7 questions)")
ax.legend()
ax.grid(axis="y", alpha=0.2)

for bars in [bars1, bars2]:
    ax.bar_label(bars, fmt="%.1f%%", padding=3)

output = Path("results/figures/amd_tfidf_vs_minilm.png")

fig.tight_layout()
fig.savefig(output, dpi=200)
plt.close(fig)

print(f"Figura guardada: {output}")