# FinanceBench experiments: Multi-Query, adaptive budgets, and a NO-GO decision

This report summarizes the FinRAG FinanceBench work from October 2026: the production retrieval
configurations, two experimental adaptive-evidence variants, a 12-question development set, and a
frozen 34-question holdout comparison. It records why the adaptive candidate was **not adopted**.

**Scope and status:** the adaptive selector, span packaging and paired holdout runner are
**experimental**. They live on a local research worktree and were never merged into `main`. The
production tree at commit `4a3f428` was not modified. This report is documentation, not a claim that
any configuration produces factually verified answers.

## 1. Research question and original limitations

**Question:** can FinRAG improve answer yield and citation traceability by replacing Multi-Query's
fixed two-chunks-per-page quota with a per-question evidence budget and one citation ID per
contiguous evidence span, without reducing the evidence the model needs?

The production path had five observed limitations:

1. **Fixed quota.** Multi-Query selects at most five pages and two chunks per page regardless of
   page content, so a page with one relevant passage reserves a second slot, while a dense table can
   be truncated mid-row.
2. **Citation identity.** One citation ID per 80-word chunk splits a financial table across several
   IDs. The model often cites the statement header or the first rows while the figure sits in a later
   chunk of the same table (observed on `01328` and `01279` before span packaging).
3. **High abstention.** Direct MiniLM abstained on 10/12 development questions; Multi-Query abstained
   on 26/34 holdout questions. Some abstentions were justified by missing context, others were not:
   `00222` (development set) had the inventory figure in context and still claimed it was absent.
4. **Produced is not correct.** Structurally valid answers included wrong-entity arithmetic (`10420`
   used parent Schedule I figures) and self-contradictory margin statements (`00684`).
5. **Retrieval is not the only bottleneck.** Page-level coverage improved substantially from direct
   retrieval to Multi-Query on the development set (1/12 to 7/12 gold-page hits), yet answer yield
   moved from 2/12 to 2/12 produced. Better page recall did not translate into better answers.

## 2. Configurations

| Configuration | Retrieval and context | Citation identity | Status |
| --- | --- | --- | --- |
| **Direct** | MiniLM, 25-word chunks, top-3 chunks, k=3 | One ID per chunk | Production baseline |
| **Multi-Query** | MiniLM, 80-word page chunks (15-word overlap), deterministic query planning, fused page ranking, `max_pages=5`, `chunks_per_page=2` | One ID per chunk | Production, selectable in the workspace |
| **Adaptive budget** (experimental) | Same pages, chunks, retriever and `max_pages`; a per-question chunk/word budget replaces the fixed per-page quota; selection is greedy inside the budget | One ID per chunk | Experimental only, not merged |
| **Adaptive + evidence spans** (frozen candidate) | Same as adaptive budget; contiguous selected chunks from one page are exposed as one evidence block | One ID per contiguous span; chunk range retained in the trace | Experimental only, not merged |

The adaptive budget is copied per question from a **retrieval-only Multi-Query pre-pass**, so the
candidate cannot exceed Multi-Query's chunk or word volume. Span packaging does not add, remove or
reorder selected text; it only changes the model-visible citation unit.

Experimental code (local worktree, no production files changed):

- `adaptive_budget_chunks.py` — budget allocation and greedy selection.
- `evidence_spans.py` — contiguous-span packaging and citation audit.
- `run_adaptive_budget_retrieval.py`, `run_adaptive_budget_qwen.py`,
  `run_adaptive_span_citations.py` — development-set runs.
- `run_holdout_paired.py` — frozen paired holdout runner; `holdout_report.py` — artifact validator.

These files are **not on `main`** and are not distributed with the application.

## 3. Development-set findings (12 questions, 2 reports)

Development set: 12 FinanceBench questions over AMD FY2022 and PepsiCo FY2022, MiniLM embeddings,
Qwen `qwen3:4b-instruct`, temperature 0, sequential inference. This is a small, previously inspected
sample used to design the candidate.

**Structural outcomes**

| Configuration | Produced | Model abstention | Invalid output | Backend error |
| --- | ---: | ---: | ---: | ---: |
| Direct MiniLM k=3 | 2/12 | 10/12 | 0/12 | 0/12 |
| Multi-Query | 2/12 | 9/12 | 1/12 | 0/12 |
| Adaptive budget | 6/12 | 6/12 | 0/12 | 0/12 |
| Adaptive + spans | 5/12 | 7/12 | 0/12 | 0/12 |

**Retrieval changes**

- Gold-page hits: direct 1/12, Multi-Query 7/12, adaptive 7/12. Adaptive reorders and repacks within
  the same ranked pages, so it does not change page-level recall.
- Annotated supporting quotes present: Multi-Query 12, adaptive budget 20, across the same 119
  selected chunks (9,290 to 9,118 context words). Fully covered questions rose from 4 to 7; among
  unambiguous questions, from 3 to 5. This was the declared pre-inference gate and it passed
  (+2 unambiguous fully supported questions, no losses, no budget overruns).
- Span packaging changed no selected text: 119 chunks and 9,118 words in both adaptive runs.

**Provisional semantic observations (AI-assisted, not human verification)**

- Direct: 2 produced; 0/2 fully correct and complete, 2/2 with no unsupported factual claim,
  3/3 citation groups supported. Two ambiguous questions were excluded from definitive grading.
- Multi-Query: `00995` correct and complete; `01009` correct but incomplete (omits New Zealand and
  China); `01328` invalid output with partially supported content. `01198` regressed from produced to
  abstention despite the revenue-driver sentence being in context.
- Adaptive + spans: 5 produced, 1 ambiguous and excluded, so 4 eligible; 2/4 fully correct and
  complete, 4/4 with no unsupported factual claim, 3/4 with fully supported citations. One of seven
  abstentions was a false abstention (`00222`).
- Span packaging fixed citation attribution for `01328` and `01279` (the cited span now contains the
  reported figures) but changed `00917` from produced to abstention. Answer yield moved 6 to 5.

The development signal was mixed: better annotated-quote coverage and one missing citation fix, but a
small sample, two ambiguous questions, and a packaging-induced status change.

## 4. Frozen holdout protocol (34 questions, 10 reports)

The holdout was run once, after the candidate was frozen:

- Candidate: experimental worktree commit `8d9420e` (Adaptive Budget + Evidence Spans).
- Baseline: production-compatible Multi-Query from `main` at `4a3f428`.
- Data: `data/eval/financebench_holdout.json`, 34 questions across 10 reports, SHA-256
  `fde6117b268f5b6fb169f5e1e382aaf586356a6a5938ff2070284d81a2e34735`.
- Model: `qwen3:4b-instruct` (Ollama ID `0edcdef34593`), temperature 0, `think=false`, unchanged
  system prompt, MiniLM `all-MiniLM-L6-v2`, `max_pages=5`, 80-word chunks with 15-word overlap.
- Budget: per question, the chunk and word totals from a retrieval-only Multi-Query pre-pass.
- Order: conditions alternated per question (even index Multi-Query first, odd index adaptive first).
- Warm-up: one inference excluded from paired metrics.
- Smoke test: one single-question `--limit 1` wiring run (`00476`) preceded the campaign; it is
  excluded from all metrics and no configuration changed afterward.
- Retries: up to three attempts only for named transport errors, with a five-second pause and an
  Ollama restart if the endpoint was down. Model abstentions and malformed outputs were never
  retried. No retries were needed.
- Gold pages and annotated evidence text were loaded only after all 68 generations finished.

Artifacts were written outside the repository (local run directory) and are not committed. The run
produced 68 final traces, a paired comparison, separate page- and chunk-level coverage, regression
and numeric-claim audits, a manifest with code/model/PDF hashes, and a 14-check validation file. All
checks passed: trace completeness, frozen configuration, alternating order, budget provenance,
span/text integrity, prompt hashes, citation ranges, and zero unresolved backend errors.

## 5. Exact structural metrics (holdout, n=34)

| Metric | Multi-Query | Adaptive + spans |
| --- | ---: | ---: |
| Produced | 8/34 | 10/34 |
| Model abstention | 26/34 | 23/34 |
| Invalid output | 0/34 | 1/34 |
| Backend error | 0/34 | 0/34 |
| Produced answers with in-range citations | 8/8 | 10/10 |
| Gold-page hit questions | 15/34 | 15/34 |
| Mean page recall | 0.353 | 0.353 |
| Mean chunk precision on gold pages | 0.088 | 0.139 |
| Mean chunk recall on gold pages | 0.121 | 0.201 |
| Annotated evidence spans present | 7/46 (15.2%) | 3/46 (6.5%) |
| Annotated evidence spans exactly reproduced | 5/46 (10.9%) | 3/46 (6.5%) |
| Questions with at least one annotated span | 7/34 | 3/34 |
| Questions with all annotated evidence | 4/34 | 2/34 |

Paired status transitions: 23 abstain→abstain, 3 abstain→produced (`00799`, `00956`, `00215`), 1
produced→invalid (`10420`), 7 produced→produced. Seven questions produced under both conditions; raw
answer text differed in all seven because the context differed.

Citation validity above is **structural only**: cited IDs exist and are in range. It does not show
that the cited text supports the claim. Numeric-token audits flagged 6 Multi-Query and 7 adaptive
produced answers with numbers absent from their cited evidence; those are review items, not proof of
error, because derived values may legitimately be absent.

## 6. Retrieval and annotated-evidence trade-offs

- **Page level:** identical. Both conditions retrieved a gold page for 15/34 questions with mean page
  recall 0.353. Both conditions draw from the same five ranked pages; adaptive changes which chunks
  are kept, not the page ranking.
- **Chunk level:** adaptive is more precise. Precision on gold pages 0.139 vs 0.088; recall 0.121 vs
  0.201.
- **Annotated evidence:** adaptive is worse. Only 3/46 annotated spans were present versus 7/46, and
  3/34 questions had any annotated span versus 7/34. This is the metric most directly tied to whether
  the model was given the passage the benchmark considers necessary.
- **Selection overlap:** the two conditions chose an identical chunk set for 0/34 questions; mean
  Jaccard overlap of selected chunks was 0.385. The budget bounds volume but does not force the same
  chunks, so this experiment compares selection, packaging and citation granularity together.
- **Context size:** selected words 25,864 → 25,070; prompt words 28,275 → 26,722; model-visible
  evidence items 374 → 217 after span merging.
- **Latency:** median generation 4.40 s → 4.51 s; total per-run time 144.9 s → 155.5 s (retrieval plus
  generation). No performance advantage is claimed.

The trade-off is the central result: the candidate spends its budget on fewer, higher-scoring chunks
and merges them into spans, which improves chunk precision but loses annotated passages that the
fixed two-chunks-per-page quota happened to include.

## 7. Semantic findings (provisional, AI-assisted)

An AI-assisted provisional review graded every produced holdout output and the one invalid output.
This was **not human verification**, and the grades must not be converted into an accuracy percentage.

**Produced outputs: 18 reviewed** (8 Multi-Query + 10 adaptive), pooled across conditions because the
assessment does not publish per-arm semantic accuracy:

| Grade | Count | Denominator |
| --- | ---: | ---: |
| Correct and complete | 9 | 18 |
| Correct but incomplete | 1 | 18 |
| Incorrect | 6 | 18 |
| Unsupported | 2 | 18 |

**Eligible subset after excluding ambiguous questions:** 8 produced outputs. Seven question IDs were
treated as ambiguous or definition-dependent (`00215`, `00566`, `00684`, `00799`, `00956`, `01912`,
`10420`), which removes 10 of the 18 produced outputs and the single invalid output from definitive
grading.

| Eligible metric | Result |
| --- | ---: |
| Fully correct and complete | 6/8 |
| No unsupported factual claim | 6/8 |
| Fully supported citations | 7/8 |
| Citation support groups | 26/32 |

**Abstentions: 49 total** (26 Multi-Query + 23 adaptive): 31 justified by missing context, 3 false
abstentions, 15 uncertain. The 3 false abstentions are the identified avoidable cases; the 15
uncertain cases mean the false-abstention rate is not settled. This is not an accuracy measure.

Per-output grades (short causes only; full answers and traces are not reproduced):

| Condition | ID | Grade |
| --- | --- | --- |
| Multi-Query | `01028` | Unsupported (over-specified geographies; incomplete relative to segment answer) |
| Multi-Query | `01351` | Correct and complete |
| Multi-Query | `01148` | Correct and complete |
| Multi-Query | `00684` | Incorrect (contradictory margin direction) |
| Multi-Query | `10420` | Incorrect (parent Schedule I figures instead of consolidated) |
| Multi-Query | `01244` | Correct and complete |
| Multi-Query | `01912` | Correct and complete |
| Multi-Query | `00566` | Incorrect (debt direction contradicted) |
| Adaptive | `01028` | Unsupported (adds unsupported countries; incomplete) |
| Adaptive | `01351` | Correct and complete |
| Adaptive | `00799` | Correct and complete under its stated quick-ratio formula |
| Adaptive | `01148` | Correct and complete |
| Adaptive | `00684` | Incorrect (contradictory; cited span lacks the gross-profit figures) |
| Adaptive | `01244` | Correct and complete |
| Adaptive | `00956` | Incorrect (asserts positive earnings growth while net earnings declined) |
| Adaptive | `01912` | Correct and complete |
| Adaptive | `00215` | Correct but incomplete (omits the capital-intensity ratio) |
| Adaptive | `00566` | Incorrect (debt direction contradicted) |
| Adaptive | `10420` (invalid) | Incorrect content and malformed nested citation |

The three gained productions were split: one correct and complete under a stated formula (`00799`),
one correct but incomplete (`00215`), one incorrect (`00956`). All three are ambiguous items excluded
from the 6/8 eligible result. The single lost production (`10420`) was already wrong under
Multi-Query and additionally failed citation syntax under the span format.

## 8. Specific failure examples and causes

- **Wrong entity / scope** — `10420` computed return on assets from parent-company Schedule I figures
  rather than consolidated statements. Cause: retrieval returned a plausible but wrong-entity page;
  the selector cannot distinguish parent and consolidated tables.
- **Self-contradiction** — `00684` says the gross margin is improving and then reports a decline from
  19.39% to 18.53%. Cause: generation composes an opening verdict before computing the comparison;
  citation support does not check the verdict against the numbers.
- **False subclaim inside a supported answer** — `00956` reports J&J net earnings falling from
  $20,878M to $17,941M, then calls earnings growth positive. Cause: sentence-level assertion not
  reconciled with the cited income statement.
- **Direction reversal** — `00566` reports a debt decrease as an increase and reverses the short-term
  debt change. Cause: numeric comparison and sign handling in generation.
- **Annotation/target mismatch** — `01028` lists countries while the benchmark answer expects four
  geographic segments. Cause: ambiguous question granularity plus model over-specification.
- **Formatting-only loss** — `10420` under the span format wrote a nested prose bracket
  (`[Supporting evidence: [1] …]`), so `classify_answer` correctly returned `invalid_output` even
  though the arithmetic matched the Multi-Query answer. Cause: citation syntax imitation; not a
  retrieval regression.
- **False abstention** — `00222` (development set) claimed inventories were not separated although the
  selected page contained the inventory line. Cause: generation ignores available evidence; unchanged
  by the candidate.

## 9. Limitations

- **AI-assisted provisional assessment.** All semantic grades come from an AI-assisted review, not
  independent human adjudication. They are directional evidence, not audited labels.
- **Ambiguous financial questions.** Seven holdout question IDs depend on definitions or expected
  answer granularity (quick-ratio formula, debt scope, geography segments, income-statement versus
  cash-flow attribution). They were excluded from the eligible 8, but they still affect the produced
  counts and the failure examples.
- **Incomplete benchmark evidence annotations.** Gold pages and evidence text are sparse. The
  page/chunk coverage audit measures overlap with the annotation, not whether a different selected
  passage could also support an answer.
- **Small development sample.** The 12-question development set covers two reports and was used to
  design the candidate; its structural gains did not persist as factual gains.
- **Previously inspected holdout retrieval results.** Holdout page-level retrieval metrics had been
  inspected before this generation campaign, and one holdout question (`00476`) was used for a
  single-question wiring smoke test. The paired generation configuration was frozen in advance, no
  holdout answers were used for tuning, and gold data was loaded only after inference, but the
  holdout is not pristine for retrieval-method selection.
- **No verified factual-accuracy improvement.** Produced counts increased from 8/34 to 10/34, but
  annotated evidence presence fell, three gained answers had no annotated evidence span in context, and
  provisional grading found incorrect and unsupported outputs under both conditions. No factual
  accuracy rate is claimed for either arm.
- **Single model, single run, one prompt.** `qwen3:4b-instruct` at temperature 0 was run once per
  question per condition. No variance estimate, no second model, no human review.
- **Structural validity is not support.** In-range citations and numeric-claim audits are checks of
  form and presence; they do not verify that cited text entails the claim.

## 10. Decision: NO-GO

**Do not adopt Adaptive Budget + Evidence Spans in production.**

The candidate did not demonstrate a factual-accuracy improvement. It increased structurally produced
answers (8/34 to 10/34) while annotated evidence presence roughly halved (7/46 to 3/46 spans), and its
extra productions included an incorrect answer and a correct-but-incomplete answer that are excluded
from the eligible denominator. Its only lost production was classified invalid for citation format,
and that answer was already wrong under Multi-Query. The chunk-level precision gain is real but
narrower than the evidence-coverage loss.

Production keeps the existing Multi-Query path. The experimental selector, span packaging and holdout
runner remain unmerged local research code; no production source, prompt, dependency or model was
changed. The decision rests on evidence coverage, provisional semantic grades and one frozen run, not
on a demonstrated accuracy gain.

## 11. Reproduction

The public retrieval evaluation uses scripts on `main`. The adaptive and paired-Qwen runs use the
local experimental worktree; those scripts are not on `main`. The commands below reproduce those runs
from the scripts' recorded defaults and the flags used for the paired holdout.

**Public data and retrieval baseline (production `main`):**

```bash
python scripts/download_financebench_holdout.py
python scripts/evaluate_financebench_pilot.py --split holdout
```

Downloaded PDFs go to `data/external/financebench/`, which is ignored and not redistributed. The
retrieval evaluation writes `results/financebench_holdout.json`; the committed summary is also
visible in the README table.

**Development-set runs (experimental worktree, `PYTHONPATH=src`):**

```bash
PYTHONPATH=src python scripts/run_real_qwen_financebench.py --out results/real_qwen_financebench
PYTHONPATH=src python scripts/run_multiquery_financebench.py --out results/multiquery
PYTHONPATH=src python scripts/run_chunk_selection_retrieval.py --out results/chunk_selection
PYTHONPATH=src python scripts/run_adaptive_budget_retrieval.py --out results/adaptive_budget
PYTHONPATH=src python scripts/run_adaptive_budget_qwen.py --out results/adaptive_budget
PYTHONPATH=src python scripts/run_adaptive_span_citations.py \
  --out results/adaptive_budget_citations \
  --retrieval-metrics results/adaptive_budget/retrieval_only.json \
  --multiquery-baseline results/multiquery/traces.jsonl \
  --previous-adaptive results/adaptive_budget/traces.jsonl
```

The two retrieval-only diagnostics read a local evidence-review packet derived from FinanceBench
annotations (`.local/financebench_evidence_review/evidence_review_packet.json`). That packet is
ignored, not distributed, and must be rebuilt before those quote-coverage tables can be reproduced
from a clean clone. The three inference runners do not read the packet directly; only the retrieval
diagnostics do.

**Frozen 34-question paired holdout (experimental worktree):**

```bash
PYTHONPATH=src python scripts/run_holdout_paired.py \
  --out results/holdout_final \
  --corpus data/external/financebench \
  --holdout data/eval/financebench_holdout.json \
  --open-source data/external/financebench/financebench_open_source.jsonl
python scripts/holdout_report.py --out results/holdout_final
```

`run_holdout_paired.py` requires a local Ollama service with `qwen3:4b-instruct`. It performs no
network download and uses only the local corpus. `holdout_report.py` re-checks trace completeness,
budgets, spans, prompt hashes and citation ranges before writing a summary.

The semantic assessments referenced in this report were produced in a local, ignored review
directory (`.local/financebench_evidence_review/`). They are not part of the repository and are not
required to reproduce the structural results.

For the separate synthetic answer-evaluation fixtures, see
[`answer_evaluation.md`](answer_evaluation.md). Those fixtures are simulated regression cases, not
model measurements, and are unrelated to the FinanceBench numbers above.
