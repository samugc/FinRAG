# Answer Evaluation (FinRAG v1)

This document describes the synthetic answer-evaluation v1 golden dataset and
deterministic regression fixtures owned by this worktree. It is the companion to
`data/eval/answer_golden_v1.json`, `tests/fixtures/answer_eval/`, and the
future evaluator owned by FINRAG 03.

## Files

| Path | Role |
| --- | --- |
| `data/eval/answer_golden_v1.json` | Versioned synthetic gold questions, expected decisions, required evidence |
| `tests/fixtures/answer_eval/traces.jsonl` | 16 simulated answer traces |
| `tests/fixtures/answer_eval/assessments.jsonl` | 14 fixture-authored semantic assessments |
| `tests/fixtures/answer_eval/expected_cases.json` | Per-case attribution and metric assertions |
| `data/raw/*.txt` | Synthetic source corpus used by all quotes and hashes |

## Contracts

### Gold (`finrag.answer_gold.v1`)

- `schema_version`, `dataset_id`, `corpus`, `questions`.
- `corpus` entries contain `document_id`, `source`, `sha256`, `provenance`.
- `questions` contain `question_id`, `question`, `document_id`, `answerability`,
  `expected_decision`, `expected_answer`, `required_evidence`, `review_status`,
  and optional `notes`.
- `required_evidence` entries contain `evidence_id`, `document_id`, `chunk_id`,
  `page_index`, `quote`, and `review_status`.
- Every `quote` must be a verbatim substring of the referenced
  `data/raw/*.txt` file.
- `chunk_id` values refer to the production chunking used by FinRAG:
  `chunk_document(..., chunk_size=25, overlap=5)`.

Gold corpus hashes (actual SHA-256):

| document_id | source | sha256 |
| --- | --- | --- |
| `sample_report` | `data/raw/sample_report.txt` | `059082ed39658209264c0e34361374e4cf1bfbc98bc4707906fbc1e7f1ec9f2c` |
| `cash_flow` | `data/raw/cash_flow.txt` | `957ce545bc6caedc8fc3378e9c5ca7e75d659dbcc601096d9925ee2e70b610f2` |
| `risk_report` | `data/raw/risk_report.txt` | `8534c5f440eafcd8a635cd58d337dd9ae78e75cc239d6cf7a25d2eb1f5fe992c` |
| `outlook` | `data/raw/outlook.txt` | `66bbf37995fab03579849cf95e1fd7b65a7321dd96951a979324912e4631a722` |

Gold file SHA-256: `b1640875d4333adcf0cfe386a823faf370bb3dc663de505ca6a531acfa7c2cb2`.

### Traces (`finrag.answer_trace.v1`)

- One trace per gold question (16 total).
- `trace_id`, `run_id`, `question_id`, `gold_sha256`, `config`, `baseline`,
  `result`, `evidence`, `candidate_evidence`, `error`.
- `result.status` is one of `produced`, `model_abstention`, `invalid_output`,
  `no_evidence`, `insufficient_context`, or `backend_error`.
- `evidence` is an ordered list. `citation_id` values are contiguous from `1`.
- Each evidence entry contains `text` and `text_sha256`; hashes are calculated
  from the exact trace text.
- `candidate_evidence` is either `null` or an ordered list of
  `{document_id, chunk_id, page_index}`.
- Source identity: gold `corpus[].source` values are repository-relative paths
  (`data/raw/...txt`) and are used for hash verification. Trace `config.source`
  and evidence `source` values are filenames (`...txt`), matching production
  `Document.source`. `document_id` is the stable identity used to join gold,
  traces, and assessments; source strings are provenance/display metadata and
  need not be byte-identical.
- Simulated fixture traces use `run_id = fixture-answer-eval-v1` and
  `baseline.git_commit = 522255530a80463ee4fce9cf4f77d9b646d805c9`,
  `baseline.git_dirty = false`.

### Assessments (`finrag.answer_assessment.v1`)

- 14 assessments (all questions except `syn-unans-tax-rate` and
  `syn-unans-offices`).
- `assessment_id`, `trace_id`, `question_id`, `reviewer_id`, `reviewed_at`,
  `context_sufficient`, `decision`, `answer_correct`, `claims`, `citations`,
  `secondary_labels`, `notes`.
- `claims` use `claim_id`, `text`, `support`
  (`supported`, `unsupported`, `contradicted`).
- `citations` use `claim_id`, `citation_id`, `evidence_sha256`, `support`
  (`supports`, `does_not_support`, `irrelevant`).
- `context_sufficient` is `sufficient`, `insufficient`, or `not_applicable`.
- `decision` is `answered` or `abstained`.
- `answer_correct` is `correct`, `incorrect`, or `not_applicable`.
- Assessment citations reference exact evidence by `evidence_sha256`.

### Expected cases (`finrag.answer_expected_cases.v1`)

- One case per gold question (16 total).
- `expected_primary` is one of `null`, `unknown`, `retrieval`,
  `context_assembly`, `generation`, `citation`, or `backend`.
- `secondary_contains` lists expected secondary labels.
- `assertions` contains required-evidence coverage, context sufficiency,
  decision, correctness, claim counts, citation syntax, citation pairs,
  abstention error, and the diagnostic flag
  `coverage_complete_but_context_insufficient`.

## Simulated fixture caveat

These traces and assessments are **deterministic regression fixtures**, not
measured inference. They were authored to exercise specific pipeline failure
modes. They must not be reported as model performance, latency, throughput, or
accuracy. No Ollama inference, network download, or new dependency was used to
create them. Aggregate metric values are intentionally not copied from any
illustrative document; only per-case oracles are provided. Status values in the
matrix are structural classifications of simulated fixture traces, not observed
Ollama outputs.

## Sixteen-case coverage matrix (simulated fixtures)

| question_id | source | trace status | context | decision | correctness | expected primary | scenario |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `syn-revenue-2025` | sample_report | produced | sufficient | answered | correct | null | correct answer |
| `syn-operating-profit-2025` | sample_report | model_abstention | insufficient | abstained | not_applicable | unknown | missing required evidence / false abstention |
| `syn-net-debt-2025` | sample_report | model_abstention | insufficient | abstained | not_applicable | context_assembly | candidate present but not in context |
| `syn-headcount-2025` | sample_report | model_abstention | insufficient | abstained | not_applicable | retrieval | candidate evidence absent |
| `syn-ocf-2025` | cash_flow | produced | sufficient | answered | incorrect | generation | contradicted numeric |
| `syn-fcf-2025` | cash_flow | produced | sufficient | answered | incorrect | generation | unsupported numeric |
| `syn-cash-2025` | cash_flow | invalid_output | sufficient | answered | correct | citation | malformed citation syntax |
| `syn-customer-concentration` | risk_report | produced | sufficient | answered | correct | citation | irrelevant citation |
| `syn-fx-exposure` | risk_report | invalid_output | sufficient | answered | correct | citation | out-of-range citation |
| `syn-workforce-expansion-2026` | outlook | invalid_output | sufficient | answered | correct | citation | missing citation |
| `syn-revenue-growth-2026` | outlook | model_abstention | insufficient | abstained | not_applicable | unknown | coverage-complete but semantic context insufficient |
| `syn-no-exact-2026-forecast` | outlook | produced | sufficient | answered | incorrect | generation | contradicted numeric |
| `syn-unans-dividend` | sample_report | model_abstention | not_applicable | abstained | not_applicable | null | correct abstention |
| `syn-unans-ceo` | sample_report | produced | not_applicable | answered | incorrect | generation | false answer |
| `syn-unans-tax-rate` | cash_flow | backend_error | n/a | n/a | n/a | backend | backend failure, no assessment |
| `syn-unans-offices` | sample_report | produced | n/a | n/a | n/a | unknown | produced answer, no assessment (exercises `unknown`) |

Counts (simulated traces): 7 `produced`, 5 `model_abstention`,
3 `invalid_output`, 1 `backend_error`.
Assessments: 14 (all except `syn-unans-tax-rate` and `syn-unans-offices`).

### Diagnostic case (`syn-revenue-growth-2026`)

This case intentionally uses a truncated evidence snapshot containing only
`revenue growth to remain positive in 2026`. Required-quote coverage is complete
(1 of 1), but the snapshot omits the subject attribution ("Management expects").
The assessment therefore marks semantic context as `insufficient`, and the
expected case sets `coverage_complete_but_context_insufficient = true`.
This is deliberate and must not be "fixed" into a normal sufficient case.

## Metric limitations

- `classify_answer()` in `src/finrag/generation.py` validates structure only:
  abstention markers, citation syntax, and citation range. It does not verify
  facts, calculations, citation support, or semantic abstentions.
- The gold dataset provides per-case oracles only. It does not provide aggregate
  measured performance values.
- Semantic assessments are fixture-authored and not model-generated.
- `unknown` is used where attribution cannot be determined from the available
  evidence; it is not a measured failure rate.
- Citation syntax validity is separate from citation support. Cases such as
  `syn-cash-2025`, `syn-fx-exposure`, and `syn-workforce-expansion-2026` have
  semantically correct answers but invalid or missing citation syntax.

## Future evaluator

Run the future evaluator (owned by FINRAG 03) with:

```sh
python scripts/evaluate_answer_quality.py \
  --gold data/eval/answer_golden_v1.json \
  --traces tests/fixtures/answer_eval/traces.jsonl \
  --assessments tests/fixtures/answer_eval/assessments.jsonl \
  --out results/answer_eval/report.json
```

The evaluator is not part of this worktree's ownership and is not included here.
Its canonical CLI output file is `report.json`.