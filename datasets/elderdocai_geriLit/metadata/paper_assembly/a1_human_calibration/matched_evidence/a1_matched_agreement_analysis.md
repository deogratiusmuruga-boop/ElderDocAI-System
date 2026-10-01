# A1 - Matched-Evidence Human-Judge Agreement Analysis (frozen protocol)

Protocol: `1.0-matched-evidence` (v1.0-matched-evidence)

- Source annotation sheet: `C:\Users\chosun\Documents\ElderDocAI-System\datasets\elderdocai_geriLit\metadata\paper_assembly\a1_human_calibration\matched_evidence\a1_matched_annotation_sheet_completed.csv`
- Annotation-sheet SHA-256: `70506d5d2fdc631f`
- Experiment 1 source: `C:\Users\chosun\Documents\ElderDocAI-System\datasets\elderdocai_geriLit\metadata\experiment1_generation_evaluation\experiment1_per_question_run1.json`
- n = 20
- Verification all-pass: **True**

## Verification checks

- exactly_20_rows: True
- no_duplicate_ids: True
- identical_id_ordering: True
- identical_question: True
- identical_answer: True
- identical_evidence_full: True
- no_evidence_truncation: True
- expected_columns_only: True
- human_columns_populated: True
- rating_domains_valid: True
- all_ids_in_run1: True
- frozen_sheet_sha256: `613261530541d0f5`
- frozen_sheet_unchanged: True
- all_pass: True

## Agreement statistics (n = 20)

| metric | exact n | exact % | Po | Pe | within-1-bin % | Cohen's kappa |
|--------|---------|---------|----|----|--------------|--------------|
| faithfulness | 12 | 60.0 | 0.6000 | 0.5325 | 75.0 | degenerate - zero-variance marginal |
| answer_relevance | 11 | 55.0 | 0.5500 | 0.4800 | 75.0 | 0.1346 |
| evidence_support | 13 | 65.0 | 0.6500 | 0.6725 | 85.0 | degenerate - zero-variance marginal |
| unsupported_claim | 15 | 75.0 | 0.7500 | 0.7500 | None | degenerate - zero-variance marginal |

### κ degeneracy note

Per the preregistered rule, kappa is reported as degenerate when a rater marginal is zero-variance (>= 19/20 in one category) or Pe >= 0.90. Degenerate kappa is not interpreted as ordinary disagreement; exact agreement / Po / Pe are reported instead.

### Human vs automated marginals

| metric | human marginal | automated marginal |

| faithfulness | {'0.75': 1, '1.0': 19} | {'0.0': 4, '0.25': 1, '0.75': 4, '1.0': 11} |
| answer_relevance | {'0.75': 4, '1.0': 16} | {'0.0': 5, '0.75': 4, '1.0': 11} |
| evidence_support | {'0.75': 1, '1.0': 19} | {'0.25': 2, '0.5': 1, '0.75': 3, '1.0': 14} |
| unsupported_claim | {'0.0': 20} | {'0': 15, '1': 5} |

### Unsupported-claim confusion (human flag vs run-1 hallucination flag)

| TP | FP | TN | FN |
|----|----|----|----|
| 0 | 0 | 15 | 5 |

- Analysis timestamp (UTC): 2026-09-29T08:04:49+00:00
- Support-bin mapping: [[0.0, 0.125, 0.0], [0.125, 0.375, 0.25], [0.375, 0.625, 0.5], [0.625, 0.875, 0.75], [0.875, 1.0, 1.0]]
