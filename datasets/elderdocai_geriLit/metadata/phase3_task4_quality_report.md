# ElderDocAI-GeriLit dev-v0.1 — Phase 3 Task 4 Quality Report

> **Development chunking parameters only** (see `chunking_specification.md`). No retrieval/QA tuning was performed; frequencies below are structural descriptions, not retrieval performance.

## Input corpus

- Articles: **500**
- Total chunks: **17930**

## Development chunking parameters

| Parameter | Value |
|---|---:|
| TARGET_WORDS | 220 |
| MAX_WORDS | 300 |
| MIN_CHUNK_WORDS | 50 |
| MIN_MERGE_WORDS | 100 |
| OVERLAP_SENTENCES | 1 |

## Chunk word-count statistics

min 1 · P25 95.0 · median 150.0 · mean 158.39 · P75 200.0 · P90 260.0 · P95 290.0 · max 4548

## Chunks per article

min 1 · P25 27.0 · median 32.0 · mean 35.86 · P75 41.0 · P95 61.95 · max 131

## Distribution by region

| Region | Chunks |
|---|---:|
| BODY | 16924 |
| ABSTRACT | 875 |
| BACK | 131 |

## Distribution by content type

| Content type | Chunks |
|---|---:|
| PROSE | 16690 |
| TABLE | 591 |
| FIGURE_CAPTION | 351 |
| LIST | 126 |
| DISP_QUOTE | 126 |
| FORMULA | 23 |
| BOXED_TEXT | 23 |

## Distribution by evidence type (heuristic metadata)

| Evidence type | Chunks |
|---|---:|
| UNKNOWN | 6410 |
| OBSERVATIONAL_STUDY | 4688 |
| RANDOMIZED_CONTROLLED_TRIAL | 2188 |
| NARRATIVE_REVIEW | 1681 |
| QUALITATIVE_OR_MIXED_METHODS | 950 |
| META_ANALYSIS | 882 |
| SYSTEMATIC_REVIEW | 574 |
| DIAGNOSTIC_OR_VALIDATION_STUDY | 120 |
| GUIDELINE_OR_CONSENSUS | 117 |
| CASE_REPORT_OR_CASE_SERIES | 112 |
| PROTOCOL | 101 |
| EDITORIAL_OR_COMMENTARY | 73 |
| METHODS_OR_TECHNICAL | 34 |

## Quality-flag counts (across all chunks)

| Flag | Chunks |
|---|---:|
| provenance_complete | 17930 |
| paragraph_split | 2001 |
| sentence_split | 2001 |
| is_too_short | 1242 |
| contains_special_content | 1240 |
| contains_table | 591 |
| is_too_long | 470 |
| contains_figure_caption | 351 |
| unsplit_long_sentence | 12 |

Overlap sentences applied (long-paragraph sentence overlaps): **1129**

## Coverage audit summary

- Articles audited: 500
- Dropped source elements: 0
- Reordered chunk sequences: 0
- Articles with multi-chunk elements (long-paragraph overlaps only): 327

## Structural audit summary (per-article counts)

- Articles with parse error: 0
- Articles without abstract: 1
- Empty source elements skipped (total): 5
- Special empty elements skipped (total): 0
- Excluded sections (ack/funding/COI/refs, total): 269

