# ElderDocAI-GeriLit — Phase 3 Task 9A
## GeriLit Corpus Topic Distribution Audit (read-only)

> **Scope**: read-only corpus characterization to prepare for a future, human-verified
> GeriLit-Gold retrieval benchmark. No benchmark questions created. No corpus/index/
> retrieval/evaluation modification.

## 1. Corpus overview
| Metric | Value |
|---|---|
| Source articles (PMCIDs) | **500** (unique 500) |
| PMIDs present | **499** (1 missing) |
| Chunks | **17,930** |
| Chunks/article | min 1 · max 131 · mean 35.86 · median 32 |
| Articles with title | 500/500 |
| Articles with abstract (structural audit) | 499/500 |
| Articles with body | 499/500 |
| Section metadata | available per chunk (`section_id`, `section_title`, `region`, `content_type`) |
| Source locator | on every chunk (Task 4) |
| Evidence/content type | on every chunk (`content_type`; `evidence_type` per article) |
| Topic metadata | `topic_ids` (Task-2 taxonomy T01–T09) on articles and chunks |
| Candidate categories (Task 9A) | 10 (C01–C10) |

## 2. Classification method (deterministic, auditable, NO LLM)
- Source text: article **title** (weight 3) + **abstract** (weight 2) + **section titles**
  (weight 1), lowercased.
- Keyword rule sets (documented in `task9a_topic_audit.py`) per candidate category.
- Multi-label allowed: a category is assigned when score ≥ `ASSIGN_THRESHOLD` (2.0).
  If none qualify, the best-scoring category with score ≥ 1.0 is assigned once (flagged
  `fallback`); otherwise the article is `NOT_CLASSIFIED`.
- Chunk membership is attributed from the article's assigned categories.

## 3. Topic distribution (multi-label)
| Category | Articles | % of 500 | Chunks | % of 17,930 |
|---|---:|---:|---:|---:|
| C01 Medication & Polypharmacy | 117 | 23.40 | 4,219 | 23.53 |
| C02 Dementia & Cognitive Health | 148 | 29.60 | 5,404 | 30.14 |
| C03 Nutrition | 101 | 20.20 | 3,619 | 20.18 |
| C04 Physical Activity & Exercise | 104 | 20.80 | 3,933 | 21.94 |
| C05 Falls & Mobility | 143 | 28.60 | 5,198 | 28.99 |
| C06 Chronic Disease & Geriatric Conditions | 236 | 47.20 | 8,335 | 46.49 |
| C07 Mental & Social Wellbeing | 167 | 33.40 | 6,435 | 35.89 |
| C08 Caregiving | 73 | 14.60 | 2,725 | 15.20 |
| C09 Preventive Care | 164 | 32.80 | 6,042 | 33.70 |
| C10 General Healthy Ageing | 25 | 5.00 | 972 | 5.42 |

Note: percentages are per-category share of the 500-article corpus (multi-label; sums >100%).
## 4. Multi-label structure
| Labels per article | Articles |
|---|---:|
| 0 (NOT_CLASSIFIED) | 7 |
| 1 | 118 |
| 2 | 142 |
| 3 | 124 |
| 4 | 64 |
| 5 | 29 |
| 6 | 10 |
| 7 | 5 |
| 8 | 1 |
Total single-label: 118 · multi-label: 375 · not-classified: 7.

## 5. Benchmark-suitable candidates (per topic)
| Category | Candidate articles | Candidate chunks |
|---|---:|---:|
| C01 | 117 | 3,686 |
| C02 | 148 | 4,740 |
| C03 | 101 | 3,224 |
| C04 | 104 | 3,451 |
| C05 | 143 | 4,593 |
| C06 | 236 | 7,336 |
| C07 | 167 | 5,640 |
| C08 | 73 | 2,417 |
| C09 | 164 | 5,341 |
| C10 | 25 | 795 |

Candidates = articles assigned to the category whose ABSTRACT/BODY chunks (PROSE/TABLE/
FIGURE_CAPTION/LIST/DISP_QUOTE, ≥60 words) can support realistic elderly-care questions.
Detailed records (pmcid, pmid, title, secondary topics, candidate chunk ids, useful
sections, reason) are in `task9a_corpus_topic_distribution.json` → `benchmark_candidates_detail`.

## 6. Proposed future gold allocation (planning only, target 96)
| Category | Proposed queries |
|---|---:|
| C01 | 22 |
| C02 | 28 |
| C03 | 19 |
| C04 | 20 |
| C05 | 27 |
| C06 | 45 |
| C07 | 32 |
| C08 | 14 |
| C09 | 31 |
| C10 | 5 |

The raw proportional allocation sums to 223 (≥96, because multi-label shares overlap). A
final 96-item design should apply explicit floors/caps (e.g., floor 5–8 per topic, cap ~20
for C06, cap ~8 for C10) during the authorized benchmark-construction task. **This is a
planning recommendation, not a benchmark.**

## 7. Coverage gaps / risks
- **Sparse**: C10 General Healthy Ageing (25 articles, 5.0%) — least supported; overlaps
  heavily with C06/C07.
- **Dominant**: C06 (47.2% articles, 46.5% chunks) — risk of over-weighting chronic
  geriatric disease content in an unstratified Gold set.
- **Caregiving** (C08) is supported but the smallest functional topic (73 articles) — needs
  a deliberate cap.
- **Not classified** (7 articles): likely methodological/molecular or narrowly scoped —
  candidates for exclusion from Gold construction.
- 1 article missing PMID, 1 missing abstract (Task-3 structural audit).
- Short/table-heavy chunks and untitled sections exist (Task 4); need manual review before
  Gold anchoring.
- Multi-label attribution inflates per-category counts; use single-label primary topics
  for stratification and multi-label for secondary topic tagging only.

## 8. Integrity
Frozen hashes unchanged (chunks `62bfdde3…`, embeddings `b0905d2f…`, FAISS `b7016b9d…`,
BM25 `17f50324…`, row_mapping `4d649f81…`). Task-2 manifests, Task-3 metadata, Task-4
chunks, Task-5/6 indexes, Gold16/Gold96, retrieval scripts, and evaluation scripts were not
modified. Validation 10/10 in `metadata/validation/phase3_task9a_validation.{json,md}`.
Audit is deterministic (content signature identical across repeated runs).

## 9. Recommendation
The corpus is **balanced enough for a useful internal gold set** if the final query
allocation is stratified by the candidate counts above with explicit floors/caps (e.g.,
floors 5–8 per topic, cap ~20 for C06, cap ~8 for C10), anchored to human-verified
supporting chunks in a separate authorized task. C10, NOT_CLASSIFIED articles, and
table-only/untitled chunks are the main risks to control.
