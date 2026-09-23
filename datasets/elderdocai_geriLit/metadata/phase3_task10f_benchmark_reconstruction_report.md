# PHASE 3 TASK 10F — GERILIT-GOLD v1.1 CANDIDATE BENCHMARK RECONSTRUCTION — REPORT

**Status: PASS** — candidate benchmark `data/geri_lit_gold_v1_1_candidate.json`
constructed evidence-first and validated 32/32. All frozen artifacts unchanged.
No retrieval evaluation was run; no human review was performed or claimed.

## 1. Objective
Construct a *separate* candidate GeriLit-Gold v1.1 benchmark using an
evidence-first methodology so that later work can determine whether the low
Task 10D retrieval scores were substantially caused by v1.0 question/evidence
alignment weaknesses. This task performs no retrieval optimization and no
human relevance judgment.

## 2. Method (deterministic, evidence-first)
Per v1.0 record:
1. Load the v1.0 gold chunk and the full article (all chunks of the PMCID).
2. Compute deterministic alignment diagnostics (lexical overlap + BGE semantic
   similarity from cached embeddings, same-article competition).
3. Extract the *best claim sentence* from the gold chunk (prefers quantitative,
   effect-bearing, focus-bearing sentences; penalises methods boilerplate,
   table rows, patient quotes, and citation fragments).
4. Decide:
   - **RETAINED** — v1.0 question kept verbatim when it is specific, its focus
     is answered by the gold claim, and alignment is strong.
   - **RECONSTRUCTED** — new question generated from the claim via shape-based,
     evidence-anchored frames (causal / association / recommendation /
     prevalence / descriptive) that embed the claim-specific focus and a
     distinctive outcome phrase; gated for answerability from the gold chunk.
     Chunks are re-anchored to a better same-article chunk only when the v1.0
     gold chunk carries no defensible claim.
   - **EXCLUDE** — recorded with a reason when no defensible candidate can be
     constructed (weak gold, or duplicate/near-duplicate evidence).
5. Every included candidate retains full provenance and diagnostics and is
   `PENDING_HUMAN_REVIEW`.

No LLM is used for question construction or judgment.

## 3. Candidate set (GeriLit-Gold v1.1 — CANDIDATE)
- **Candidates: 121** (reduced from 134 v1.0 records; 13 excluded with reasons)
- Retained verbatim: **18** · Reconstructed from evidence: **103** (of which
  re-anchored to a better same-article chunk: **23**)
- Excluded: **13** (1 weak/no-defensible-claim, 2 unresolved exact-duplicates,
  10 near-duplicates on the same gold chunk)
- All records: `review_status = "PENDING_HUMAN_REVIEW"`

## 4. Frozen integrity & hashes (verified before/after)
| Artifact | SHA-256 | Status |
|---|---|---|
| data/geri_lit_gold.json (v1.0) | `28ef54fa…` | unchanged |
| chunks/chunks.jsonl | `62bfdde3…` | unchanged |
| index/embeddings.npy | `b0905d2f…` | unchanged |
| index/faiss_index.bin | `b7016b9d…` | unchanged |
| index/bm25.pkl | `17f50324…` | unchanged |
| index/row_mapping.json | `4d649f81…` | unchanged |
| task10d_retrieval_results.json | baseline `503b199e…` | unchanged |
| task10d_summary.json | baseline `e81230ec…` | unchanged |
| task10e_failure_analysis.json | baseline `3a262687…` | unchanged |
| phase3_task10e report.md | baseline `a0628c5c…` | unchanged |

Full baseline recorded in `metadata/task10f_frozen_hashes.json`.

## 7. Question statistics
- Raw token count: mean 8.8 · median 10 · min 3 · max 12
  (v1.0 questions were ~3.5–4.7 content tokens; candidates are longer and
  reference concrete claim content)
- Content-token count: mean 2.2 · median 2
- Exact duplicate questions: **0** (post-hoc near-duplicate dedup: 10 dropped)
- Near-duplicate pairs within the set: **0**

## 8. Alignment diagnostics (deterministic + cached BGE embeddings only)
| Metric | mean | median | min | max |
|---|---:|---:|---:|---:|
| Lexical overlap (v1.0 all: ~0.40) | 0.703 | 0.667 | 0.40 | 1.00 |
| Semantic similarity (v1.0 all: ~0.65) | 0.745 | 0.749 | 0.599 | 0.852 |

Same-article competition: mean 14.0 competing chunks per candidate
(consistent with Task 10E methodology).

## 9. Template-family diagnostics (construction transparency)
study_reported_about 53 · association_between 19 · retained_v1_0 18 ·
study_reports_about 13 · how_does_relate_to 12 · recommended_regarding 6.
Questions within a family remain distinct because frames embed the
claim-specific focus + distinctive outcome phrase (answerability is gated to
the gold chunk).

## 10. Exclusions (13) — with reasons
- 1 weak gold with no defensible same-article alternative (GLG-122)
- 2 unresolved exact duplicates after re-anchor onto shared evidence
  (GLG-087, GLG-090)
- 10 near-duplicates (same gold chunk, question Jaccard ≥ 0.7); first
  occurrence kept (GLG-004, 006, 011, 016, 031, 070, 086, 089, 102, 108)

## 11. Provenance
Every candidate preserves: candidate_id (GLG11-C001…), original v1.0 id
(GLG-001…), original candidate id (GLG-C001…), original question, question,
PMCID, PMID, title, topic + secondary topics, gold chunk ids, source document,
section id/title, source locator, evidence type, reconstruction status/reason,
review status, and per-candidate diagnostics (claim sentence/score/shape,
focus, population, alignment, region, competition).

## 12. Validation
`metadata/task10f_validate.py` → `validation/phase3_task10f_validation.{json,md}`
— **32/32 PASS** (frozen v1.0 + 5 indexes + Task 10D/10E artifacts unchanged,
candidate structure, unique IDs, PMCID/gold chunk/topic validity,
PENDING_HUMAN_REVIEW, provenance, evidence rationale, no duplicates,
v1.0 traceability, exclusion accounting, determinism signature).

## 13. Reproducibility
Two independent runs produced the identical candidate set (signature
`3866bf31…`); frozen hashes identical before/after (`frozen_unchanged=True`).
Only `generated_at_utc` differs between runs.

## 14. Integrity
Human decisions encoded without automated reinterpretation: N/A (no human
review exists yet — candidates are PENDING_HUMAN_REVIEW).
v1.0 benchmark unchanged: **YES** · Corpus/embeddings/FAISS/BM25/row_mapping
unchanged: **YES** · Task 10D/10E artifacts unchanged: **YES** ·
production/API/frontend code untouched: **YES**.
## 5. Topic coverage (C01–C10)
C01 11 · C02 15 · C03 13 · C04 13 · C05 14 · C06 16 · C07 14 · C08 9 · C09 10 ·
C10 6 — all 10 topics present; coverage reflects available defensible evidence
(no forced balancing).

## 6. Evidence statistics
- Unique PMCIDs: **45**
- Unique gold chunks: **116**
- Single-gold candidates: 121 (100%) · multi-gold: 0
- ABSTRACT: 32 (26.4%) · BODY: 88 (72.7%) · TABLE: 1 (0.8%)
## 15. Files created
- `data/geri_lit_gold_v1_1_candidate.json` (candidate benchmark, n=121)
- `datasets/elderdocai_geriLit/metadata/task10f_benchmark_reconstruction.py`
- `datasets/elderdocai_geriLit/metadata/task10f_benchmark_reconstruction.json`
- `datasets/elderdocai_geriLit/metadata/task10f_frozen_hashes.json`
- `datasets/elderdocai_geriLit/metadata/task10f_validate.py`
- `datasets/elderdocai_geriLit/metadata/validation/phase3_task10f_validation.{json,md}`
- `datasets/elderdocai_geriLit/metadata/phase3_task10f_benchmark_reconstruction_report.md` (this file)

## 16. Files modified
None. No frozen benchmark, corpus, index, retriever, router, adapter, API,
frontend, or production file was modified.

## 17. Git status
Branch `feature/mimic-pmc-migration` · HEAD `8b0696c` (unchanged).
Modified (pre-existing): `.gitignore`, `scripts/rag_chat.py`.
Untracked new: Task 10F artifacts above + earlier Phase-3 artifacts.
Untracked pre-existing: GeriLit dir, mimic_demo, candidate/review/gold JSON,
router/adapter/tests, stray log/temp files. **Nothing staged, committed, or
pushed.**

## 18. STOP
Task 10F is complete. **Not done** (per constraints): no retrieval evaluation
(no Recall@K/MRR/nDCG against v1.1), no retrieval optimization, no human
relevance judgment, no v1.0 modification, no LLM reviewer, no model downloads,
no commit/push.

Candidates must remain `PENDING_HUMAN_REVIEW` until the researcher personally
reviews them; v1.1 retrieval evaluation happens only after human review and
freezing.