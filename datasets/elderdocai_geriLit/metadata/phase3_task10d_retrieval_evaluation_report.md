# PHASE 3 TASK 10D — QUANTITATIVE GERILIT-GOLD RETRIEVAL EVALUATION — REPORT

## A. Objective
Measure how effectively the frozen GeriLit retrieval pipeline retrieves the
human-verified relevant GeriLit evidence for the 134 GeriLit-Gold questions,
across four configurations: dense, sparse (BM25), hybrid (0.6 dense + 0.4
sparse), and hybrid + CrossEncoder reranking. Evaluation only — no retrieval
code, benchmark, or artifact was modified.

## B. Dataset
- GeriLit-Gold questions: **134** (GLG-001..GLG-134)
- Unique PMCIDs: **46**
- Unique gold relevant chunks: **131**
- Topic distribution: C01 14 · C02 17 · C03 13 · C04 13 · C05 15 · C06 20 ·
  C07 16 · C08 9 · C09 11 · C10 6
- Source: `data/geri_lit_gold.json` (human-verified, frozen)

## C. Retrieval configurations (frozen, unchanged)
- **A. Dense** — BGE `BAAI/bge-base-en-v1.5` query embedding (L2-normalized)
  → FAISS IndexFlatIP top-5 (`GeriLitRetriever.dense_retrieve`, DENSE_K=5).
- **B. Sparse** — `rank_bm25.BM25Okapi` with `lower().split()` top-5
  (`sparse_retrieve`, SPARSE_K=5).
- **C. Hybrid** — existing `0.6*dense_norm + 0.4*sparse_norm` fusion
  (`hybrid_retrieve`, 5 candidates).
- **D. Hybrid + CrossEncoder** — C followed by
  `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking, final top-3
  (`rerank`, FINAL_K=3).
All use the frozen indexes (FAISS/BM25/chunks/row_mapping) and cached models.

## D. Metric definitions (per question, binary relevance)
- **Recall@K** = 1 if >=1 gold-relevant chunk appears in the top K, else 0.
  K in {1,3,5}.
- **MRR** = reciprocal rank of the FIRST gold-relevant chunk in the ranking
  (0 if none retrieved).
- **nDCG@5** = DCG@5 / IDCG@5 with binary gains (relevant=1, else 0)
  strictly from `gold_relevant_chunk_ids`.
- **Hit@K** reported = Recall@K (equivalent for single/multiple gold).

## E. Overall results (mean over 134 questions)
| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| Dense | 0.0149 | 0.0149 | 0.0373 | 0.0198 | 0.0229 |
| BM25 (sparse) | 0.0075 | 0.0075 | 0.0224 | 0.0108 | 0.0129 |
| Hybrid | 0.0075 | 0.0149 | 0.0299 | 0.0146 | 0.0179 |
| Hybrid + CrossEncoder | 0.0149 | 0.0299 | 0.0299 | 0.0199 | 0.0224 |

Questions with >=1 gold chunk in top-5: Dense 5, Sparse 3, Hybrid 4,
Rerank 4; **8/134 questions** retrieve the gold chunk in top-5 under any
configuration. These are the measured values — reported honestly, without
manipulation.
## F. Topic-level results (Recall@5 / MRR / nDCG@5 by topic)
| Topic | n | Dense | Sparse | Hybrid | Rerank |
|---|---|---|---|---|---|
| C01 Medication | 14 | R5 0.0 | R5 0.0 | R5 0.0 | R5 0.0 |
| C02 Dementia/Cog | 17 | R5 .059 | R5 .000 | R5 .059 | R5 .059 |
| C03 Nutrition | 13 | R5 .000 | R5 .077 | R5 .077 | R5 .077 |
| C04 Physical Activity | 13 | R5 0.0 | R5 0.0 | R5 0.0 | R5 0.0 |
| C05 Falls/Mobility | 15 | R5 .133 | R5 .000 | R5 .000 | R5 .000 |
| C06 Chronic/Geriatric | 20 | R5 .100 | R5 .000 | R5 .000 | R5 .000 |
| C07 Mental/Social | 16 | R5 0.0 | R5 0.0 | R5 0.0 | R5 0.0 |
| C08 Caregiving | 9 | R5 0.0 | R5 0.0 | R5 0.0 | R5 0.0 |
| C09 Preventive | 11 | R5 .000 | R5 .091 | R5 .091 | R5 .091 |
| C10 Healthy Ageing | 6 | R5 .000 | R5 .167 | R5 .167 | R5 .167 |
All 10 topics evaluated (MRR/nDCG values in the JSON summary). Topics with
zero hits (C01, C04, C07, C08) are reported exactly as measured.

## G. Reproducibility
The full evaluation was run three times; results and summary were identical
across runs (`DETERMINISTIC=True`; `SUMMARY_DETERMINISTIC=True`). Per-query
chunk rankings, scores, and metrics are unchanged between runs. Validator
20/20 PASS including determinism check.

## H. Limitations
- Benchmark is corpus-native and single-reviewer (human-reviewed by one
  researcher; no inter-rater reliability).
- Relevance is binary (gold membership) — no graded relevance.
- Gold evidence is typically ONE chunk per question (131 unique chunks);
  zero-shot/large searches (e.g., top-200) still frequently miss it,
  indicating the generated natural-language questions do not strongly
  re-surface their anchor chunk under this corpus/model.
- N=134 questions over 46 articles — small, no clinical effectiveness claim,
  no claim of configuration superiority beyond what the numbers show.

## I. Research interpretation
The measured results show low retrieval of the gold chunks under all four
frozen configurations (mean Recall@5 ~ 0.02-0.04; overall best top-5 hits 8/134
across any config, 5/134 for dense alone). Dense slightly edges other
configs on MRR/nDCG@5; rerank marginally improves Recall@3 versus hybrid but
does not outperform dense on Recall@5/MRR in the aggregate. **No configuration
is claimed to be superior**; the honest statement is that the current frozen
retrieval pipeline retrieves the human-verified gold evidence for only a small
subset of GeriLit-Gold questions at these depths. These results should be
interpreted as an evaluation of the frozen GeriLit pipeline + corpus-native
benchmark, not as a clinical or production-quality claim.

## Validation
`metadata/validation/phase3_task10d_validation.{json,md}` — **20/20 PASS**:
gold loads; 134 questions; no duplicate IDs; every question has gold; all four
configs executed; 134x4 results; no NaN/Inf; per-query results preserved;
10 topics preserved and covered; deterministic run1 vs run2; frozen corpus/
embeddings/FAISS/BM25/row-mapping hashes unchanged; `data/geri_lit_gold.json`
unchanged; router/adapter untracked-new (no tracked modification); rag_chat
still has only the known 10B 4-line change; result file reproducible.

## Files created
- `metadata/evaluate_task10d.py`
- `metadata/task10d_retrieval_results.json` (per-query, all configs/ranks/metrics)
- `metadata/task10d_summary.json` (overall + per-topic)
- `metadata/task10d_validate.py`
- `metadata/validation/phase3_task10d_validation.{json,md}`
- `metadata/phase3_task10d_retrieval_evaluation_report.md` (this file)

## Files modified
None (evaluation-only; `scripts/rag_chat.py` retains the prior Task 10B
4-line change; no new production modifications).

## Git status
Branch `feature/mimic-pmc-migration`, HEAD `8b0696c`. Modified:
`.gitignore` (pre-existing), `scripts/rag_chat.py` (10B). New untracked files:
Task 10D artifacts above + pre-existing untracked dirs/files. **Nothing staged,
committed, or pushed.**