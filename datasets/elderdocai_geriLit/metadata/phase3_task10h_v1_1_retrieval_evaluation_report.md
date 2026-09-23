# PHASE 3 TASK 10H — GERILIT-GOLD v1.1 QUANTITATIVE RETRIEVAL EVALUATION — REPORT

**Status: PASS** — evaluation-only run of the unchanged frozen GeriLit retrieval
pipeline against the frozen GeriLit-Gold v1.1 benchmark (n=121). Reproducible
across 3 runs; methodology verified equivalent to Task 10D via a v1.0 rerun
(0 mismatches over all 134 v1.0 rankings/metrics). **No retrieval code, index,
benchmark, or production file was modified; nothing committed or pushed.**

## A. Benchmark (evaluated)
- GeriLit-Gold **v1.1** (`data/geri_lit_gold_v1_1.json`) · status FROZEN · version 1.1
- SHA-256: `1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72`
- n = **121** (121 ACCEPTED / 0 rejected / 0 pending)
- Unique PMCIDs: **45** · Unique gold chunks: **116**

## B. Overall retrieval results (v1.1, n=121) — measured
| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 | top-5 hits | no-top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 0.0826 | 0.1653 | 0.2231 | 0.1303 | 0.1523 | 27 | 94 |
| Sparse (BM25) | 0.2893 | 0.3967 | 0.4463 | 0.3518 | 0.3769 | 54 | 67 |
| Hybrid (0.6/0.4) | 0.0165 | 0.0496 | 0.2975 | 0.0821 | 0.1223 | 36 | 85 |
| Hybrid + CrossEncoder | 0.2066 | 0.2727 | 0.2727 | 0.2383 | 0.2493 | 33 | 88 |

Top-1 / top-3 / top-5 hits by configuration: dense 10/20/27 · sparse 35/48/54 ·
hybrid 2/6/36 · rerank 25/33/33. Gold retrieved in top-5 by ≥1 configuration:
**61 / 121**; never in top-5 under any configuration: **60 / 121**.

## C. Topic-level results (v1.1, descriptive)
| Topic | n | Dense R@5 / MRR | Sparse R@5 / MRR | Hybrid R@5 | Rerank R@5 / MRR |
|---|---|--:|---:|---:|---:|
| C01 Medication | 11 | 0.091 / 0.091 | 0.545 / 0.432 | 0.455 | 0.364 / 0.318 |
| C02 Dementia/Cog | 15 | 0.267 / 0.124 | 0.267 / 0.233 | 0.267 | 0.267 / 0.233 |
| C03 Nutrition | 13 | 0.077 / 0.019 | 0.077 / 0.077 | 0.077 | 0.077 / 0.077 |
| C04 Physical Activity | 13 | 0.000 / 0.000 | 0.308 / 0.208 | 0.308 | 0.308 / 0.308 |
| C05 Falls/Mobility | 14 | 0.571 / 0.336 | 0.643 / 0.536 | 0.214 | 0.143 / 0.107 |
| C06 Chronic/Geriatric | 16 | 0.250 / 0.169 | 0.563 / 0.458 | 0.375 | 0.375 / 0.302 |
| C07 Mental/Social | 14 | 0.214 / 0.119 | 0.357 / 0.268 | 0.143 | 0.143 / 0.071 |
| C08 Caregiving | 9 | 0.444 / 0.231 | 0.667 / 0.370 | 0.333 | 0.333 / 0.333 |
| C09 Preventive | 10 | 0.100 / 0.100 | 0.700 / 0.650 | 0.600 | 0.500 / 0.500 |
| C10 Healthy Ageing | 6 | 0.167 / 0.083 | 0.500 / 0.367 | 0.333 | 0.333 / 0.250 |

## D. v1.0 (Task 10D, frozen) vs v1.1 (measured) — descriptive comparison
| Configuration | Benchmark | R@1 | R@3 | R@5 | MRR | nDCG@5 |
|---|---|---|---:|---:|---:|---:|---:|
| Dense | v1.0 | 0.0149 | 0.0149 | 0.0373 | 0.0198 | 0.0229 |
| Dense | v1.1 | 0.0826 | 0.1653 | 0.2231 | 0.1303 | 0.1523 |
| Sparse | v1.0 | 0.0075 | 0.0075 | 0.0224 | 0.0108 | 0.0129 |
| Sparse | v1.1 | 0.2893 | 0.3967 | 0.4463 | 0.3518 | 0.3769 |
| Hybrid | v1.0 | 0.0075 | 0.0149 | 0.0299 | 0.0146 | 0.0179 |
| Hybrid | v1.1 | 0.0165 | 0.0496 | 0.2975 | 0.0821 | 0.1223 |
| Rerank | v1.0 | 0.0149 | 0.0299 | 0.0299 | 0.0199 | 0.0224 |
| Rerank | v1.1 | 0.2066 | 0.2727 | 0.2727 | 0.2383 | 0.2493 |

Measured absolute differences (v1.1 − v1.0) per metric:
| Configuration | ΔR@1 | ΔR@3 | ΔR@5 | ΔMRR | ΔnDCG@5 |
|---|---:|---:|---:|---:|---:|
| Dense | +0.0677 | +0.1504 | +0.1858 | +0.1105 | +0.1295 |
| Sparse | +0.2818 | +0.3892 | +0.4239 | +0.3410 | +0.3639 |
| Hybrid | +0.0091 | +0.0347 | +0.2677 | +0.0675 | +0.1044 |
| Rerank | +0.1917 | +0.2429 | +0.2429 | +0.2184 | +0.2269 |

Methodological equivalence: the same evaluation code re-run on v1.0 reproduced
the frozen Task 10D per-query rankings and metrics **exactly (0 mismatches over
134 queries × 4 configurations × rankings and metrics)** — the measured v1.1
increase is attributable to the benchmark change, not to pipeline drift.

**Interpretation (separated from measurement):** under the same frozen retrieval
system, v1.1 questions are retrieved at substantially higher rates than v1.0
questions. The most probable benchmark-construction interpretation is that the
evidence-first v1.1 questions embed claim-specific vocabulary that more closely
matches their gold evidence (cf. Task 10F lex 0.40→0.70 / sem 0.65→0.75
alignment), particularly benefiting lexical (BM25) retrieval. These measured
differences are descriptive; no configuration is labeled superior and no claim
about retriever quality is made.
## E. Reproducibility
- Runs performed: **3** (run1 & canonical with `--v10-equivalence`, run2 without)
- Deterministic signature (metrics + per-query rankings): **`f3accb4e27f7934de02ae13204e723778294e2bd773c025cbd9bcbe4179b5e9c`** — identical across all runs
- `deterministic = TRUE` (only `generated_at_utc`/`run_name` differ between result files)
- v1.0 equivalence rerun: **0 mismatches** over 134 queries × 4 configs (rankings and metrics)

## F. Validation
- `metadata/task10h_validate.py` → `metadata/validation/phase3_task10h_validation.{json,md}`
- **29 / 29 checks PASS, 0 FAIL** (v1.1 SHA/n=121/ACCEPTED, no dup IDs, gold present; v1.0 + 5 frozen corpus/index artifacts unchanged; Task 10D K-settings, 0.6/0.4 weights, metric definitions, ranking caps; every question evaluated once; gold & retrieved chunk ids traceable to corpus; metrics in [0,1]; signatures match; Task 10D/10E artifact hashes match the Task 10F baseline; Task 10G benchmark + review unchanged; Task 10E coverage; v1.0 equivalence rerun matches)

## G. Integrity — explicit confirmations
- v1.0 unchanged: **YES** (`28ef54fa…`) · v1.1 unchanged: **YES** (`1488d164…`)
- Task 10D unchanged: **YES** · Task 10E unchanged: **YES** · Task 10F unchanged: **YES** · Task 10G unchanged: **YES**
- Corpus unchanged: **YES** · Indexes unchanged (embeddings/FAISS/BM25/row_mapping): **YES**
- Retrieval implementation unchanged: **YES** (evaluation-only script; frozen `geri_lit_retriever.py` used as-is)
- No model downloads: **YES** (offline: HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE; cached BGE + CrossEncoder only)
- No package installation: **YES** · No retrieval optimization: **YES**

## H. Git
- Branch `feature/mimic-pmc-migration` · HEAD `8b0696c` (unchanged)
- Staged: **none** · Committed: **none** · Pushed: **none**
- Modified (pre-existing): `.gitignore`, `scripts/rag_chat.py`
- Newly created (Task 10H): `metadata/task10h_retrieval_evaluation.py` ·
  `metadata/task10h_retrieval_results.json` · `metadata/task10h_summary.json` ·
  `metadata/task10h_validate.py` · `metadata/validation/phase3_task10h_validation.{json,md}` ·
  `metadata/phase3_task10h_v1_1_retrieval_evaluation_report.md` (this file)

## Interpretation guardrail
No conclusion is drawn that "the retriever was broken", "v1.1 is superior",
"v1.0 was invalid", or that any configuration is universally better. The
measured change is reported descriptively: the same frozen retrieval system
retrieves v1.1 gold evidence at substantially higher rates than v1.0 gold
evidence, and the largest increase is in the lexical (BM25) configuration —
consistent with the evidence-first reconstruction — while dense (semantic)
retrieval also increases. These are benchmark-construction effects to be
interpreted by the researcher; no retriever change is justified by this task.

## STOP
Task 10H complete. Await explicit authorization before any retriever
optimization, additional benchmark changes, or ablation study.