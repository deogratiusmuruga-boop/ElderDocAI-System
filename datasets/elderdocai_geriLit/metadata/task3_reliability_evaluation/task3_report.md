# Task 3 — Scientific Reliability Re-evaluation (post-Task-2)

## 1. Evaluation Status
- **PASS**
- Benchmark questions: **121**
- Benchmark: **GeriLit-Gold v1.1** (`data/geri_lit_gold_v1_1.json`, FROZEN)
- Benchmark SHA-256: `1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72` (verified)

## 2. System Configuration
- Retrieval backend: **geri_lit** (dense BGE + BM25 + hybrid 0.6/0.4 + CrossEncoder rerank, top-3), frozen stack from `datasets/elderdocai_geriLit/`
- Embedding model: `BAAI/bge-base-en-v1.5`
- CrossEncoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- LLM: **not invoked** (gate simulated programmatically; generation-permitted flags recorded)
- Reliability formula: `0.3*authority + 0.3*relevance + 0.2*support + 0.1*coverage + 0.1*consistency`
- Thresholds: ACCEPT ≥ 0.80 · REFINE ≥ 0.65 · RE-RETRIEVE ≥ 0.45 · REJECT < 0.45
- **Confirmation: thresholds were NOT tuned; weights were NOT altered.**

## 3. Baseline Discovery
- Pre-Task-2 baseline found: **NO** (for the GeriLit-Gold v1.1 population)
- Legacy artifact found (NOT a valid v1.1 baseline): `data/evaluation_results/reliability_gating_results.json`
  - SHA-256: `e73f96d764206c7e…` (full hash recorded in `task3_frozen_hashes.json` + legacy artifact listing)
  - Evaluation population: the legacy **16 gold-QA** questions on the legacy PDF knowledge base (RQ3), **not** GeriLit-Gold v1.1; produced with the pre-Task-1/pre-Task-2 stack. It is a legacy regression artifact and was **preserved unchanged**.
- Why no valid v1.1 baseline: no reliability-factor evaluation existed for the GeriLit-Gold v1.1 population before Task 2 was implemented (Tasks 10D–10H were retrieval metrics only). Per instructions, no baseline was manufactured and the code was never reverted.

**Before/after comparison: unavailable** because no valid pre-Task-2 scientific baseline artifact for the GeriLit-Gold v1.1 population was found.

## 4. Reliability Results (post-Task-2, n=121)
| Stat | Overall reliability |
|---|---:|
| mean | 0.7884 |
| median | 0.7783 |
| std | 0.0329 |
| min | 0.7050 |
| max | 0.9020 |

## 5. Factor Results
| Factor | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|
| authority | 0.8500 | 0.8500 | 0.0000 | 0.85 | 0.85 |
| relevance | 0.5534 | 0.5000 | 0.0966 | 0.50 | 0.8319 |
| support | 1.0000 | 1.0000 | 0.0000 | 1.00 | 1.00 |
| coverage | 0.6875 | 0.6667 | 0.1416 | 0.40 | 1.00 |
| consistency | 0.9862 | 1.0000 | 0.0793 | 0.3333 | 1.00 |

## 6. Decision Distribution
| Decision | count | % |
|---|---:|---:|
## 7. Evidence / Gate Behaviour
- empty evidence: **0**
- refinement cases: **84** (single bounded refinement each)
- re-retrieval cases: **0**
- rejection cases: **0**
- generation-permitted (gate → LLM allowed): **121**
- generation-blocked (gate → no LLM): **0**
- evidence count: mean 2.99 · median 3
- Error/fallback conditions: none occurred.

## 8. Topic-Level Results (descriptive, not ranked)
| Topic | n | mean reliability | mean relevance | ACCEPT / REFINE |
|---|---:|---:|---:|---:|
| C01 | 11 | 0.7879 | 0.5443 | 4 / 7 |
| C02 | 15 | 0.7828 | 0.5320 | 3 / 12 |
| C03 | 13 | 0.7929 | 0.5371 | 4 / 9 |
| C04 | 13 | 0.7786 | 0.5253 | 2 / 11 |
| C05 | 14 | 0.8023 | 0.6037 | 7 / 7 |
| C06 | 16 | 0.7824 | 0.5402 | 4 / 12 |
| C07 | 14 | 0.7756 | 0.5349 | 3 / 11 |
| C08 | 9 | 0.8017 | 0.5897 | 4 / 5 |
| C09 | 10 | 0.7925 | 0.5644 | 3 / 7 |
| C10 | 6 | 0.8016 | 0.6085 | 3 / 3 |

Per-question detail (evidence ids, dense cosines, retrieval scores, factors, decisions, gate flags) is in `task3_per_question.json`.

## 9. Before vs After Task 2
**No valid pre-Task-2 scientific baseline artifact was found for the GeriLit-Gold v1.1 population**, so no controlled before/after comparison is possible for this population. Observed post-Task-2 measurements are reported descriptively above. (A legacy RQ3 reliability artifact exists on a different dataset/population and was left untouched.)

## 10. Determinism
- run 1 signature: `effebbdd146d7b57d03f58765b36921110ef5ceeedc2cccd56afc18ec44c1781`
- run 2 signature: `effebbdd146d7b57d03f58765b36921110ef5ceeedc2cccd56afc18ec44c1781`
- identical: **YES** (also identical per-question evidence ids, factors, decisions, and gate flags)

## 11. Frozen Artifact Integrity (verified in `task3_frozen_hashes.json` + validator)
- v1.0 `28ef54fa…` ✓ · v1.1 `1488d164…` ✓
- chunks `62bfdde3…` ✓ · embeddings `b0905d2f…` ✓ · FAISS `b7016b9d…` ✓ · BM25 `17f50324…` ✓ · row_mapping `4d649f81…` ✓
- Task 10D results/summary & Task 10E analysis: unchanged vs Task 10F baseline hashes ✓
- Task 10F/10G/10H diagnostics: recorded, run-invariant ✓
- `config/reliability_config.json`: unchanged ✓ (weights/thresholds content validated)

## 12. Files Created/Modified
Created under `datasets/elderdocai_geriLit/metadata/task3_reliability_evaluation/`:
- `task3_reliability_evaluation.py` (evaluation runner; NEW)
- `task3_per_question.json` (per-question results; NEW)
- `task3_summary.json` (aggregates/topics/determinism; NEW)
- `task3_frozen_hashes.json` (integrity manifest; NEW)
- `task3_run_manifest.json` (run signatures; NEW)
- `task3_validate.py` (validator; NEW)
- `phase3_task3_reliability_validation.{json,md}` (validation outputs; NEW)
- `task3_report.md` (this report; NEW)
**No existing source, benchmark, corpus, index, or frozen research file was modified.**

## 13. Scientific Interpretation (descriptive only)
- With the corrected semantics (dense cosine → relevance), the measured relevance factor mean is 0.553 (median 0.50) and overall reliability mean is 0.788 over 121 v1.1 questions.
- No question produced empty evidence; none fell below the RE-RETRIEVE/REJECT boundaries; 30.6% finish ACCEPT and 69.4% finish REFINE (each with exactly one bounded refinement), and the gate permitted generation for all 121 questions (no LLM was actually invoked in this evaluation).
- These are descriptive measurements of the current fixed system. No claim of "improvement", no threshold suggestion, no ranking of topics/configurations, and no causal claim beyond "reproducible post-Task-2 measurements on GeriLit-Gold v1.1" is made.

## 14. Git Status
No files staged, committed, or pushed (details in the final task report below).
| ACCEPT | 37 | 30.58% |
| REFINE | 84 | 69.42% |
| RE-RETRIEVE | 0 | 0.00% |
| REJECT | 0 | 0.00% |