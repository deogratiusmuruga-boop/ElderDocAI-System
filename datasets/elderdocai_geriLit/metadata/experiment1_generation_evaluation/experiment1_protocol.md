# Experiment 1 — Final-Stack GeriLit-Gold v1.1 Generation-Quality Evaluation

**Status:** PROTOCOL (frozen before generation)
**Date:** 2026-09-22
**Checkpoint:** `fe20c5ae6351ba2ee3fe3aae8d73d722078b9fbc`
**Benchmark:** GeriLit-Gold v1.1 (`data/geri_lit_gold_v1_1.json`), SHA-256
`1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72`, n = 121.

---

## 1. Research question

How accurately, faithfully, and relevantly does the final ElderDocAI pipeline
generate answers from retrieved GeriLit evidence under the currently enforced
reliability-gating mechanism?

## 2. System under test (all frozen, read-only)

| Component | Value | Source |
|---|---|---|
| Retrieval backend | `geri_lit` (selected via `ELDERDOCAI_RETRIEVAL_BACKEND=geri_lit`) | `scripts/retrieval_router.py` |
| Dense | BGE `BAAI/bge-base-en-v1.5`, FAISS IndexFlatIP, top-5 | `datasets/elderdocai_geriLit/geri_lit_retriever.py` |
| Sparse | BM25Okapi (stored index), top-5 | same |
| Fusion | `0.6 * dense + 0.4 * sparse` | same |
| Rerank | CrossEncoder `cross-encoder/ms-marco-MiniLM-L-6-v2`, top-5 candidates -> final top-3 | same |
| Reliability formula | `0.3*authority + 0.3*relevance + 0.2*support + 0.1*coverage + 0.1*consistency` | `scripts/reliability_evaluation.py` + `config/reliability_config.json` |
| Thresholds | ACCEPT >= 0.80, REFINE >= 0.65, RE-RETRIEVE >= 0.45, REJECT < 0.45 | `config/reliability_config.json` |
| Gate | Task 1 programmatic gate (ACCEPT/REFINE/RE-RETRIEVE/REJECT enforced; MAX_REFINE=1, MAX_RETRIEVE=1) | `scripts/rag_chat.py::_run_gated_generation` |
| Relevance semantics | `similarity_score` = dense cosine; `retrieval_score` = CrossEncoder logit (Task 2) | `scripts/rag_chat.py::prepare_evidence`, `scripts/geri_lit_adapter.py` |
| Generation | Ollama `llama3.2:latest`, temperature 0, top_p 0.1, top_k 10 | `scripts/rag_chat.py::GENERATION_SYSTEM_PROMPT`, `_run_gated_generation` |
| Adaptive context / profile | `user_profile=None` (no patient context supplied; questions are general clinical), matching the legacy `generate_answer(question)` call contract | `scripts/rag_chat.py::generate_answer` |

No prompt, model, weight, threshold, chunking, index, or benchmark change is
introduced by this experiment.
### 4.4 Unsupported-claim / hallucination assessment
- Operationalized from the faithfulness rubric: an answer is flagged as
  containing unsupported claims when `faithfulness <= 0.50`. When
  `faithfulness == 0.0` the answer is unrelated to or contradicts the
  evidence. Reported per-question as `hallucination` (bool) and aggregated as
  `unsupported_claim_rate` / `contradiction_rate` (faithfulness == 0.0).
- No claim is labelled as hallucinated purely on lexical non-identity.

### 4.5 Refusal behavior
- `refused` flag comes from the gate. If `refused=True`, answer is the
  production REJECTION/EMPTY-EVIDENCE response; `generated=False`.
- Correct/incorrect refusal is NOT re-judged here because the fixed protocol
  requires a reproducible ground-truth refusal label that the frozen benchmark
  does not provide. Refusal counts and only the recorded `why_refused` reason
  are reported. If REJECT/empty-evidence counts are zero, they are reported as
  zero.

## 5. Required per-question record fields

`question_id, topic, question, gold_pmcid, gold_chunk_ids, retrieved_evidence_ids,
retrieved_evidence_texts, retrieval_backend, evidence_count (final),
initial_reliability (factor dict), initial_decision, reliability (final factor
dict), decision, retrieval_attempts, refinement_attempts, refused, generated,
answer, latency_seconds, faithfulness, faithfulness_reason, answer_relevance,
answer_relevance_reason, evidence_support, hallucination, gold_chunk_retrieved,
error`.

## 6. Aggregation

Report per metric `{n, mean, median, std, min, max}`; decision distribution
counts+percentages; gate-behavior counts (empty evidence, refinement cases,
re-retrieval cases, rejection cases, generation-permitted, generation-blocked);
topic-level `{n, mean reliability, mean relevance, decision distribution}`;
evidence statistics (mean/median final evidence count). No topic ranking, no
"better/worse" language.

## 7. Reproducibility

Run the full evaluation twice (run-id 1 then run-id 2) with identical inputs.
Compare substantive fields: evidence IDs, reliability factors, decision,
refinement/retrieval counts, refused, generated, answer, faithfulness,
answer_relevance, evidence_support, hallucination. Latency and timestamps are
non-substantive. Report per-field equality and an overall signature.
## 8. Integrity

Record SHA-256 of the frozen artifacts before and after each run: v1.0, v1.1,
chunks, embeddings, FAISS, BM25, row mapping, reliability config, Task 10D–10H
and Task 3 artifacts. Any unexpected change => STOP.

## 9. Outputs (all under this directory)

`experiment1_generation_evaluation.py`, `experiment1_per_question_run1.json`,
`experiment1_per_question_run2.json`, `experiment1_summary_run{1,2}.json`,
`experiment1_reproducibility.json`, `experiment1_frozen_hashes.json`,
`experiment1_validate.py`, `experiment1_validation.json`,
`experiment1_validation.md`, `experiment1_run_manifest.json`,
`experiment1_report.md`, `experiment1_protocol.md`.

## 10. Scope limits

No retrieval optimization; no threshold/weight/model/prompt changes; no
fine-tuning; no downloads; no benchmark modification; no production-code
modification; no comparison of legacy-stack results as if they were the same
experiment.