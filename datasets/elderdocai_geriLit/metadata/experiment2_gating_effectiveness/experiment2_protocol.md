# Experiment 2 - Reliability-Gating Effectiveness

**Status:** PROTOCOL (frozen before generation)
**Date:** 2026-09-22
**Checkpoint:** `fe20c5ae6351ba2ee3fe3aae8d73d722078b9fbc`
**Benchmark:** GeriLit-Gold v1.1 (`data/geri_lit_gold_v1_1.json`), SHA-256
`1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72`, n = 121.

## 1. Primary research question

Does reliability-gated generation improve answer faithfulness, answer
relevance, and evidence support compared with otherwise identical ungated
generation?

This is a controlled within-subject ablation on a single frozen pipeline.

## 2. System under test (frozen, read-only; identical to Experiment 1)

- Retrieval backend `geri_lit`: BGE `BAAI/bge-base-en-v1.5` top-5 + BM25 top-5
  -> 0.6/0.4 hybrid -> CrossEncoder `ms-marco-MiniLM-L-6-v2` rerank -> top-3.
- Reliability: `0.3*A + 0.3*R + 0.2*S + 0.1*Cov + 0.1*Con`; thresholds
  ACCEPT >= 0.80, REFINE >= 0.65, RE-RETRIEVE >= 0.45, REJECT < 0.45.
- Gate budgets: MAX_REFINE = 1, MAX_RETRIEVE = 1 (enforced, Task 1).
- Relevance semantics: `similarity_score` = dense cosine (Task 2).
- LLM: Ollama `llama3.2:latest`; temperature 0, top_p 0.1, top_k 10.
## 3. Experimental design (paired per-question)

For each of the 121 benchmark questions one paired record is produced:

1. Retrieve ONCE: `chunks = retrieve_evidence(question)` (deterministic),
   `initial_evidence = prepare_evidence(chunks)`,
   `initial_rel = evaluate_reliability(question, initial_evidence)`,
   `initial_dec = make_reliability_decision(initial_rel)`.

2. **Condition A - Gate ON (production path):** call
   `generate_answer(question, return_evaluation=True)`. The production gated
   flow retrieves (deterministically identical to step 1), evaluates
   reliability, applies the gate (REFINE/RE-RETRIEVE/REJECT with the frozen
   budgets), and generates from the FINAL evidence. Record initial+final
   reliability, decisions, refinement/retrieval attempts, refused, answer,
   latency.

3. **Condition B - Gate OFF (bypass):** use the SAME initial evidence from
   step 1; build the grounded prompt with
   `build_grounded_prompt(question, initial_evidence, initial_rel,
   initial_dec, ...)` and the SAME system prompt / generation options. NO
   refinement, NO re-retrieval, NO reliability decision enforcement. If
   `initial_evidence` is empty, return the same empty-evidence response as
   production (this is the no-evidence path, not a gate decision). Apply the
   SAME answer post-processing as production.

4. **Prompt control:** the prompt is produced by the same
   `build_grounded_prompt` template and the same system prompt in both
   conditions. The only differences are intrinsic to the gate's operation:
## 4. Primary statistical methodology (pre-registered)

Paired differences per question: `delta = ON - OFF` for each primary metric
(faithfulness, answer relevance, evidence support).

For each primary metric report:
- ON mean, OFF mean, paired mean difference (mean delta),
  median delta, std delta, counts of ON>OFF / ON=OFF / ON<OFF.
- **Paired t-test** (`scipy.stats.ttest_rel`, two-sided, alpha=0.05,
  H0: mean delta = 0).
- **Wilcoxon signed-rank test** (`scipy.stats.wilcoxon`, two-sided,
  zero_method="wilcox"; H0: distribution of deltas symmetric about 0).
- **Bootstrap 95% percentile CI** of the mean delta (B = 10000 resamples,
  random seed = 0 for determinism).
- **Effect size:** paired Cohen's d_z = mean(delta) / std(delta).

Interpretation rule: a difference is reported as statistically supported when
the paired t-test p-value < 0.05 AND the Wilcoxon p-value < 0.05. Otherwise the
difference is reported as not statistically supported at alpha=0.05. No
test is selected based on observed results; all are reported for every primary
metric.
## 6. Topic analysis (descriptive, not ranked)

Per topic C01-C10: n, ON/OFF faithfulness + mean delta, ON/OFF relevance +
mean delta, ON/OFF evidence support + mean delta. No ranking.

## 7. Reproducibility

Run the full paired evaluation twice. Compare per-question substantive fields:
initial evidence IDs, ON decision/refinement/retrieval/refused/evidence IDs,
OFF evidence IDs, ON/OFF answers, ON/OFF faithfulness/relevance/evidence
support. Latency and timestamps non-substantive. Deterministic signatures for
both runs and for the paired comparison.

## 8. Integrity

SHA-256 of frozen artifacts recorded before/after: v1.0, v1.1, chunks,
embeddings, FAISS, BM25, row mapping, reliability config, Task 10D/10E,
Task 3, Experiment 1 run manifests. Any unexpected change -> STOP.

## 9. Outputs (all in this directory)

Scripts, run1/run2 per-question JSON, summary, statistics, topic results,
reproducibility, frozen hashes, run manifest, report, validation json/md.

## 10. Scope limits

No production change; no benchmark change; no retrieval/model/prompt change;
no threshold/weight/budget change; no downloads; no fine-tuning; no causal
claim for refinement from this design; no ranking of topics or systems.

## 5. Secondary analysis

- Gate behavior: ACCEPT/REFINE/RE-RETRIEVE/REJECT counts, refinement rate,
  generation-permitted rate (both conditions).
- Refinement analysis (expected ~84 REFINE cases): initial vs. final
  reliability and evidence IDs only. No "pre-refinement answer" is
  manufactured: the architecture does not generate one, so answer-quality
  causation cannot be isolated from this experiment and will not be claimed.
- Gold evidence analysis: fraction of questions whose final evidence contains
  the gold chunk, Gate ON vs. Gate OFF, and paired difference (descriptive
  only).
   (a) the evidence items shown (initial full set vs. gate-final set), and
   (b) the reliability numbers/decision block (initial vs. gate-final).
   This is the minimal internal control required to bypass enforcement.
   Gate OFF never removes, re-retrieves, rejects, or reorders evidence.

5. **Judging (identical methodology for both conditions; reuse Experiment 1):**
   - Faithfulness: `FAITHFULNESS_PROMPT` + llama3.2 judge (deterministic
     sampler), each condition judged against ITS OWN generation evidence.
   - Answer relevance: `RELEVANCE_PROMPT` + llama3.2 judge (deterministic
     sampler).
   - Evidence support: `span_token_coverage(answer, condition_evidence_text)`.
- System prompt: `scripts/rag_chat.py::GENERATION_SYSTEM_PROMPT` (unchanged).
- Prompt builder: `scripts/build_grounded_prompt.py::build_grounded_prompt`
  (unchanged).