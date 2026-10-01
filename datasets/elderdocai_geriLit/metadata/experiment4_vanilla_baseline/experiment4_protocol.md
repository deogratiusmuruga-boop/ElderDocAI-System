# Experiment 4 - Vanilla-RAG Baseline on GeriLit-Gold v1.1

**Status:** PROTOCOL (frozen before generation)
**Date:** 2026-09-22
**Checkpoint:** `fe20c5ae6351ba2ee3fe3aae8d73d722078b9fbc`
**Benchmark:** GeriLit-Gold v1.1, SHA-256
`1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72`, n = 121.

## 1. Core question

Does the complete ElderDocAI generation pathway produce different
generation-quality outcomes from a conventional Vanilla-RAG pathway when
BOTH receive the same question and the same retrieved evidence?

This isolates the downstream framework/generation pathway (reliability
gating, refinement, ElderDocAI grounded prompt) vs plain evidence-grounded
generation, holding retrieval, LLM, generation parameters, and evaluator
constant.

## 2. Design (paired, controlled)

For each of the 121 benchmark questions:

1. Compute the frozen initial evidence ONCE:
   `chunks = retrieve_evidence(question)` (GeriLit frozen stack: BGE + BM25 +
   0.6/0.4 hybrid + CrossEncoder, top-3);
   `initial_evidence = prepare_evidence(chunks)`.
   This SAME evidence is supplied to both conditions.

2. **Condition FULL (ElderDocAI framework):** run the production generation
   pathway (`generate_answer(question, return_evaluation=True)`) - the exact
   code path used by the service/API, including reliability evaluation, the
   programmatic gate (REFINE/RE-RETRIEVE/REJECT budgets), and the ElderDocAI
   grounded prompt. No adaptive_context override (care state not part of this
   head-to-head; the framework's default generation behavior).

3. **Condition VANILLA:** an ordinary evidence-grounded RAG generation using
   the SAME initial evidence, a minimal frozen prompt (protocol section 4),
   the same LLM (llama3.2:latest), and the same generation options
   (temperature 0, top_p 0.1, top_k 10). No reliability, no gate, no
   refinement, no re-retrieval, no rejection, no adaptive context, no
   assistance plan, no profile.

## 3. Controlled variables (identical across conditions)

- Question (query)
- Retrieved evidence (FULL initial == VANILLA evidence; verified per
  question)
- LLM (llama3.2:latest)
- Generation options (temperature 0, top_p 0.1, top_k 10)
- Evaluator (FAITHFULNESS_PROMPT / RELEVANCE_PROMPT judges + deterministic
  span_token_coverage)
- Retrieval backend (geri_lit, frozen)

## 4. Vanilla-RAG prompt (frozen baseline)

Existing repository plain-RAG prompt audit (recorded in the report):
- All historical generation prompts in the repository are the ElderDocAI
  grounded prompt (`scripts/build_grounded_prompt.py`); the ablation
  conditions A0/A1/A5 are variants of that prompt (A5 strips only the
  reliability section but retains ElderDocAI instructions, profile, care
  state, and assistance plan).
- No plain prompt that predates the reliability/adaptive mechanisms
  free of all framework metadata exists. Therefore a MINIMAL baseline prompt
  is defined below, following the experiment specification template.

VANILLA_SYSTEM = "Answer the user's question using only the provided evidence."

VANILLA_PROMPT (user message) =
"Answer the user's question using only the provided evidence.

If the evidence does not contain enough information to answer the question,
state that the evidence is insufficient rather than inventing information.

Retrieved evidence:
{evidence_text}

Question:
{question}"

where {evidence_text} renders each evidence item as a numbered block:
"Evidence {i}:\n{text}".

This prompt contains NO reliability metadata, no care-state, no
adaptive-assistance metadata, no policy instructions, no profile, no
framework-specific instructions.

The exact baseline prompt text and its SHA-256 are recorded in the run
manifest.

## 5. Evidence identity control (critical)

For every question, the validator requires:
  VANILLA.evidence_ids == FULL.initial_evidence_ids
and both are recorded per question. FULL may subsequently refine its
evidence via the frozen gate (recorded as refinement_attempts); FULL's
generation may therefore use a refined subset, but the INITIAL evidence
matches VANILLA exactly.

## 6. Evaluation methodology (identical to Experiments 1-3)

- Faithfulness: FAITHFULNESS_PROMPT + llama3.2 judge (temperature 0,
  top_p 0.1, top_k 10), each condition judged against ITS OWN generation
  evidence.
- Answer relevance: RELEVANCE_PROMPT + same judge configuration.
- Evidence support: deterministic span_token_coverage(answer, evidence).
- Paired deltas (VANILLA - FULL or FULL - VANILLA, defined in the stats
  block) reported with means/medians/+- counts and both a paired t-test and
  a Wilcoxon signed-rank test (pre-registered: a difference is reported as
  statistically supported when BOTH p-values < 0.05).

## 7. Per-question record

Each pair records: pair_id, question, topic, gold ids, FULL (answer,
evidence_ids, reliability, decision, refinement/retrieval attempts, refused,
generated, faithfulness, relevance, evidence_support, latency, error),
VANILLA (answer, evidence_ids, generated, faithfulness, relevance,
evidence_support, latency, error), paired_differences, and evidence identity
flag.

## 8. Reproducibility

Run the full experiment twice. Compare substantive fields (evidence IDs,
decisions, answers, faithfulness, relevance, support). Judge-instrument
variability (as in Experiments 2-3) is reported separately from system
variability.

## 9. Integrity

SHA-256 of frozen artifacts before/after: v1.0, v1.1, chunks, embeddings,
FAISS, BM25, row mapping, reliability config, Task 10D/10E, Task 3,
Experiments 1-3 artifacts. Any unexpected change -> STOP.

## 10. Isolated outputs

All under
`datasets/elderdocai_geriLit/metadata/experiment4_vanilla_baseline/`.

## 11. Scientific interpretation rules

Report descriptive differences. Do NOT claim the framework is "better" or
"worse" beyond the measured, statistically supported differences. This
experiment does not measure retrieval, clinical benefit, or generalizability
beyond the frozen benchmark.