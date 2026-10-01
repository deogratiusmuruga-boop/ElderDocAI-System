# Experiment 3 - Dynamic Care-State and Adaptive-Assistance Evaluation

**Status:** PROTOCOL (frozen before generation)
**Date:** 2026-09-22
**Checkpoint:** `fe20c5ae6351ba2ee3fe3aae8d73d722078b9fbc`
**Mechanism:** `scripts/care_state.py` (real, non-synthetic care-state
estimator, authorized under Option 1) integrated through
`build_grounded_prompt` / `rag_chat` / `carebuddy_service` / `api.main`
(additive; default paths byte-identical to the freeze checkpoint).

## 1. Research questions

- RQ4: Does ElderDocAI respond differently to the same care-related question
  under different dynamic care states?
- RQ5: Does the resulting assistance adapt to the user's current care state
  while preserving evidence grounding?

This evaluates **system behavior** (not clinical effectiveness).

## 2. Care-state mechanism under test (real, non-synthetic)

`scripts/care_state.py::compute_care_state(profile, medications,
appointments, conversation_history, previous_state, previous_score, now,
patient_id)` computes a deterministic state record:

- **Dimensions** (weighted: Medication Burden 0.25, Condition Burden 0.25,
  Encounter Intensity 0.20, Care Complexity 0.20, Interaction Signal 0.10):
  - Medication Burden: unique meds (profile + DB rows); >=5 HIGH, >=3
    MODERATE, >=1 LOW.
  - Condition Burden: # chronic conditions; >=5 HIGH, >=3 MODERATE.
  - Encounter Intensity: # appointments within 30-day horizon.
  - Care Complexity: (conditions+meds)/8 + age boost (>=85 +0.25, >=75 +0.15,
    >=65 +0.05).
  - Interaction Signal: conversation-history length.
- **State bands:** <0.35 all-LOW -> STABLE; <0.40 -> LOW_ACTIVITY; <0.60 ->
  MODERATE_ACTIVITY; >=0.60 -> HIGH_ACTIVITY; no inputs -> NO_DATA.
- **Transition:** INITIAL (no previous), CONTINUATION/STABLE, ESCALATION (UP),
  DE_ESCALATION (DOWN), with score_delta and magnitude.
- **Assistance:** deterministic mode (MAINTENANCE_REINFORCEMENT /
  ACTIVITY_MAINTENANCE / MONITORING_AND_GUIDANCE / ESCALATED_SUPPORT),
  priority (LOW/MEDIUM/HIGH), and an assistance-plan record
  (assistance_strategy, priority, actions, safety_constraints).

Rendering: the existing production chain (`prepare_adaptive_context`,
`prepare_assistance_plan`, `build_grounded_prompt`) renders the same fields
those functions consumed pre-change; a "CARE-STATE CONTEXT (response
adaptation, NOT evidence)" block is added to the prompt ONLY when a real
care-state record is supplied (default path byte-identical to the freeze
checkpoint - verified).
## 3. Controlled variables (identical across state conditions)

- Question (query) - identical.
- Retrieval: window - GeriLit frozen stack (BGE + BM25 + 0.6/0.4 + CE, top-3).
- LLM: llama3.2:latest; temperature 0, top_p 0.1, top_k 10.
- Reliability: 0.3/0.3/0.2/0.1/0.1; ACCEPT>=0.80, REFINE>=0.65,
  RE-RETRIEVE>=0.45, REJECT<0.45; MAX_REFINE=1, MAX_RETRIEVE=1.
- Prompt structure: same template; only the care-state block (and the
  assistance plan derived from the state) differ between conditions.
- User profile: identical between the two conditions (same age/conditions/
  profile medications). Only the REAL care-state signals (DB medication rows,
  appointments, conversation history) differ, which changes the computed
  care state while keeping the profile block identical.

## 4. State conditions (paired contrast)

Shared profile (identical across conditions): age 72, chronic conditions
[diabetes, hypertension, copd], no profile-listed medications.

| Condition | Real signals | Resulting state | Score |
|-----------|--------------|-----------------|-------|
| LOW (A)   | no DB meds, no appointments, no history | LOW_ACTIVITY | 0.2825 |
| HIGH (B)  | 7 DB meds, 3 upcoming appointments, history 4 | HIGH_ACTIVITY | 0.6450 |

Same profile for both conditions; only real DB-level signals differ (medication
rows, appointments, conversation history). The previous_state for condition B
is set to A's state so the transition is recorded as ESCALATION/UP
(descriptive). Both states are produced by the deterministic estimator and
verified before the run (see protocol trace).

## 5. Question set (experiment-only, frozen before run)

- Source: read-only selection from the frozen GeriLit-Gold v1.1 benchmark
  records. The benchmark itself is NOT modified.
- Selection protocol (deterministic, no manual tuning): for each topic
  C01..C10 take the first two records by `final_benchmark_id` order ->
  **20 questions** (C01, C02, ..., C10 x2). Topics with fewer than two
  records are excluded from this experiment (none expected).
- A mirror file records: question_id, question, topic, gold pmcid,
  gold chunk ids, construction note. No question text is altered.

## 6. Outcome measures (descriptive; no composite invented)

Per paired observation record: pair_id, question_id, topic, question,
state_A, state_B, state_difference, evidence_A/B, reliability_A/B,
decision_A/B, answer_A/B, faithfulness_A/B, relevance_A/B,
evidence_support_A/B, adaptation_judgment, appropriateness_judgment, errors.

Metrics (all descriptive):
- **state_response_rate**: fraction of pairs where answer_B != answer_A.
- **evidence_identical_rate**: fraction of pairs where evidence_A ==
  evidence_B (isolating that any answer difference is a prompt/state effect).
- **paired faithfulness/relevance/support deltas** (B - A): mean/median and
  +/- counts.
- **groundedness_rate**: fraction of pairs where faithfulness >= 0.75 for
  BOTH conditions.
- **appropriateness_judgment**: LLM judge with a FROZEN pre-registered prompt
  scoring whether answer_B reflects the higher care need relative to
  answer_A (rubric 0/1 + reason); reported as a rate with raw judgments.
- Transition responsiveness: descriptive only (from the recorded transition).

## 7. Evaluation methodology (identical to Experiments 1-2)

- Faithfulness: `FAITHFULNESS_PROMPT` + llama3.2 judge (temperature 0,
  top_p 0.1, top_k 10), each answer judged against ITS OWN evidence.
- Answer relevance: `RELEVANCE_PROMPT` + same judge configuration.
- Evidence support: deterministic `span_token_coverage(answer, evidence)`.
- Appropriateness judge (NEW, frozen): same model/options; a fixed prompt
  given {question, state_A, state_B, answer_A, answer_B} returns JSON
  {"appropriate": 0|1, "reason": "..."} evaluating only whether the higher
  care-need state receives appropriately stronger/earlier assistance
  language - without rewarding arbitrary differences.
## 8. Procedure

1. Compute real care-state record for each condition (deterministic).
2. Call the production generation path with the override
   (`generate_answer(..., adaptive_context=record)`) - the same code path
   the service/API use; capture answer, evidence, reliability, decision,
   refinement/retrieval attempts, refused, latency.
3. Judge both answers as described.
4. Record full paired observation.

Evidence retrieval depends ONLY on the query and runs through the frozen
GeriLit stack; it is byte-identical across conditions (recorded and verified).

## 9. Reproducibility

Run the full experiment twice. Compare substantive fields (state inputs,
questions, evidence IDs, reliability, decisions, answers, faithfulness,
relevance, support, appropriateness). Judge instability, if it appears (as in
Experiment 2), is reported separately and distinguished from system
variability.

## 10. Integrity

SHA-256 of frozen artifacts before/after: v1.0, v1.1, chunks, embeddings,
FAISS, BM25, row mapping, reliability config, Task 10D/10E, Task 3,
Experiment 1/2 artifacts. Any unexpected change -> STOP.

## 11. Isolated outputs

All under `datasets/elderdocai_geriLit/metadata/experiment3_care_state_adaptation/`.

## 12. Scientific interpretation rules

Report system behavior descriptively. Do NOT claim "personalization is
proven", "the system understands the patient", "clinically optimal care", or
"dynamic care state improves outcomes". No production tuning.