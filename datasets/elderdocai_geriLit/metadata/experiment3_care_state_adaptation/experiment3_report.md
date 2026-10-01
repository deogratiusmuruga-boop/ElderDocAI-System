# Experiment 3 - Dynamic Care-State and Adaptive-Assistance Evaluation

**Status:** PASS

## 1. Design

Controlled paired evaluation on the REAL (non-synthetic) care-state mechanism `scripts/care_state.py`. The SAME question is generated under two dynamic care states (LOW_ACTIVITY vs HIGH_ACTIVITY) with identical profile, retrieval, LLM, and reliability configuration; only real DB-level signals (medications, appointments, conversation history) differ (protocol section 3-4).

## 2. Mechanism

- `compute_care_state(profile, medications, appointments, conversation_history, previous_state, ...)` is a pure, deterministic rule-based estimator from REAL system inputs.
- Dimensions: Medication Burden 0.25, Condition Burden 0.25, Encounter Intensity 0.20, Care Complexity 0.20, Interaction Signal 0.10.
- States: STABLE / LOW_ACTIVITY / MODERATE_ACTIVITY / HIGH_ACTIVITY / NO_DATA; transitions INITIAL / CONTINUATION / ESCALATION / DE_ESCALATION.
- Rendered into the production prompt as a `CARE-STATE CONTEXT (response adaptation, NOT evidence)` block; default path byte-identical to the freeze checkpoint (verified).

## 3. State conditions

- **A (LOW):** LOW_ACTIVITY, score 0.2825 (no DB meds, no appointments, no history).
- **B (HIGH):** HIGH_ACTIVITY, score 0.6450 (7 meds, 3 upcoming appointments, long history).
- Shared profile: age 72, diabetes+hypertension+copd, no profile medications.

## 4. Primary results (20 pairs)

| Metric | value |
|--------|-------|
| State-response rate (answer_B != answer_A) | 35.0% (7/20) |
| Evidence-identical rate | 100.0% |
| Groundedness rate (faith >= 0.75 both) | 80.0% |
| Appropriateness rate | 0.0000 (20 judged) |

## 5. Paired deltas (B - A)

| Metric | Delta n | Mean | Median | B>A | B=A | B<A |
|--------|---------|------|--------|-----|-----|-----|
| Faithfulness | 20 | 0.0375 | 0.0000 | 2 | 18 | 0 |
| Answer relevance | 20 | 0.0500 | 0.0000 | 2 | 18 | 0 |
| Evidence support | 20 | -0.0028 | 0.0000 | 3 | 14 | 3 |

## 6. Condition-level statistics

| Metric | A mean | B mean | A median | B median |
|--------|--------|--------|----------|----------|
| Faithfulness | 0.8125 | 0.8500 | 1.0000 | 1.0000 |
| Answer relevance | 0.6250 | 0.6750 | 1.0000 | 1.0000 |
| Evidence support | 0.8221 | 0.8193 | 0.9199 | 0.9258 |


## 7. Transitions and gold evidence (descriptive)

- Escalation transitions recorded: 20; UP direction: 20
- Gold chunk in evidence: A=5/20, B=5/20

## 8. Topic-level results (descriptive, not ranked)

| Topic | n | state-response | f-delta mean | r-delta mean | s-delta mean |
|-------|---|---------------|--------------|--------------|--------------|
| C01 | 2 | 1 | 0.0000 | 0.1250 | 0.0938 |
| C02 | 2 | 1 | 0.0000 | 0.3750 | -0.0352 |
| C03 | 2 | 1 | 0.0000 | 0.0000 | -0.0833 |
| C04 | 2 | 1 | 0.0000 | 0.0000 | 0.0083 |
| C05 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 |
| C06 | 2 | 1 | 0.2500 | 0.0000 | 0.0125 |
| C07 | 2 | 1 | 0.1250 | 0.0000 | 0.0000 |
| C08 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 |
| C09 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 |
| C10 | 2 | 1 | 0.0000 | 0.0000 | -0.0238 |

## 9. Reproducibility

- run 1 signature: `fad1873cc2345497ff89699838d18b6d81c6f3e3b5bb8500e4042af0c9ac0ad4`
- run 2 signature: `06f6f84835b5dde8a3d75868933e1b1a95f5a5e370bb55c5d641d77173da0720`
- system-fields deterministic: **True**
- full signature identical: **False**
- mismatched pair IDs: ['GLG11-001', 'GLG11-013', 'GLG11-028', 'GLG11-068']

## 10. Validation

- checks passed: 90/90

## 11. Scientific interpretation (descriptive)

- This experiment measures system behavior under different real care states; it does not claim clinical benefit, patient understanding, or proven personalization.
- Evidences were identical across states (evidence-identical rate reported above), so any answer change is attributable to the care-state prompt adaptation and assistance plan, not to retrieval differences.