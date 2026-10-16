# Manual Classification of the 28 Operational Unsupported-Content Flags (Experiment 1)

**Total reviewed:** 28

| Category | Count |
|----------|-------|
| REFUSAL | **11** |
| PARTIALLY_SUPPORTED | **12** |
| FABRICATED_OR_CONTRADICTORY | **5** |

**Verification:** X + Y + Z = 11 + 12 + 5 = **28** ✓

**Method.** Each of the 28 responses flagged with the frozen unsupported-content
(`hallucination = faithfulness ≤ 0.50`) flag in
`experiment1_per_question_run1.json` was reviewed manually: the question, the
generated answer, the full retrieved evidence for that question, and the recorded
scores. Categories were assigned from the answer content against the supplied
evidence; the numerical faithfulness score was **not** used to decide any
category (it was only used to identify the flagged set). Frozen experiment
artifacts were not modified.

## Classification table (all 28 cases)

| Question ID | Category | Brief rationale |
| ----------- | -------- | --------------- |
| GLG11-011 | PARTIALLY_SUPPORTED | Answer mirrors evidence chunk 1 (anticholinergic meds → delirium/cognitive impairment/ADRD) but generalizes "strong anticholinergic medications" to "adverse drug events"; substantively grounded |
| GLG11-012 | PARTIALLY_SUPPORTED | Core claim (services/social support reduce caregiver burnout) grounded in chunks 2–3; exact phrase "family social services" not in evidence |
| GLG11-019 | REFUSAL | "I couldn't find that information in the knowledge base." No substantive claim |
| GLG11-023 | PARTIALLY_SUPPORTED | Chunk 1 frames the statin question as "unclear", but chunk 3 reports a demonstrated null; answer echoes the framing yet omits the study result — ambiguous, classified partial |
| GLG11-027 | REFUSAL | "I couldn't find that information in the knowledge base." No substantive claim |
| GLG11-029 | PARTIALLY_SUPPORTED | Supported verbatim by chunk 1 (62.6%; overweight/underweight → undernutrition); minor cross-population generalization |
| GLG11-030 | PARTIALLY_SUPPORTED | Chunk 1 supports protein intake / preserving muscle mass; answer generalizes the weight-loss-context recommendation |
| GLG11-032 | PARTIALLY_SUPPORTED | Figures 2.35% vs 0.82% appear verbatim in chunk 1; grounded though it does not address the question's polypharmacy/UI/depression relation (relevance gap, not fabrication) |
| GLG11-033 | REFUSAL | "I couldn't find that information in the knowledge base." No substantive claim |
| GLG11-036 | FABRICATED_OR_CONTRADICTORY | Answer assigns sarcopenia cutoff ≤7.0 kg/m² to women; evidence chunk 2 states ≤7.0 is the MALE cut-off and ≤5.5 for females — contradicts supplied evidence |
| GLG11-051 | REFUSAL | "I couldn't find that information in the knowledge base." No substantive claim |
| GLG11-052 | REFUSAL | "I couldn't find that information in the knowledge base." No substantive claim |
| GLG11-061 | FABRICATED_OR_CONTRADICTORY | Answer claims multicomponent exercise "showed a balance improvement"; supplied evidence is a study protocol plus a variability discussion — no balance-improvement outcome is reported |
| GLG11-063 | FABRICATED_OR_CONTRADICTORY | Answer claims deprescribing is associated with reduced fracture risk; none of the three evidence chunks reports a deprescribing–fracture finding |
| GLG11-075 | REFUSAL | "I couldn't find that information in the knowledge base." No substantive claim |
| GLG11-079 | FABRICATED_OR_CONTRADICTORY | Answer fuses stroke (AF/exercise chunk 1) with bowel incontinence/preparedness (falls chunk 2); no source supports the composite claim |
| GLG11-082 | REFUSAL | "I couldn't find that information in the knowledge base." No substantive claim |
| GLG11-084 | REFUSAL | Answer explicitly states the association "is not explicitly stated in the provided sources." No substantive claim |
| GLG11-089 | REFUSAL | "I couldn't find that information in the knowledge base." No substantive claim |
| GLG11-091 | PARTIALLY_SUPPORTED | Answer is a verbatim sentence from evidence chunk 1 ("diminished levels of anxiety exert a mitigating influence on vaccine hesitancy") |
| GLG11-092 | REFUSAL | Answer states the gender impact "was not mentioned in the provided SOURCE information." No substantive claim |
| GLG11-102 | PARTIALLY_SUPPORTED | Chunk 1 supports a "small" association (caregiver self-care interventions and mental health); answer rephrases it imprecisely |
| GLG11-107 | PARTIALLY_SUPPORTED | Chunk 2 verbatim: prior influenza-vaccine recipients were 1.9× more likely to seek vaccination; answer grounded |
| GLG11-110 | PARTIALLY_SUPPORTED | Chunk 1 verbatim: clinical decision support during hospitalization improves deprescribing; answer grounded |
| GLG11-111 | PARTIALLY_SUPPORTED | 0.8 g/kg is the general RDA in chunk 2, but chunk 3 reports higher elderly-specific guidance (ESPEN 1–1.2 g/kg); answer is incomplete |
| GLG11-117 | FABRICATED_OR_CONTRADICTORY | No evidence chunk discusses "active aging" maintaining functional abilities/independence; claim unsupported by supplied evidence |
| GLG11-118 | PARTIALLY_SUPPORTED | Chunk 2 supports the life-expectancy / disability-dementia statement nearly verbatim; awkward phrasing only |
| GLG11-120 | REFUSAL | Answer states "The study does not report on cognitive decline high risk in older adults." No substantive claim |

## Ambiguous cases (documented, not silently re-categorized)

1. **GLG11-011, GLG11-032, GLG11-091, GLG11-107, GLG11-110, GLG11-118** — the answers are substantively grounded in (often verbatim from) the supplied evidence yet carried a low faithfulness flag. Under the three-category scheme they are classified PARTIALLY_SUPPORTED; these look like flag artifacts (judge/evidence mismatch) rather than fabricated or contradictory content. They are recorded as PARTIALLY_SUPPORTED per the category definitions and are explicitly flagged here for transparency.
2. **GLG11-023** — the evidence itself is internally ambiguous: chunk 1 frames the statin–dementia association as "unclear", while chunk 3 reports a demonstrated null association. Classified PARTIALLY_SUPPORTED with the tension documented.
3. **GLG11-111** — partial in the sense of completeness: 0.8 g/kg is the general RDA present in the evidence, but the same evidence reports higher elderly-specific guidance (ESPEN ≥1–1.2 g/kg), making the answer's blanket claim incomplete.

## Caveat

These categories were assigned by manual (human-assisted) inspection of each
answer against its full retrieved evidence; they are an audit artifact and do
not alter any frozen Experiment 1 result or flag.