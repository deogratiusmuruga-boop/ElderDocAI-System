# A1 — Matched-Evidence Human-Calibration Protocol (frozen)

**Protocol version:** `1.0-matched-evidence`
**Status:** FROZEN before human annotation
**Sample:** n = 20 (seed 42) — the exact original A1 pilot question IDs
**Source of truth:** Experiment 1 run-1 per-question artifact
**Supersession:** this matched-evidence protocol replaces the original
abridged-evidence pilot for the primary human–judge calibration analysis.

---

## Study purpose

This is an evaluator-concordance/calibration analysis between one human
annotator and the automated LLM judge.

It is NOT a validation study.

> This study is an evaluator-concordance/calibration analysis: a single human
> annotator's judgments are compared with the automated judge's scores on n=20
> matched-evidence items. It does not validate or invalidate the judge, and no
> claim about judge accuracy, precision, bias, or clinical validity is drawn
> from n=20.

## Sample

> This is a calibration sample of n=20 stratified by the automated judge's
> faithfulness outcome (11 high / 4 partial / 5 low, seed 42). It is not a
> representative sample of the 121-question benchmark; no population-level or
> benchmark-level rate is derived from it.

The stratum label is NOT exposed to the human annotator.

## Evidence matching

For each question the annotator sees `evidence_full` = the exact recorded
final evidence texts for that question from the frozen Experiment 1 run-1
artifact (`retrieved_evidence_texts`), joined in the exact recorded order,
WITHOUT truncation, summarization, paraphrase, regeneration, reordering,
or added text. No gold evidence, reliability scores, gate decisions, or judge
metadata are added. These are the same full evidence texts the Experiment 1
automated judge consumed.

## Faithfulness rubric

* `1.00` — All substantive claims are supported by the provided evidence; no meaningful overreach.
* `0.75` — Essentially supported, with only minor imprecision or limited overstatement.
* `0.50` — Mixed support; some substantive claims are supported while others are weakly supported or unsupported.
* `0.25` — Mostly unsupported or substantially overreaching.
* `0.00` — Unsupported or contradicted by the provided evidence.

## Answer relevance rubric

* `1.00` — Directly answers the question and is fully useful/relevant.
* `0.75` — Mostly answers the question with minor irrelevant or incomplete content.
* `0.50` — Partially answers the question; meaningful relevant content is present but incomplete.
* `0.25` — Largely off-topic or minimally answers the question.
* `0.00` — Does not answer the question.

## Evidence-support rubric

* `1.00` — Essentially all substantive answer content is directly supported by the provided evidence.
* `0.75` — Most substantive content is supported, with minor unsupported gaps.
* `0.50` — Approximately half of the substantive content is directly supported.
* `0.25` — Only limited answer content is directly supported.
* `0.00` — No meaningful answer content is supported by the provided evidence.

Human instruction:

> Score how much of the answer is directly supported by the evidence you see.

## Unsupported-claim rubric

Binary:

* `1` — At least one substantive unsupported claim is present.
* `0` — No substantive unsupported claim is present.

## Evidence-support comparability rule

The automated evidence-support metric is continuous span-token coverage.

For the primary human–judge comparison, map the automated continuous value to
the same five-point scale using these frozen bins:

* `[0.00, 0.125)` → `0.00`
* `[0.125, 0.375)` → `0.25`
* `[0.375, 0.625)` → `0.50`
* `[0.625, 0.875)` → `0.75`
* `[0.875, 1.00]` → `1.00`

Boundary rule:

> Exact midpoint boundaries are assigned to the lower label.

Primary reporting:

* exact agreement on the five-point scale
* Cohen's kappa on the five-point scale

Secondary robustness:

* exact agreement plus within-one-bin agreement

Transparency:

* retain the raw continuous automated support values in the analysis artifact
  later; do NOT expose them to the human annotator.

## Cohen's kappa degeneracy rule

If either rater has a zero-variance marginal, defined as at least 19 of 20
observations receiving the same category:

Report:

`κ = degenerate — zero-variance marginal`

and also report:

* observed agreement (`Po`)
* expected agreement (`Pe`)
* exact agreement

Do NOT interpret a mathematically degenerate κ as ordinary disagreement.

Numerical κ should only be reported when neither marginal is degenerate and
`Pe < 0.90`.

This rule applies symmetrically to all evaluated categorical metrics.

## Blinding

The annotator must not have access to:

* automated scores
* judge reasoning
* benchmark gold labels
* topic labels
* stratum labels
* gate decisions
* reliability scores
* gold-chunk indicators
* experiment condition labels

The evidence and answer must be presented without judge-derived annotations.

## Pilot relationship

The original A1 pilot remains preserved as a historical methodological pilot.

The matched A1 is the primary human–judge calibration analysis.

The pilot should not be presented as a second independent calibration estimate.

## No validation claim

Do not describe the result as:

* judge validation
* judge invalidation
* judge accuracy
* judge precision
* judge bias estimation
* clinical validity