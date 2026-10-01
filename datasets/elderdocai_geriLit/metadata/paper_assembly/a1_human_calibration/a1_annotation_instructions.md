# A1 - Human Annotation Instructions (pre-registered rubric)

You are rating answers produced by a clinical question-answering system that
retrieves evidence from geriatric-literature articles and generates an
answer for **elderly patients with multimorbidity**. The annotation sheet
(`a1_annotation_sheet.csv`) contains 20 question-answer pairs. For each,
read the question and the generated answer, refer to the abridged retrieved
evidence, and enter your ratings.

## Task

Answer the four questions below for **every** row. Leave no cell blank
(use `notes` for any reservations).

1. **faithfulness** — Does the answer stay within what the evidence
   actually states?
   - `1.0` = all claims in the answer are supported by the evidence
   - `0.75` = most claims supported, at most minor overreach
   - `0.5` = mixed: roughly half of the claims are unsupported
   - `0.25` = most claims are unsupported by the evidence
   - `0.0` = answer is essentially fabricated or contradicts the evidence
2. **answer_relevance** — Does the answer directly address the question the
   (older, multimorbid) patient asked?
   - same scale: `1.0` = fully on-topic and useful, ... `0.0` = irrelevant.
3. **evidence_support** — Given the retrieved evidence shown, how well
   supported is the answer?
   - `1.0` = fully supported by the retrieved evidence, `0.0` = no support.
4. **unsupported_claim** — binary `0/1`: enter `1` if the answer contains
   any material claim that is NOT supported by the evidence; else `0`.
   (Use this to detect hallucinated content.)

Rules: rate what the answer literally says; ignore grammar/style; do not
penalize missing information you would like to see; when in doubt choose the
closest of the five scale values.

## Deliverable
Save the filled CSV back in this folder as `a1_human_labels.csv`
(same columns, with the four human_* columns filled). The agreement script
(`a1_compute_agreement.py`) merges your labels with the (hidden) judge
scores and reports exact-match agreement and Cohen's kappa per metric.