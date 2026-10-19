# ElderDocAI — An Evidence-Grounded, Reliability-Gated Adaptive Framework for Elderly Care Assistance
### Phase 3 evaluation on the GeriLit-Gold v1.1 benchmark (frozen results)

*Working title — final title to be aligned with the target venue. All statistics in this
manuscript are frozen; experiments are read-only artifacts. The primary A1 calibration
analysis is the matched-evidence human–judge calibration (n = 20; Section 7); the
original abridged-evidence pilot is historical only.*

---

## Abstract

Large conversational models are increasingly considered for caregiver-facing
assistance, yet ungrounded symptom interpretations and risk statements make
unconstrained use unsafe in elderly care. We present **ElderDocAI**, whose final
stack separates (i) a deterministic personalization layer — validated at component
and controlled-evaluation level (unit tests; Experiment 3), with longitudinal
behavior explicitly not yet evaluated — that turns documented care activity into a
rule-based adaptive assistance plan bounded by explicit no-diagnosis /
no-risk-prediction guardrails, and (ii) an
evidence-grounded generation layer that retrieves from a curated geriatric-literature
corpus, applies a programmatic reliability gate, and generates under a strict
evidence-only mandate (temperature 0).

On the frozen GeriLit-Gold v1.1 benchmark (121 clinical questions, 45 PubMed
Central articles), we report the locked evaluation of the final v1.1 stack:
mean faithfulness **0.754** (95% CI 0.684–0.825), answer relevance **0.672**
(0.591–0.752), and evidence support **0.853** (0.810–0.896), with an
hallucination-flag rate of **23.1%** (95% binomial CI 16.0–31.7%).
In a controlled comparison holding retrieved evidence identical across
conditions, the full framework outperformed a minimal vanilla-RAG baseline on
all three metrics (paired deltas −0.194, −0.140, −0.179 for faithfulness,
relevance, support; all deltas statistically supported by paired t-test and
Wilcoxon signed-rank tests). A reliability-gate ablation found **no statistically
supported benefit** of the gate on this benchmark (a transparent negative
result), and a care-state adaptation study confirmed a real, evidence-identical
mechanism with low observed adaptation (state-response 35%; 0/20 judged
appropriate), reported descriptively rather than as a clinical claim.
LLM-judge run-to-run reproducibility was high (90.0–100% exact match across
metrics and experiments).

In a generator-model ablation (Experiment 5) holding the same retrieval,
reliability gate, grounded prompt, sampling options (temperature 0, top_p 0.1,
top_k 10), and Llama‑3.2 judge constant, five open‑weight decoder models were
evaluated on the identical 121‑question benchmark. Phi3.5 achieved the highest
mean faithfulness (0.900–0.923; n = 120) and the lowest hallucination‑flag rate
(6.6–8.3%), but exhibited a repeatable token‑repeat‑limit generation failure on
one question in both runs; mistral:7b produced the most balanced quality with
high evidence support (0.82) and a moderate hallucination‑flag rate of 15.7–17.4%.
Retrieval metrics (Recall@k, Hit@k, Precision@k, MRR, nDCG@k, MAP) were
computed from the ordered gold‑relevant‑chunk evidence and are reported
separately (gold‑hit rate 27.3% for k=3; MRR = MAP = 0.2383), reflecting the
frozen retrieval layer that is identical across all generator conditions.

The central contribution is the *joint* evaluation discipline: frozen benchmark,
pre-registered analyses, exhaustive validators, transparent reporting of
null results, and a controlled generator-model comparison against five
alternative decoder models.

---

## 1. Introduction

Informal caregivers of older adults with multimorbidity face a high information
burden and a genuinely unsafe failure mode in generative AI: a confident but
ungrounded answer can be mistaken for medical advice. Retrieval-augmented
generation (RAG) mitigates but does not eliminate this risk, because answer
quality depends on retrieved evidence quality, prompt construction, and
generator adherence to evidence.

ElderDocAI is designed around two empirically separable research claims.
First, the *deterministic core* — a rule-based care-state estimator and adaptive
assistance planner over documented activities (medications, appointments,
conversation history) — must be validated to the level demonstrated in this paper
(component unit tests; controlled behavioral evaluation) and must never overstep
into diagnosis or risk prediction. Second, the *generation layer* must stay
grounded in retrieved evidence and be *measurably reliable* under a
pre-registered evaluation protocol.

This manuscript reports the Phase 3 evaluation of the final v1.1 stack. It
makes four contributions: (1) a complete, frozen evaluation of generation
quality on a purpose-built geriatric QA benchmark; (2) controlled ablations
that measure the effects of the reliability gate, the grounded-prompt pathway,
and the care-state mechanism; (3) a reliability-quantification apparatus
(two-run judge reproducibility, confidence intervals, human-calibration
infrastructure) that supports scientific reporting of LLM-judge-based metrics;
and (4) a generator-model comparison that isolates the effect of the generator
LLM by holding retrieval, prompts, the reliability gate, and the judge
unchanged, allowing comparison of five open‑weight decoder models under a
single shared RAG pipeline. We report honestly mixed results, including one
statistically null ablation, because the evaluation discipline — not
selectively positive outcomes — is the contribution.

---

## 2. Related Work

**RAG for clinical text.** Retrieval-augmented generation is the dominant
architecture for grounding open models in domain corpora. Clinical deployments
inherit standard failure modes: missed evidence, boundary hallucination, and
evidence-answer misalignment (Gao et al.; Lewis et al.). Our evaluation applies
three complementary metrics — LLM-judged faithfulness, LLM-judged relevance,
and deterministic span-based evidence support — following the RAGAS-style
decomposition used in recent clinical QA work.

**Reliability gating and refusal.** Multi-stage pipelines that verify retrieval
relevance or self-consistency before answering reduce, but cannot eliminate,
ungrounded generation. We contribute a *programmatic* gate (thresholds on a
composite reliability score with REFINE / RE-RETRIEVE / REJECT transitions) and
— critically — an ablation that measures whether gating changes answer quality
on a benchmark where refinement rarely alters evidence.

**Adaptive behavior in care technology.** Prior adaptive-care work emphasizes
escalation policies for monitoring or alerting. ElderDocAI instead adapts the
*support style* of a conversational assistant to a rule-based care state, and we
explicitly evaluate whether the injected care-state context changes the
generated answer (state-response) while holding evidence identical.

**LLM judges as measurement instruments.** LLM-as-judge scores are practical
but must be treated as instruments with their own reproducibility characteristics.
We quantify run-to-run agreement of our judge (Section 5.6) and pre-register and
run an external human-calibration study (Section 7) so that judge scores cannot
be mistaken for ground truth.
---

## 3. System Design (final stack, v1.1)

The generation path evaluated here consists of four stages.

**3.1 Hybrid retrieval.** For each question, the retriever combines a dense
backbone (BGE-base-en-v1.5, top-5) and BM25 (top-5), fuses them with a
0.6/0.4 hybrid score, and reranks the union with a MiniLM-L-6-v2 CrossEncoder
to produce the final top-3 evidence chunks. The *initial* retrieval is the
shared, fixed input for all paired comparisons; the downstream REFINE step of
the reliability gate may subsequently filter the final evidence set (it changed
the evidence ID set in only 1/84 REFINE cases in Experiment 2).

All Experiment 1–4 results in this paper measure the configuration in which
retrieval is served by the frozen GeriLit stack — 17,930 curated
geriatric-literature chunks, retrieved by BGE dense top-5 and stored BM25
top-5, fused 0.6/0.4 hybrid, CrossEncoder-reranked, final top-3 — selected
through the production retrieval router by `ELDERDOCAI_RETRIEVAL_BACKEND=geri_lit`.
The shipped service default for the retrieval backend remains legacy: when the
environment variable is unset, the router serves a separate 1,143-chunk
caregiver-manual knowledge base. The legacy backend is a distinct corpus and
configuration and is not the subject of any result reported in this paper.
Deployment of the evaluated configuration therefore requires explicitly enabling
the GeriLit backend; every performance claim below applies to that GeriLit
configuration and does not describe the unmodified legacy default.

**3.2 Reliability evaluator and gate.** Each question receives a composite
reliability score

  reliability = 0.3·Authority + 0.3·Relevance + 0.2·Support +
                0.1·Coverage + 0.1·Consistency,

where Relevance uses the relevance-normalization fix (dense cosine similarity
semantics). The programmatic gate applies the fixed thresholds
ACCEPT ≥ 0.80, REFINE ∈ [0.65, 0.80), RE-RETRIEVE ∈ [0.45, 0.65),
REJECT < 0.45, with one refinement and one re-retrieval budget
(MAX_REFINE = 1, MAX_RETRIEVE = 1). REFINE selects evidence items sharing at
least two query content terms, re-evaluates reliability, and generates once
from the surviving evidence; retrieval is not repeated and no separate
pre-refinement answer is produced.

**3.3 Grounded generation (ElderDocAI prompt).** The generator (Llama 3.2,
3.2B, temperature 0, top_p 0.1, top_k 10) is constrained to answer only from
the provided evidence and to explicitly state when the evidence is
insufficient rather than inventing content. System metadata (reliability
components, gate decision) is injected as control signals, not as evidence.

**3.4 Care-state mechanism.** A pure, deterministic rule-based estimator
(`care_state.py`) computes a care state from real database-level signals:
medication burden (0.25), condition burden (0.25), encounter intensity (0.20),
care complexity (0.20), and interaction signal (0.10), with states
STABLE / LOW_ACTIVITY / MODERATE_ACTIVITY / HIGH_ACTIVITY / NO_DATA. The
INITIAL / CONTINUATION / ESCALATION / DE_ESCALATION transition vocabulary is
implemented and unit-tested, and Experiment 3 exercised transition behavior
using controlled previous-state inputs. The production API currently does not
persist or pass `previous_state` / `previous_score`, so API-served requests
currently resolve to INITIAL/NONE transitions; longitudinal care-state
transition behavior has not yet been evaluated. The state is rendered into the
prompt as a CARE-STATE CONTEXT block explicitly marked "response adaptation,
NOT evidence".

**3.5 Deterministic safety boundary.** The care-state output is used for
response adaptation only: it is explicitly not evidence, care-state labels must
not be converted into medical claims, and assistance planning is constrained to
response structure/pacing and grounded interaction. The generation system
prompt applies no-diagnosis / no-risk-prediction style constraints, and
retrieved evidence remains the basis for all grounded factual content. Plan
correctness is out of scope for generation-quality scoring (no ground-truth
labels exist), consistent with the frozen protocol.

---

## 4. Evaluation Methodology

**4.1 Benchmark.** GeriLit-Gold v1.1 is a frozen benchmark of 121 accepted
clinical questions derived from 45 PubMed Central articles (116 unique gold
chunks; SHA-256 `1488d164...d72`), covering 10 topics (C01–C10). Every question
has a gold chunk, and every generated answer is compared against the retrieved
evidence set.

**4.2 Metrics.** Faithfulness and answer relevance are LLM-judged on the
0/0.25/0.5/0.75/1.0 scale by Llama 3.2 under a deterministic sampler, using the
verbatim prompts from the evaluation suite. Evidence support is deterministic:
the token-level overlap of the answer spans with the final evidence. An
hallucination-flag = faithfulness ≤ 0.50 (flagged by the LLM judge; the flag
does not classify the answer type — refusals, partial‑support, and
contradictions are all captured under this binary flag); contradicting
claims = faithfulness == 0.0.

**4.3 Pre-registered statistical rules.** For paired ablations, a metric is
flagged statistically supported only when BOTH a paired t-test and a Wilcoxon
signed-rank test return p < 0.05 (joint rule), with bootstrap and t-based 95%
CIs reported alongside. Proportions use exact Clopper-Pearson binomial CIs.

**4.4 Reproducibility regime.** Every experiment was run twice on the frozen
benchmark; per-question artifacts, reproducibility manifests, and exhaustive
validators (63/63, 89/89, 90/90, 94/94 checks for Experiments 1–4) are
archived under the experiment directories.

**4.5 Scope of the benchmark runs.** Experiments 1–2 and 4 evaluate the fixed
retrieval/generation pathway and do not constitute an evaluation of
longitudinal care-state adaptation. Care-state/adaptive-assistance behavior was
specifically evaluated in Experiment 3 using controlled state inputs; the
121-question benchmark runs did not inject profile or care-state context.

**4.6 Experiment 5 — generator-model ablation.** To isolate the contribution
of the generator LLM, five open-weight decoder models — llama3.1:8b,
qwen2.5:7b, mistral:7b, gemma2:9b, and phi3.5 — were evaluated on the identical
121-question GeriLit-Gold v1.1 benchmark under a *single shared RAG pipeline*.
Only the generator model string in the production generation call differs;
retrieval, the reliability gate, the grounded prompt, the sampling options
(temperature 0, top_p 0.1, top_k 10), and the Llama‑3.2 judge are held
byte-identical to Experiment 1. This is a generator comparison within one RAG
pipeline, not a comparison of five distinct RAG architectures.

Each generator was run twice (run-id 1 and run-id 2) with identical inputs and
sampling. Model identity (tag, digest, quantization) was recorded per run from
`ollama show`. The number of context tokens is not explicitly set
(`num_ctx` unset), matching Experiment 1; the Ollama server default context
window therefore applies identically across all generators.

**Evaluation metrics.** Faithfulness is the primary comparison metric;
answer relevance and evidence support are reported as supporting metrics, and
the hallucination-flag rate (faithfulness ≤ 0.50) and contradiction count
(faithfulness == 0.0) are secondary flags. Because the benchmark has no gold
reference *answers* (only gold evidence chunks), EM / F1 / ROUGE / BLEU /
BERTScore are not computable and are not reported. RAGAS context-precision /
context-recall and TruLens / ARES scorers would require additional judge
tooling — incompatible with the frozen judge — and are therefore excluded
(recorded as future work, Section 9).

**Retrieval metrics.** Because the retrieval layer is frozen and byte-identical
across all five generators, retrieval quality is computed once and reported
separately from generator results. For each question, a retrieved chunk is
relevant iff its chunk_id equals the question's gold relevant chunk. With the
frozen cross-encoder rerank producing an ordered top-3 list (k ∈ {1, 2, 3}),
we compute Recall@k, Hit@k, Precision@k, MRR, nDCG@k, and MAP. MAP equals MRR
here because every question has exactly one gold chunk, so per-question
average precision equals reciprocal rank.
---

## 5. Results

### 5.1 Experiment 1 — Generation quality of the final stack (n = 121)

**Table 2. Generation quality (95% CIs: t-interval; binomial CI for rates).**

| metric | mean | sd | 95% CI |
|--------|------|----|--------|
| Faithfulness | 0.7541 | 0.3966 | [0.6835, 0.8248] |
| Answer relevance | 0.6715 | 0.4496 | [0.5914, 0.7516] |
| Evidence support | 0.8528 | 0.2408 | [0.8099, 0.8957] |

- Hallucination-flag rate (faithfulness ≤ 0.5): **23.1%** (28/121;
  95% binomial CI [16.0%, 31.7%]).
- Contradiction rate (faithfulness == 0.0): **18.2%** (22/121).
- Judge reproducibility for this experiment was perfect between the two frozen
  runs (exact match 100% on all three metrics; Table 6).

Figure: `figures/fig2_generation_quality.png` (means with 95% CIs).

The distribution is strongly bimodal (mean below median 1.000): most answers
are fully faithful, but a substantial minority are complete contradictions.
Section 5.5 characterizes these failures; they coincide exactly with the
absence of the gold chunk from the retrieved evidence.

### 5.2 Experiment 2 — Reliability gating (transparent null result)

Controlled within-subject ablation over identical initial retrieval: the
production gate replayed (ON) vs. reliability computed but not enforced (OFF).

**Table 3. Gate ON vs OFF (n = 121; delta = ON − OFF).**

| metric | ON mean | OFF mean | mean delta | t p | Wilcoxon p | bootstrap 95% CI | ON=OFF |
|--------|---------|----------|------------|-----|-----------|-----------------|--------|
| Faithfulness | 0.7541 | 0.7810 | −0.0269 | 0.0685 | 0.0684 | [−0.0579, −0.0021] | 115 |
| Answer relevance | 0.6715 | 0.6674 | 0.0041 | 0.7404 | 0.5887 | [−0.0207, 0.0289] | 115 |
| Evidence support | 0.8528 | 0.8543 | −0.0015 | 0.5351 | 0.4236 | [−0.0061, 0.0031] | 110 |

No metric met the pre-registered joint rule (paired t-test p < 0.05 AND
Wilcoxon p < 0.05). Figure: `figures/fig3_gating_contrast.png`.

Interpretation: under the joint rule this ablation found **no statistically
supported effect** of gating. Two descriptors explain why: refinement changed
the evidence ID set in only **1/84** REFINE cases on this benchmark, so most
answers (115/121 for faithfulness) were byte-identical between conditions, and
the reliability estimator correlates weakly with actual faithfulness
(Spearman ρ = 0.06, Section 5.5). This is a property of this
benchmark + gate combination; it does not license the conclusion that gating
is useless elsewhere. The gate additionally rejected 0 questions at
RE-RETRIEVE/REJECT (121/121 generations permitted in both arms).

### 5.3 Experiment 3 — Care-state adaptation (descriptive)

Paired generation of the *same* question under two real care states
(A = LOW_ACTIVITY: score 0.283; B = HIGH_ACTIVITY: score 0.645) with identical
profile, retrieval, generator, and reliability configuration; only real
database-level signals differ.

**Table 4. Care-state contrast (n = 20 pairs; A = LOW, B = HIGH).**

| metric | A mean | B mean | mean delta (B−A) | B>A | B<A |
|--------|--------|--------|------------------|-----|-----|
| Faithfulness | 0.8125 | 0.8500 | 0.0375 | 2 | 0 |
| Answer relevance | 0.6250 | 0.6750 | 0.0500 | 2 | 0 |
| Evidence support | 0.8221 | 0.8193 | −0.0028 | 3 | 3 |

- State-response rate (answer_B ≠ answer_A): **35.0%** (7/20; 95% binomial CI
  [15.4%, 59.2%]).
- Evidence-identical rate: **100.0%** — any answer change is attributable to the
  care-state prompt adaptation alone.
- Appropriateness: **0/20** scored appropriate by the judge (95% CI
  [0%, 16.8%]).

Figure: `figures/fig5_care_state_contrast.png`.

This confirms a *real, working mechanism*: the care-state context changes
generated answers under strict evidence identity. It also documents low
observed adaptation quality: two-thirds of answers are insensitive to the state,
and no answer was judged "appropriate" under the frozen rubric. These numbers
are descriptive of system behavior, not a clinical or personalization claim.

### 5.4 Experiment 4 — Vanilla-RAG baseline (headline contrast)

Controlled paired comparison holding the identical frozen retrieval evidence in
both arms: the FULL ElderDocAI generation pathway (gate + refinement + grounded
prompt) vs. a minimal vanilla-RAG prompt ("answer using only the evidence").
Evidence identity held on all 121 pairs (identity failures: none).

**Table 5. FULL vs Vanilla-RAG (n = 121; delta = VANILLA − FULL).**

| metric | FULL mean | VANILLA mean | mean delta | t p | Wilcoxon p | supported | V<F |
|--------|-----------|--------------|------------|-----|-----------|-----------|-----|
| Faithfulness | 0.7645 | 0.5702 | −0.1942 | <0.0001 | 0.0001 | ✓ | 38 |
| Answer relevance | 0.6880 | 0.5475 | −0.1405 | 0.0025 | 0.0028 | ✓ | 42 |
| Evidence support | 0.8550 | 0.6765 | −0.1785 | <0.0001 | <0.0001 | ✓ | 99 |

Figure: `figures/fig4_vanilla_contrast.png`.

On every metric the FULL pathway exceeded the vanilla baseline and every paired
delta cleared the joint significance rule. Because both conditions were
initialized with identical retrieved evidence, the comparison isolates
differences in the downstream generation and reliability-controlled pathway
rather than the initial retrieval stage. No single component is causally
attributed with the difference; in particular Experiment 2 found refinement
changed the evidence ID set in only 1/84 REFINE cases, which further limits any
component-level attribution. This is the downstream pathway result of
the study.
### 5.5 Failure-mode diagnostics (B4/B5)

Full diagnostics are archived in
`b4_v11_failure_diagnostics/b4_v11_failure_diagnostics.{json,md}`.

**Taxonomy.** Of the 28 unsupported answers, 22 are full contradictions
(faithfulness = 0) and 6 are partially unsupported. Failure rates are
concentrated in topics C03 (6/13 = 46.2%), C10 (3/6 = 50.0%), C09 (3/10 =
30.0%), and C07 (4/14 = 28.6%).

**Retrieval failures coincide with unsupported-generation outcomes.** The gold
chunk was present in the retrieved evidence for **0 of the 28 failing
questions** and for 33/121 overall. Failure rate with a gold chunk present:
**0.0%** (0/33) vs. **31.8%** with it absent (28/88). On this benchmark,
unsupported answers coincide exactly with retrieval failure to surface the
gold content — a clean failure locus and a clear engineering lever (retrieval
quality / re-retrieval policy).

**Gate calibration.** ACCEPT and REFINE outcomes had similar observed
hallucination rates (24.3%, 9/37, vs 22.6%, 19/84; no hypothesis test was
applied to this comparison); mean final reliability of failing questions 0.786
— above the ACCEPT threshold.
Spearman rank correlation between the composite reliability estimator and
judge-assessed faithfulness is **0.061**, i.e., the estimator is nearly
uninformative about factual faithfulness on this benchmark. The gate cannot be
expected to suppress failures that its input score does not discriminate.

**Failure examples** (all 28 rows, deterministic sort) appear in
`b4_v11_failure_diagnostics.md`; representative excerpt:

> **Q:** According to the article, what is the association between cognitive
> ... in older adults? **A:** I couldn't find that information in the
> knowledge base.

This excerpt also illustrates a rubric edge: refusals and low-content answers
received faithfulness 0.0 — the judge treats them as ungrounded claims, which
is conservative but inflates the contradiction count for withheld answers.

### 5.6 LLM-judge reproducibility (A2)

The evaluation judge (Llama 3.2, deterministic sampler) was re-run on every
frozen per-question artifact; per-metric exact match and Cohen's kappa across
the two runs:

**Table 6. Judge run1-vs-run2 reproducibility (min–max over metrics).**

| experiment | n | exact match min–max % | kappa min–max |
|------------|---|----------------------|---------------|
| Exp 1 (generation) | 121 | 100.0 | 1.000 |
| Exp 2 (gating) | 121 | 98.3–100.0 | 0.9695–1.000 |
| Exp 3 (care state) | 20 | 90.0–100.0 | 0.8214–1.000 |
| Exp 4 (vanilla) | 121 | 90.1–100.0 | 0.8784–1.000 |

Figure: `figures/fig6_judge_reliability.png`.

Only the answer-relevance judge shows boundary instability (identical input
text, different 0.25-grid scores for ≤ 2 questions per run in Exp 2 and ≤ 12
in Exp 4); faithfulness/support judgments were stable where tested. Judged as
a measurement instrument, the judge is highly reproducible on this benchmark,
with the residual variation isolated to the fine-grained relevance boundary.
One statistical artifact is documented: the Exp 3 binary "appropriateness"
kappa collapses to 0.0 despite 95% exact match because run 1 rated all 20
items identically (the kappa paradox); exact-match agreement is reported
alongside kappa.

### 5.7 Experiment 5 — Generator-model comparison (n = 121)

Five generators were evaluated under the shared GeriLit pipeline (Section 4.6).
All models produced complete (non-empty) answers on the benchmark; phi3.5 is
the exception (see below). Judge reproducibility for Experiment 5 is reported
alongside the two-run faithfulness means in Table 7.

**Table 7. Generator-model comparison (R1/R2; faithfulness n = 121 except
phi3.5 n = 120). 95% t-CIs for means; exact binomial CIs for rates.**

| Generator | Faithfulness R1 (95% CI) | Faithfulness R2 (95% CI) | Ans. relevance (R1/R2) | Evidence support (R1/R2) | Hallucination-flag rate R1 (95% CI) | Hallucination-flag rate R2 (95% CI) | Contradiction (R1/R2) | Errors (R1/R2) | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|
| phi3.5 | 0.9000 (0.853–0.947) | 0.9229 (0.881–0.965) | 0.669 / 0.679 | 0.796 / 0.805 | 8.3% (10/121; 4.0–14.7%) | 6.6% (8/121; 2.9–12.6%) | 7 / 5 | 1 / 1 | 1121 |
| mistral:7b | 0.8120 (0.749–0.875) | 0.8140 (0.751–0.877) | 0.713 / 0.709 | 0.816 / 0.818 | 17.4% (21/121; 11.1–25.3%) | 15.7% (19/121; 9.7–23.4%) | 15 / 16 | 0 / 0 | 334 |
| gemma2:9b | 0.7149 (0.640–0.790) | 0.7417 (0.669–0.815) | 0.502 / 0.506 | 0.789 / 0.789 | 27.3% (33/121; 19.6–36.1%) | 25.6% (31/121; 18.1–34.4%) | 26 / 22 | 0 / 0 | 1275 |
| llama3.1:8b | 0.6901 (0.612–0.768) | 0.6963 (0.619–0.773) | 0.572 / 0.554 | 0.779 / 0.778 | 30.6% (37/121; 22.5–39.6%) | 29.8% (36/121; 21.8–38.7%) | 29 / 28 | 0 / 0 | 214 |
| qwen2.5:7b | 0.6446 (0.561–0.728) | 0.6322 (0.549–0.716) | 0.519 / 0.498 | 0.625 / 0.615 | 34.7% (42/121; 26.3–43.9%) | 35.5% (43/121; 27.1–44.8%) | 39 / 40 | 0 / 0 | 201 |

Figure: `figures/fig7_generator_comparison.png`.

Faithfulness varied substantially across generators — from 0.63–0.65
(qwen2.5:7b) to 0.90–0.92 (phi3.5) — while run-to-run faithfulness differences
within each generator were small (|Δ| ≤ 0.027). This indicates generator choice
is a first-order source of quality variation under this fixed pipeline, but
these are descriptive rankings: two run-level means per model do not constitute
independent samples, and no paired between-model statistical test was applied.
The five models' hallucination-flag rates are likewise reported descriptively
(phi3.5 6.6–8.3% vs qwen2.5:7b 34.7–35.5%). The parameter count of the five
models (3.8B–9.2B) does **not** monotonically predict quality — phi3.5
(3.8B) outperformed gemma2:9b (9.2B) on faithfulness, so no parameter-scaling
claim is made.

**Table 8. Retrieval metrics — frozen retrieval layer (n = 121, k ∈ {1, 2, 3}).**

| Metric | k=1 | k=2 | k=3 |
|--------|-----|-----|-----|
| Hit@k / Recall@k | 0.2066 | 0.2645 | **0.2727** |
| Precision@k | 0.2066 | 0.1322 | 0.0909 |
| nDCG@k | 0.2066 | 0.2431 | 0.2472 |
| MRR | — | — | **0.2383** |
| MAP | — | — | **0.2383** |

Retrieval quality is reported separately because the retriever is frozen and
byte-identical for every generator — it neither improves nor degrades across
generator models. Gold-chunk hit rate (Hit@3) = 33/121 = 27.3%, exactly the
fraction of questions whose gold chunk appears in the frozen top-3 evidence.
MAP equals MRR because each question has a single gold chunk, so per-question
average precision equals reciprocal rank.

### 5.8 Experiment 5 — phi3.5 token-repeat-limit failure

The only per-question generation error across the ten Experiment-5 runs is a
repeatable failure of phi3.5 on question GLG11-001 in both runs, reported by
the Ollama runtime as `ResponseError: prediction aborted, token repeat limit
reached (status code 500)`. The identical question completed on every other
generator and on the Experiment-1 llama3.2 baseline, and the failure reproduced
across runs, so it is a stable, model-specific reproducibility issue rather than
a transient server fault. The **underlying cause is unresolved**: the failure
is reported descriptively and is not attributed to model architecture or to the
(unset) context window, for which there is no evidence.

Accounting: the errored record has `generated = False` and `faithfulness =
None`, so it is excluded from the faithfulness, answer-relevance, and
evidence-support means (denominator n = 120 for phi3.5 on those three metrics).
It is **not** counted as a verified non-hallucinated answer; rather, the
hallucination-flag denominator is the full n = 121 (the rate conservatively
treats the error as neither a hallucination nor a clean answer, and the 1/121
generation error is reported separately). Both phi3.5 runs report exactly one
error (1/121).

---

## 6. Confidence Intervals (A3)

All headline statistics carry interval estimates (Section 5; full output in
`a3_confidence_intervals/a3_confidence_intervals.{json,md}`): Exp 1 means as
95% t-CIs (Table 2), Exp 1 hallucination-flag rate as a 95% Clopper-Pearson CI
([16.0%, 31.7%]), Exp 3 rates as exact binomial CIs, and Exp 4 deltas as
t-CIs that exclude 0 on all three metrics ([−0.2836, −0.1048],
[−0.2298, −0.0512], [−0.2258, −0.1312]) — consistent with the hypothesis
tests. Bootstrap percentile CIs (B = 10,000) agree with the t-intervals in
all cases.

---

## 7. Human Calibration of the LLM Judge (A1)

**Status: COMPLETE (matched-evidence).** The primary human–judge calibration
analysis is the matched-evidence study (protocol `1.0-matched-evidence`;
artifacts in `a1_human_calibration/matched_evidence/`). The original
abridged-evidence pilot is preserved as a historical methodological pilot and
is not the primary calibration estimate.

**Calibration design.** The matched A1 uses the same 20 question IDs as the
original pilot — a seed-42, faithfulness-stratified selection (11 high /
4 partial / 5 low by automated faithfulness) — which is a non-representative
calibration sample, not a sample of the full 121-question benchmark. The human
annotator saw the exact full final evidence from Experiment 1 run-1
(untruncated, in recorded order) and the exact generated answers from the same
run, rendered neutrally with no judge scores, judge reasoning, or other
automated metadata exposed. Automated evidence-support scores (continuous
span-token coverage) were mapped to the five-level scale using the
preregistered bins [0, 0.125) → 0.0, [0.125, 0.375) → 0.25,
[0.375, 0.625) → 0.5, [0.625, 0.875) → 0.75, [0.875, 1.0] → 1.0, with
midpoint boundaries assigned to the lower bin.

**Results (n = 20).**

| Metric | Exact agreement | κ | Po | Pe | Within-1-bin |
|--------|-----------------|-----|--------|--------|--------------|
| Faithfulness | 60.0% (12/20) | degenerate — zero-variance marginal | 0.6000 | 0.5325 | 75.0% |
| Answer relevance | 55.0% (11/20) | 0.1346 | 0.5500 | 0.4800 | 75.0% |
| Evidence support | 65.0% (13/20) | degenerate — zero-variance marginal | 0.6500 | 0.6725 | 85.0% |
| Unsupported claim | 75.0% (15/20) | degenerate — zero-variance marginal | 0.7500 | 0.7500 | n/a |

**κ degeneracy.** Under the preregistered rule, Cohen's kappa is considered
degenerate when a rater marginal has ≥ 19/20 observations in one category. The
human faithfulness marginal was 19/20 at 1.0, the human support marginal 19/20
at 1.0, and the human unsupported-claim marginal 20/20 at 0; numerical kappa is
therefore not interpreted for these three metrics. Answer relevance is the only
metric with a numerically interpretable kappa (κ = 0.1346).

**Unsupported-claim result.** The human annotator flagged 0/20 answers as
containing an unsupported claim, while the automated flag marked 5/20
(TP = 0, FP = 0, TN = 15, FN = 5). This is reported as a disagreement pattern
only.

**Interpretation.** Exact agreement ranged from 55–75% depending on the metric,
and within-one-bin agreement for the three ordinal metrics ranged from 75–85%.
Human ratings were strongly concentrated at the top of the scale, which produced
degenerate kappa for three of the four metrics; the only numerically
interpretable kappa is relevance (0.1346). The sample is non-representative, a
single human annotator was used, and no inter-annotator reliability can be
estimated. This is a calibration/measurement analysis, not a validation study:
no claim is made that the automated evaluator is validated or invalidated.
---

## 8. Discussion

The experiments jointly tell a coherent story about where this architecture's
value does and does not lie.

**The framework's value is in its generation pathway, not its gate.** The
headline contrast (Exp 4) shows that, with identical initial evidence, the full
downstream generation/reliability-controlled pathway beat a minimal RAG prompt
on every metric and every paired test — yet switching the reliability gate on
and off (Exp 2) changed almost nothing on this benchmark. Both results are
consistent: the gate adds little *here* because refinement rarely changes the
evidence set on this corpus and the composite reliability score is nearly
uncorrelated with factual faithfulness (ρ = 0.06). Reporting both results
jointly is the honest characterization of the framework.

**Failures are a retrieval problem on this benchmark.** Because the failing set
coincides exactly with gold-chunk absence (0/28 vs. 0.0% failure with gold
present), the largest realistic lever for quality on GeriLit-Gold is retrieval
coverage and re-retrieval policy, not generation temperature or refusal
thresholds. This isolates the failure locus empirically rather than by
intuition.

**Care-state adaptation is real but immature.** The mechanism demonstrably
influences the answer under strict evidence identity (35% state-response), but
observed appropriateness is 0/20. This validates the *mechanism's existence*,
not its clinical skill; the appropriate use is a descriptive behavioral
finding, and the adaptation rubric itself needs refinement (Section 9).

**LLM judges are reproducible but must be interpreted with measured caution.**
Two-run exact match of 90–100% (A2) shows the judge is a stable instrument; the
matched-evidence human–judge calibration (A1, Section 7) documents moderate
human–judge exact agreement (55–75%), within-one-bin agreement of 75–85%,
degenerate kappa for three metrics due to strong top-of-scale human rating
concentration, and a disagreement pattern on unsupported claims (automated flag
5/20, human 0/20). Read together, these results are precisely why the
reliability quantification (A2) and the human calibration (A1) belong in the
paper rather than the supplement: high run-to-run stability does not by itself
establish human concordance.

**Generator choice is a first-order source of quality variance, but the judge
shares a family with one generator.** Experiment 5 shows a faithfulness span of
0.63–0.65 (qwen2.5:7b) to 0.90–0.92 (phi3.5) under an otherwise byte-identical
pipeline, and hallucination-flag rates from 6.6% (phi3.5) to 35.5%
(qwen2.5:7b). This indicates generator selection materially alters grounded
generation quality within a single RAG design — a practically important lever
for deployment. These are descriptive rankings, not significance claims, and
are subject to a measured caveat: the Llama-3.2 judge shares the Llama model
family with the llama3.1:8b generator (but not with phi3.5, mistral:7b,
qwen2.5:7b, or gemma2:9b). Judge-family bias cannot be excluded by the current
single-judge, single-annotator calibration (Section 7), and is recorded as an
open measurement risk (Section 9).

---

## 9. Limitations

1. **Benchmark scope.** 121 questions from 45 open-access gerontology PMC
   articles; results describe this corpus, not general clinical utility.
2. **Benchmark construction.** The benchmark gold set was reviewed by a single
   reviewer, precluding inter-rater reliability estimation, and a minority of
   questions retain imperfect machine-paraphrase wording.
3. **Judge-instrument validity.** Faithfulness is LLM-judged. Reproducibility is
   quantified (A2, 90–100% two-run exact match) and the matched-evidence human
   calibration (A1, n = 20) provides external agreement evidence; both must be
   read together — a stable judge is not necessarily a human-concordant one.
   The matched A1 is limited by its n = 20, stratified/non-representative
   sample, single human annotator (no inter-annotator reliability), strong
   top-of-scale human rating concentration (degenerate kappa for three of the
   four metrics), and support comparability that required preregistered binning
   (Section 7).
4. **Exp 2 is a local null, not a global refutation.** Gating may matter where
   refinement changes evidence or where the estimator discriminates better;
   on GeriLit-Gold it neither searched nor rejected.
5. **Exp 3 is behavioral.** "Appropriateness 0/20" reflects the frozen
   appropriateness rubric; changing the rubric will change the rate. Also,
   n = 20 yields wide binomial CIs (e.g., 35% ± ~22 points).
6. **Longitudinal care-state persistence.** Longitudinal care-state persistence
   and production-level state transitions were not evaluated.
7. **Refusal quality is unlabeled.** The corpus has no ground-truth refusal
   labels; refusals are counted within the hallucination-flag bucket by the
   frozen rubric (a conservative choice that inflates contradiction counts).
8. **Prompt differences in Exp 4.** The vanilla baseline is the minimal
   evidence-grounded prompt defined in the frozen protocol; it is a
   *minimal-baseline operationalization*, not a historical production prompt.
9. **Generator comparison — judge-family bias.** The Llama-3.2 evaluation judge
   shares the Llama model family with the llama3.1:8b generator (Experiment 5).
   The single-judge design cannot exclude the possibility that faithfulness for
   llama3.1:8b is measured more favorably than for out-of-family generators
   (phi3.5, mistral:7b, qwen2.5:7b, gemma2:9b). This is unresolved and is
   recorded as future work (multi-judge evaluation, Section 11).
10. **phi3.5 token-repeat-limit failure.** phi3.5 failed deterministically on
    1/121 questions (GLG11-001) in both runs (`ResponseError: 500, prediction
    aborted, token repeat limit reached`). The cause is unresolved and is not
    attributed to architecture or context window. Its three evaluation means
    are computed over n = 120, and its hallucination-flag denominator is the
    full 121 (the error is reported separately, not counted as a clean answer).
11. **Generator rankings are descriptive.** Experiment 5 reports two run-level
    means per generator. No paired between-model significance test was applied,
    and the two runs are not independent benchmark samples; therefore no
    "best model" claim is made, and no p-value is attached to the ranking.

---

## 10. Reproducibility

- Frozen benchmark GeriLit-Gold v1.1, SHA-256
  `1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72`
  (121/121 accepted, 45 PMCIDs, 116 gold chunks).
- Model + sampler: Ollama `llama3.2:latest` (3.2B Q4_K_M), temperature 0,
  top_p 0.1, top_k 10, for both generator and judge.
- Retrieval: BGE-base-en-v1.5 (dense) + BM25, 0.6/0.4 hybrid fusion,
  MiniLM-L-6-v2 CrossEncoder rerank → top-3.
- Validators: Experiment 1 63/63, Experiment 2 89/89, Experiment 3 90/90,
  Experiment 4 94/94; Task 3 reliability validator 27/27; repository test
  suite 36/36 at checkpoint `fe20c5a`.
- Two frozen runs per experiment with per-question artifacts and
  reproducibility manifests; mismatches are limited to documented LLM judge
  boundary scores (Exp 2/4) and are catalogued per question in the
  reproducibility JSONs.
- Matched A1 calibration artifacts: `a1_human_calibration/matched_evidence/
  A1_MATCHED_ANNOTATION_PROTOCOL.md`, `a1_matched_annotation_sheet.csv`,
  `a1_matched_agreement_analysis.{json,md}` (completed annotation-sheet
  SHA-256 `70506d5d2fdc631f5af77b592c6cec8b8c6be4201d24b8e46ca7b4372d0bb1c3`).
- **Experiment 5 reproducibility.** Five generators × 2 runs = 40 output files
  (`experiment5_<generator_tag>_per_question_run{1,2}.json`,
  `experiment5_<generator_tag>_summary_run{1,2}.json`,
  `experiment5_<generator_tag>_frozen_hashes_run{1,2}.json`,
  `experiment5_<generator_tag>_run_manifest_run{1,2}.json`) under
  `datasets/elderdocai_geriLit/metadata/experiment5_generator_models/`. All 12
  frozen-artifact SHA-256 hashes were verified unchanged before and after every
  run; zero `FROZEN ARTIFACT` aborts. Generator tags (llama3.1:8b, qwen2.5:7b,
  mistral:7b, gemma2:9b, phi3.5) and quantizations are recorded in
  `experiment5_ollama_show_snapshot.json`.
- All artifact directories and outputs referenced above are relative to
  `datasets/elderdocai_geriLit/metadata/paper_assembly/`.

---

## 11. Conclusion

ElderDocAI's final v1.1 stack delivers evidence-grounded answers on
the GeriLit-Gold benchmark — and, judged by the controlled evidence-identical
contrast, its grounded generation pathway is what creates the advantage over a
minimal RAG prompt. The same evaluation discipline that produced this headline
result also surfaced two honest negatives — a null gating effect and an
immature care-state adaptation — and an empirically grounded failure locus in
retrieval coverage. The generator-model comparison (Experiment 5) further
shows that, holding retrieval, prompts, and the gate constant, generator choice
alone spans a wide faithfulness range (0.63–0.92) and an 8.3%–35.5%
hallucination-flag rate — but also a single unexplained phi3.5 failure that we
would rather report than hide. We consider the reliability quantification
apparatus (two-run judge reproducibility, interval estimates, and the
pre-registered human-calibration protocol) as much a contribution as the
benchmark numbers, because an evaluation that can report its own measurement
risk is the prerequisite for trusting either positive or negative results.
Future work is to repeat the generator comparison under multiple judges and to
benchmark ElderDocAI against a broader set of RAG baselines — the present work
compares generators within one pipeline, not competing RAG architectures.
---

## Appendix A — Dataset and system configuration (Table 1)

| component | configuration |
|-----------|----------------|
| Benchmark | GeriLit-Gold v1.1 (121 questions, 45 PMCIDs) |
| Retrieval | dense BGE top-5 + BM25 top-5, hybrid 0.6/0.4, CrossEncoder rerank → top-3 |
| Generator / judge (Exp 1–4) | llama3.2:latest, temperature 0, top_p 0.1, top_k 10 |
| Generator (Exp 5) | llama3.1:8b, qwen2.5:7b, mistral:7b, gemma2:9b, phi3.5 (same sampler options, judge constant) |
| Judge (all experiments) | llama3.2:latest, temperature 0, top_p 0.1, top_k 10 |
| Reliability gate | ACCEPT ≥ 0.80, REFINE ≥ 0.65, RE-RETRIEVE ≥ 0.45, REJECT < 0.45 |
| Judge scale | 0/0.25/0.5/0.75/1.0 (faithfulness, answer relevance; evidence support deterministic) |

*Configuration note: Experiments 1–4 were run with
`ELDERDOCAI_RETRIEVAL_BACKEND=geri_lit`. The shipped default when the variable
is unset is the legacy backend (1,143-chunk caregiver-manual corpus), which was
not evaluated in these experiments; see Section 3.1.*

## Appendix B — Reliability-gate decision distribution (Exp 1)

| decision | count | percentage |
|----------|------:|-----------:|
| ACCEPT | 37 | 30.6% |
| REFINE | 84 | 69.4% |
| RE-RETRIEVE | 0 | 0.0% |
| REJECT | 0 | 0.0% |

Figure: `figures/fig1_decision_distribution.png`. The gate exercised only its
ACCEPT/REFINE branches on this benchmark (no re-retrieval, no rejection).
Mean final reliability 0.788; refinement cases 84.

## Appendix C — Figure list

| Figure | File | Content |
|--------|------|---------|
| Fig 1 | `figures/fig1_decision_distribution.png` | Gate decision distribution (Exp 1) |
| Fig 2 | `figures/fig2_generation_quality.png` | Exp 1 metric means with 95% CIs |
| Fig 3 | `figures/fig3_gating_contrast.png` | Exp 2 gate ON vs OFF contrast |
| Fig 4 | `figures/fig4_vanilla_contrast.png` | Exp 4 FULL vs Vanilla-RAG (headline) |
| Fig 5 | `figures/fig5_care_state_contrast.png` | Exp 3 care-state contrast |
| Fig 6 | `figures/fig6_judge_reliability.png` | Judge run1-vs-run2 exact match |
| Fig 7 | `figures/fig7_generator_comparison.png` | Exp 5 generator faithfulness/relevance contrast (5 models) |

All figure-data CSVs: `fig_data/*.csv`; figure code: `c7_c8_figures.py`;
narrative tables (verbatim frozen numbers): `c7_c8_narrative_tables.{md,json}`.

## Appendix D — Pipeline validation status (deterministic core)

System tests: **36/36 PASS** (repository, checkpoint `fe20c5a`).
Task-3 reliability validator: **27/27 PASS**.
Experiment validators: Exp 1 63/63, Exp 2 89/89, Exp 3 90/90, Exp 4 94/94.
Care-state deterministic layer: pure rule-based estimator verified against the
freeze checkpoint (default path byte-identical to the freeze commit); safety
boundary of the real mechanism: care-state output is response adaptation only,
explicitly not evidence, care-state labels must not become medical claims, and
assistance planning is constrained to response structure/pacing and grounded
interaction, with no-diagnosis / no-risk-prediction style constraints enforced
by the generation system prompt.