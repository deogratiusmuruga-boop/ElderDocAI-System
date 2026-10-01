# A2 - LLM Judge Reliability (run1 vs run2)

Per-metric exact-match agreement and Cohen's kappa across the two frozen runs per experiment. Judge = llama3.2:latest (temperature 0, top_p 0.1, top_k 10).

## Exp 1 - generation quality

| metric | n | exact match % | Cohen's kappa | median |A-B| | max |A-B| |
|--------|---|---------------|---------------|------------|-----------|
| faithfulness | 121 | 100.00 | 1.0 | 0.0000 | 0.0000 |
| answer_relevance | 121 | 100.00 | 1.0 | 0.0000 | 0.0000 |
| evidence_support | 121 | 100.00 | 1.0 | 0.0000 | 0.0000 |

## Exp 2 - gating effectiveness

| metric | n | exact match % | Cohen's kappa | median |A-B| | max |A-B| |
|--------|---|---------------|---------------|------------|-----------|
| gate_on_faithfulness | 121 | 100.00 | 1.0 | 0.0000 | 0.0000 |
| gate_on_answer_relevance | 121 | 99.17 | 0.9848 | 0.0000 | 0.2500 |
| gate_on_evidence_support | 121 | 100.00 | 1.0 | 0.0000 | 0.0000 |
| gate_off_faithfulness | 121 | 100.00 | 1.0 | 0.0000 | 0.0000 |
| gate_off_answer_relevance | 121 | 98.35 | 0.9695 | 0.0000 | 0.2500 |
| gate_off_evidence_support | 121 | 100.00 | 1.0 | 0.0000 | 0.0000 |

## Exp 3 - care-state adaptation

| metric | n | exact match % | Cohen's kappa | median |A-B| | max |A-B| |
|--------|---|---------------|---------------|------------|-----------|
| A_faithfulness | 20 | 95.00 | 0.8953 | 0.0000 | 0.5000 |
| A_answer_relevance | 20 | 90.00 | 0.8214 | 0.0000 | 0.2500 |
| A_evidence_support | 20 | 100.00 | 1.0 | 0.0000 | 0.0000 |
| B_faithfulness | 20 | 100.00 | 1.0 | 0.0000 | 0.0000 |
| B_answer_relevance | 20 | 95.00 | 0.9029 | 0.0000 | 0.2500 |
| B_evidence_support | 20 | 100.00 | 1.0 | 0.0000 | 0.0000 |
| appropriateness | 20 | 95.00 | 0.0 | 0.0000 | 1.0000 |

For the Exp 3 `appropriateness` metric, the near-zero kappa with 95% exact
match is the classic *kappa paradox*: run1 assigned the same value (0) to
all 20 cases, so its zero-variance marginal makes chance agreement (pe)
approximately equal observed agreement (po), compressing kappa to ~0. The
exact-match figure (95%) is the appropriate agreement statistic here; kappa
should be interpreted with caution for this highly skewed binary metric.
## Exp 4 - vanilla RAG baseline

| metric | n | exact match % | Cohen's kappa | median |A-B| | max |A-B| |
|--------|---|---------------|---------------|------------|-----------|
| full_faithfulness | 121 | 94.21 | 0.8853 | 0.0000 | 1.0000 |
| full_answer_relevance | 121 | 96.69 | 0.9371 | 0.0000 | 0.2500 |
| full_evidence_support | 121 | 90.08 | 0.8784 | 0.0000 | 0.2571 |
| vanilla_faithfulness | 121 | 100.00 | 1.0 | 0.0000 | 0.0000 |
| vanilla_answer_relevance | 121 | 99.17 | 0.9875 | 0.0000 | 0.5000 |
| vanilla_evidence_support | 121 | 100.00 | 1.0 | 0.0000 | 0.0000 |
