# C7/C8 - Narrative tables and figure data (frozen numbers)

All values below are lifted verbatim from the frozen experiment summaries (validation status and hashes recorded in each experiment report). See the per-experiment reports for full statistics.

## Table 1 - Benchmark and system configuration

| component | configuration |
|-----------|----------------|
| Benchmark | GeriLit-Gold v1.1 (121 questions, 45 PMCIDs) |
| Retrieval | dense BGE top-5 + BM25 top-5, hybrid 0.6/0.4, CrossEncoder rerank -> top-3 |
| Generator / judge | llama3.2:latest, temperature 0, top_p 0.1, top_k 10 |
| Reliability gate | ACCEPT>=0.80, REFINE>=0.65, RE-RETRIEVE>=0.45, REJECT<0.45 |
| Judge scale | 0/0.25/0.5/0.75/1.0 (faithfulness, answer relevance; evidence support deterministic) |

## Table 2 - Experiment 1: generation quality (n=121)

| metric | mean | sd | 95% CI |
|--------|------|----|--------|
| Faithfulness | 0.7541 | 0.3966 | [0.6835, 0.8248] |
| Answer relevance | 0.6715 | 0.4496 | [0.5914, 0.7516] |
| Evidence support | 0.8528 | 0.2408 | [0.8099, 0.8957] |

- Unsupported-claim rate (faithfulness <= 0.5): 23.14% (28/121; 95% binomial CI [15.96%, 31.68%])
- Contradiction rate (faithfulness == 0): 18.18%

## Table 3 - Experiment 2: reliability gate ON vs OFF (n=121; delta = ON - OFF)

| metric | ON mean | OFF mean | mean delta | t p | Wilcoxon p | bootstrap 95% CI | ON=OFF |
|--------|---------|----------|------------|-----|-----------|-----------------|--------|
| Faithfulness | 0.7541 | 0.7810 | -0.0269 | 0.0685 | 0.0684 | [-0.0579, -0.0021] | 115 |
| Answer relevance | 0.6715 | 0.6674 | 0.0041 | 0.7404 | 0.5887 | [-0.0207, 0.0289] | 115 |
| Evidence support | 0.8528 | 0.8543 | -0.0015 | 0.5351 | 0.4236 | [-0.0061, 0.0031] | 110 |

No metric met the pre-registered joint significance rule (paired t-test p<0.05 AND Wilcoxon p<0.05).

## Table 4 - Experiment 3: care-state adaptation (n=20 pairs; A=LOW_ACTIVITY, B=HIGH_ACTIVITY)

| metric | A mean | B mean | mean delta (B-A) | B>A | B<A |
|--------|--------|--------|------------------|-----|-----|
| Faithfulness | 0.8125 | 0.8500 | 0.0375 | 2 | 0 |
| Answer relevance | 0.6250 | 0.6750 | 0.0500 | 2 | 0 |
| Evidence support | 0.8221 | 0.8193 | -0.0028 | 3 | 3 |

- State-response rate: 35.0% (7/20; 95% binomial CI [15.4%, 59.2%])
- Evidence-identical rate: 100.0%
- Appropriateness: 0/20 scored appropriate (95% binomial CI [0%, 16.8%])

## Table 5 - Experiment 4: FULL vs Vanilla-RAG baseline (n=121; delta = VANILLA - FULL)

| metric | FULL mean | VANILLA mean | mean delta | t p | Wilcoxon p | supported | V<F |
|--------|-----------|--------------|------------|-----|-----------|-----------|-----|
| Faithfulness | 0.7645 | 0.5702 | -0.1942 | 0.0000 | 0.0001 | True | 38 |
| Answer relevance | 0.6880 | 0.5475 | -0.1405 | 0.0025 | 0.0028 | True | 42 |
| Evidence support | 0.8550 | 0.6765 | -0.1785 | 0.0000 | 0.0000 | True | 99 |

Evidence identity control held on all 121 pairs (identical retrieved evidence in both conditions).

## Table 6 - LLM judge reproducibility (run1 vs run2, A2)

| experiment | n | exact match min% | exact match max% | kappa min | kappa max |
|------------|---|------------------|------------------|-----------|-----------|
| exp1 | 121 | 100.0 | 100.0 | 1.000 | 1.000 |
| exp2 | 121 | 98.3 | 100.0 | 0.970 | 1.000 |
| exp3 | 20 | 90.0 | 100.0 | 0.000 | 1.000 |
| exp4 | 121 | 90.1 | 100.0 | 0.878 | 1.000 |

Appropriateness kappa in Exp 3 is suppressed (kappa paradox with an all-zero run1 marginal; exact match 95%).
