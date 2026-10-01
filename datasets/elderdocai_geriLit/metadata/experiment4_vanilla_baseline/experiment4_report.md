# Experiment 4 - Vanilla-RAG Baseline on GeriLit-Gold v1.1

**Status:** PASS

## 1. Design

Controlled paired comparison: the FULL ElderDocAI generation pathway (reliability gate, refinement, ElderDocAI grounded prompt) vs a minimal Vanilla-RAG generation (plain evidence-grounded prompt, no framework metadata). Both conditions receive the SAME question and the SAME initial frozen retrieval evidence (GeriLit v1.1, n=121, SHA-256 `1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72`).

## 2. Vanilla-RAG prompt (frozen)

No plain RAG prompt predating the framework exists in the repository (all historical prompts are the ElderDocAI grounded prompt; ablation A5 strips only the reliability section). A minimum baseline prompt was defined per the experiment specification:

```
Answer the user's question using only the provided evidence.

If the evidence does not contain enough information to answer the question,
state that the evidence is insufficient rather than inventing information.

Retrieved evidence:
[EVIDENCE BLOCKS]

Question:
[USER QUESTION]
```

Prompt SHA-256 recorded in `experiment4_run_manifest.json`.

## 3. Paired primary statistics (VANILLA - FULL deltas)

| Metric | VAN mean | FULL mean | mean d | median d | std d | V>F | V=F | V<F | t p | Wilcoxon p | supported |
|--------|----------|-----------|--------|----------|-------|-----|-----|-----|-------|------------|-----------|
| faithfulness | 0.5702 | 0.7645 | -0.1942 | 0.0000 | 0.5015 | 13 | 70 | 38 | 0.0000 | 0.0001 | True |
| answer_relevance | 0.5475 | 0.6880 | -0.1405 | 0.0000 | 0.5009 | 15 | 64 | 42 | 0.0025 | 0.0028 | True |
| evidence_support | 0.6765 | 0.8550 | -0.1785 | -0.2308 | 0.2655 | 22 | 0 | 99 | 0.0000 | 0.0000 | True |

## 4. Condition-level statistics

| Metric | FULL mean | VANILLA mean | FULL median | VANILLA median |
|--------|-----------|--------------|-------------|---------------|
| faithfulness | 0.7645 | 0.5702 | 1.0000 | 1.0000 |
| answer_relevance | 0.6880 | 0.5475 | 1.0000 | 0.7500 |
| evidence_support | 0.8550 | 0.6765 | 0.9474 | 0.6786 |

## 5. Evidence identity control

- evidence identity (VANILLA IDs == FULL initial IDs) held for **True** (failures: [])
- FULL gate: ACCEPT 37, REFINE 84, RE-RETRIEVE 0, REJECT 0; refinement cases 84
- generation permitted: FULL 121/121, VANILLA 121/121


## 6. Refusals and gold evidence

- refusals: FULL 0, VANILLA 0
- gold chunk in evidence: FULL 33/121, VANILLA 33/121

## 7. Topic-level results (descriptive, not ranked)

| Topic | n | f-delta mean | r-delta mean | s-delta mean |
|-------|---|--------------|--------------|--------------|
| C01 | 11 | -0.3636 | -0.2045 | -0.3312 |
| C02 | 15 | -0.1500 | -0.0833 | -0.1052 |
| C03 | 13 | 0.0192 | -0.1154 | -0.0603 |
| C04 | 13 | -0.2115 | -0.2692 | -0.1697 |
| C05 | 14 | -0.2679 | -0.1429 | -0.1321 |
| C06 | 16 | -0.2500 | -0.2969 | -0.1914 |
| C07 | 14 | -0.2857 | 0.0357 | -0.2011 |
| C08 | 9 | -0.3611 | 0.0000 | -0.2751 |
| C09 | 10 | 0.1250 | -0.0750 | -0.2790 |
| C10 | 6 | -0.1667 | -0.2500 | -0.0654 |

## 8. Reproducibility

- run 1 signature: `77812cf0e5330f56f7bd8d4a2f282bcd16e974c17f639415d4ea27ace00dd4fc`
- run 2 signature: `ccc466fef49f6ae7cfab7486b2a7ac3cc126859df9bfcab5dddc170cee2f0132`
- system-fields deterministic: **False**
- full signature identical: **False**
- mismatched pair IDs: ['GLG11-057', 'GLG11-060', 'GLG11-061', 'GLG11-063', 'GLG11-068', 'GLG11-074', 'GLG11-077', 'GLG11-078', 'GLG11-086', 'GLG11-087', 'GLG11-092', 'GLG11-093', 'GLG11-094', 'GLG11-095', 'GLG11-096', 'GLG11-103', 'GLG11-112']

## 9. Validation

- checks passed: 94/94

## 10. Scientific interpretation (descriptive)

- This experiment compares generation pathways under identical evidence; it does not measure retrieval quality, clinical utility, or generalizability.
- Differences are reported descriptively. A metric is flagged statistically supported only when BOTH the paired t-test and the Wilcoxon signed-rank p-values are below 0.05 (pre-registered).
