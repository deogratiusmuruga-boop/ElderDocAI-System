# ElderDocAI Ablation Study Report

> Synthetic longitudinal clinical records (Synthea-derived) and 96 gold QA questions over the ElderDocAI knowledge base.
> **Scope:** architectural / internal-consistency evaluation. This is NOT a clinical validation study and implies no clinical benefit.

## 1. Experimental Objective

Quantify the contribution of each ElderDocAI component across retrieval / evidence grounding, answer quality, reliability assessment, dynamic care-state awareness, and adaptive assistance behavior, via controlled condition-by-condition comparison.

## 2. Experimental Design

A0 (full system) is the reference. Each ablation removes exactly one component while holding inputs constant: same 96 gold questions, same KB index / embedding model, same generation model (llama3.2, temperature 0), same reliability config, same judge prompt/rubric, same patient profile for every condition attaching a patient, and the same 9,723 windows for the care-state/adaptive analysis.

## 3. Conditions

| Cond | Name | Retrieval | Rerank | Reliability gate | Dynamic care state | Adaptive assistance |
|---|---|---|---|---|---|---|
| A0 | Full ElderDocAI | hybrid_full | True | True | True | adaptive |
| A1 | No Dynamic Care State | hybrid_full | True | True | False | none |
| A2 | No Adaptive Assistance | hybrid_full | True | True | True | fixed_default |
| A3 | Dense Retrieval Only | dense_only | False | True | True | adaptive |
| A4 | Hybrid Without Reranking | hybrid_no_rerank | False | True | True | adaptive |
| A5 | No Reliability Gate | hybrid_full | True | False | True | adaptive |
| A6 | Static Care Profile Only | hybrid_full | True | True | False | none |

## 4. Dataset and Number of Cases

- 96 gold QA questions per condition, 9,723 synthetic patient x year windows (178 synthetic patients) for the adaptive-assistance analysis.

## 5. Metrics

- Retrieval: source accuracy, evidence/chunk recall@3, supporting-span support, precision@3, MRR(source).
- Answer: RQ2-judge faithfulness and relevance, successful-generation rate.
- Reliability: mean/min reliability, decision distribution (same config).
- Dynamic/adaptive: assistance-mode diversity, state-driven and transition-driven adaptation rates, exact agreement with the full system.

## 6. Per-Condition Results

### 6.1 Main table

|Condition|Retrieval Accuracy (%)|Evidence Recall@3|Faithfulness|Answer Relevance|Reliability|Adaptive coverage|
|--:|--:|--:|--:|--:|--:|--:|
|Full ElderDocAI|100.0|1.0|0.9089|0.8974|0.9059|window-level (Sec. 7)|
|No Dynamic Care State|100.0|1.0|0.9141|0.8958|0.9059|window-level (Sec. 7)|
|No Adaptive Assistance|100.0|1.0|0.9193|0.9115|0.9059|window-level (Sec. 7)|
|Dense Retrieval Only|98.96|0.9167|0.8906|0.9453|0.899|window-level (Sec. 7)|
|Hybrid Without Reranking|98.96|0.9062|0.9036|0.8984|0.9025|window-level (Sec. 7)|
|No Reliability Gate|100.0|1.0|0.9349|0.9219|0.9059|window-level (Sec. 7)|
|Static Care Profile Only|100.0|1.0|0.9141|0.8958|0.9059|window-level (Sec. 7)|

*Adaptive coverage is a window-level property (Sec. 7), not a question-level metric.*

### 6.2 Per-condition aggregates

**A0 - Full ElderDocAI**

- retrieval accuracy 100.0% | recall@3 1.0 | precision@3 0.3333 | MRR 0.9722 | span support 100.0%
- faithfulness 0.9089 | relevance 0.8974 | successful gen 100.0%
- reliability mean 0.9059 | decision dist {'ACCEPT': 88, 'REFINE': 8}
- runtime 142.32s | errors 0

**A1 - No Dynamic Care State**

- retrieval accuracy 100.0% | recall@3 1.0 | precision@3 0.3333 | MRR 0.9722 | span support 100.0%
- faithfulness 0.9141 | relevance 0.8958 | successful gen 100.0%
- reliability mean 0.9059 | decision dist {'ACCEPT': 88, 'REFINE': 8}
- runtime 131.44s | errors 0

**A2 - No Adaptive Assistance**

- retrieval accuracy 100.0% | recall@3 1.0 | precision@3 0.3333 | MRR 0.9722 | span support 100.0%
- faithfulness 0.9193 | relevance 0.9115 | successful gen 100.0%
- reliability mean 0.9059 | decision dist {'ACCEPT': 88, 'REFINE': 8}
- runtime 136.72s | errors 0

**A3 - Dense Retrieval Only**

- retrieval accuracy 98.96% | recall@3 0.9167 | precision@3 0.3055 | MRR 0.9653 | span support 94.79%
- faithfulness 0.8906 | relevance 0.9453 | successful gen 100.0%
- reliability mean 0.899 | decision dist {'ACCEPT': 85, 'REFINE': 11}
- runtime 136.55s | errors 0

**A4 - Hybrid Without Reranking**

- retrieval accuracy 98.96% | recall@3 0.9062 | precision@3 0.3021 | MRR 0.9549 | span support 93.75%
- faithfulness 0.9036 | relevance 0.8984 | successful gen 100.0%
- reliability mean 0.9025 | decision dist {'ACCEPT': 87, 'REFINE': 9}
- runtime 135.01s | errors 0

**A5 - No Reliability Gate**

- retrieval accuracy 100.0% | recall@3 1.0 | precision@3 0.3333 | MRR 0.9722 | span support 100.0%
- faithfulness 0.9349 | relevance 0.9219 | successful gen 100.0%
- reliability mean 0.9059 | decision dist {'ACCEPT': 88, 'REFINE': 8}
- runtime 135.04s | errors 0

**A6 - Static Care Profile Only**

- retrieval accuracy 100.0% | recall@3 1.0 | precision@3 0.3333 | MRR 0.9722 | span support 100.0%
- faithfulness 0.9141 | relevance 0.8958 | successful gen 100.0%
- reliability mean 0.9059 | decision dist {'ACCEPT': 88, 'REFINE': 8}
- runtime 131.44s | errors 0

## 7. Adaptive-Assistance Comparison (9,723 windows)

|Policy|Distinct modes|Mode distribution|Priority distribution|
|--:|--:|--:|--:|
|full|9|{"WAIT_FOR_DATA": 7445, "CONTEXTUAL_SUPPORT": 476, "ENHANCED_SUPPORT": 452, "ADAPTIVE_ESCALATION": 446, "ADAPTIVE_DEESCALATION": 302, "LIGHT_SUPPORT": 202, "INITIAL_CONTEXT": 178, "MONITORING_SUPPORT": 121, "FOLLOW_UP_SUPPORT": 101}|{"LOW": 7807, "HIGH": 969, "MODERATE": 947}|
|state_only_A2_proxy|5|{"WAIT_FOR_DATA": 6356, "LIGHT_SUPPORT": 1271, "CONTEXTUAL_SUPPORT": 1257, "ENHANCED_SUPPORT": 838, "MAINTENANCE_SUPPORT": 1}|{"LOW": 7628, "MODERATE": 1257, "HIGH": 838}|
|static_patient_A6_proxy|3|{"CONTEXTUAL_SUPPORT": 4313, "ENHANCED_SUPPORT": 4093, "MAINTENANCE_SUPPORT": 1317}|{"MODERATE": 4313, "HIGH": 4093, "LOW": 1317}|
|fixed_default_A1_proxy|1|{"GENERAL_SUPPORT": 9723}|{"LOW": 9723}|

- state-sensitive adaptation rate: **100.0%** (9723 windows changed vs fixed default)
- transition-sensitive adaptation rate: **23.01%** (2237 windows changed vs state-only policy)
- exact agreement full vs state-only (A2 proxy): **76.99%**
- exact agreement full vs static-patient (A6 proxy): **6.14%**
- exact agreement full vs fixed-default (A1 proxy): **0.0%**
- priority-shift rate vs state-only: **5.17%**

Transition sensitivity breakdown:

|Transition|Windows|Mode differs from state-only|%|
|--:|--:|--:|--:|
|INITIAL_STATE|178|178|100.0|
|NO_CHANGE|1130|0|0.0|
|STATE_ESCALATION|446|446|100.0|
|STATE_DEESCALATION|302|302|100.0|
|INCREASING_ACTIVITY|121|121|100.0|
|DECREASING_ACTIVITY|101|101|100.0|
|GAP|7445|1089|14.63|

## 8. Component-Wise Comparison (Delta = Ablated - Full)

|Ablation|Faithfulness (full -> abl)|Relevance (full -> abl)|Reliability (full -> abl)|Retrieval Accuracy (full -> abl)|Interpretation|
|--:|--:|--:|--:|--:|--:|
|No Dynamic Care State|0.9089 -> 0.9141|0.8974 -> 0.8958|0.9059 -> 0.9059|100.0% -> 100.0%|Static profile + plain RAG. Removes dynamic care state, transitions, and state-derived adaptive assistance.|
|No Adaptive Assistance|0.9089 -> 0.9193|0.8974 -> 0.9115|0.9059 -> 0.9059|100.0% -> 100.0%|Dynamic care state is detected, but assistance is a FIXED default strategy (GENERAL_SUPPORT/LOW).|
|Dense Retrieval Only|0.9089 -> 0.8906|0.8974 -> 0.9453|0.9059 -> 0.899|100.0% -> 98.96%|FAISS dense retrieval only. No BM25, no score fusion, no CrossEncoder reranking.|
|Hybrid Without Reranking|0.9089 -> 0.9036|0.8974 -> 0.8984|0.9059 -> 0.9025|100.0% -> 98.96%|Dense + BM25 hybrid fusion but no CrossEncoder reranking.|
|No Reliability Gate|0.9089 -> 0.9349|0.8974 -> 0.9219|0.9059 -> 0.9059|100.0% -> 100.0%|Reliability computed/recorded, but the reliability+decision section is absent from the prompt; generation always proceeds.|
|Static Care Profile Only|0.9089 -> 0.9141|0.8974 -> 0.8958|0.9059 -> 0.9059|100.0% -> 100.0%|Conventional personalized RAG baseline: static profile + retrieval + grounded answer. Prompt-identical to A1 by construction.|

## 9. Reliability Comparison (A0 vs A5)

- **A0**: avg 0.9059, min 0.6984, decision dist {'ACCEPT': 88, 'REFINE': 8, 'RE-RETRIEVE': 0, 'REJECT': 0}
- **A5**: avg 0.9059, min 0.6984, decision dist {'ACCEPT': 88, 'REFINE': 8, 'RE-RETRIEVE': 0, 'REJECT': 0}
- Reliability calculation and retrieved evidence are identical in A0/A5; only the prompt differs (A5 removes the reliability/decision block).
- On these 96 in-scope gold questions, the A0/A5 decision distributions are identical ({'ACCEPT': 88, 'REFINE': 8}); only the prompt differs (A5 removes the reliability/decision block).

## 10. Retrieval Comparison

- **A0**: acc 100.0% | recall@3 1.0 | span 100.0% | MRR 0.9722
- **A1**: acc 100.0% | recall@3 1.0 | span 100.0% | MRR 0.9722
- **A2**: acc 100.0% | recall@3 1.0 | span 100.0% | MRR 0.9722
- **A3**: acc 98.96% | recall@3 0.9167 | span 94.79% | MRR 0.9653
- **A4**: acc 98.96% | recall@3 0.9062 | span 93.75% | MRR 0.9549
- **A5**: acc 100.0% | recall@3 1.0 | span 100.0% | MRR 0.9722
- **A6**: acc 100.0% | recall@3 1.0 | span 100.0% | MRR 0.9722

## 11. Statistical / Significance Analysis

Paired deltas (ablated - A0) per question, with a paired Wilcoxon signed-rank test where justified. n is small (<=16); results are descriptive and must not be read as proof of significance.

### No Dynamic Care State

|Metric|n paired|Delta mean|Delta median|Test|p-value|Note|
|--:|--:|--:|--:|--:|--:|--:|
|Faithfulness|96|0.0052|0.0|wilcoxon_signed_rank|0.64695|descriptive only; small n; dz undefined/inflated when sd=0|
|Answer Relevance|95|0.0053|0.0|wilcoxon_signed_rank|0.97015|descriptive only; small n; dz undefined/inflated when sd=0|
|Reliability|0|n/a|n/a|not_applicable|n/a|no paired values (judge parse failures or missing rows)|

### No Adaptive Assistance

|Metric|n paired|Delta mean|Delta median|Test|p-value|Note|
|--:|--:|--:|--:|--:|--:|--:|
|Faithfulness|96|0.0104|0.0|wilcoxon_signed_rank|0.546494|descriptive only; small n; dz undefined/inflated when sd=0|
|Answer Relevance|95|0.0211|0.0|wilcoxon_signed_rank|0.519365|descriptive only; small n; dz undefined/inflated when sd=0|
|Reliability|0|n/a|n/a|not_applicable|n/a|no paired values (judge parse failures or missing rows)|

### Dense Retrieval Only

|Metric|n paired|Delta mean|Delta median|Test|p-value|Note|
|--:|--:|--:|--:|--:|--:|--:|
|Faithfulness|96|-0.0182|0.0|wilcoxon_signed_rank|0.672957|descriptive only; small n; dz undefined/inflated when sd=0|
|Answer Relevance|95|0.0474|0.0|wilcoxon_signed_rank|0.070949|descriptive only; small n; dz undefined/inflated when sd=0|
|Reliability|0|n/a|n/a|not_applicable|n/a|no paired values (judge parse failures or missing rows)|

### Hybrid Without Reranking

|Metric|n paired|Delta mean|Delta median|Test|p-value|Note|
|--:|--:|--:|--:|--:|--:|--:|
|Faithfulness|96|-0.0052|0.0|wilcoxon_signed_rank|0.872447|descriptive only; small n; dz undefined/inflated when sd=0|
|Answer Relevance|95|0.0079|0.0|wilcoxon_signed_rank|0.853936|descriptive only; small n; dz undefined/inflated when sd=0|
|Reliability|0|n/a|n/a|not_applicable|n/a|no paired values (judge parse failures or missing rows)|

### No Reliability Gate

|Metric|n paired|Delta mean|Delta median|Test|p-value|Note|
|--:|--:|--:|--:|--:|--:|--:|
|Faithfulness|96|0.026|0.0|wilcoxon_signed_rank|0.095581|descriptive only; small n; dz undefined/inflated when sd=0|
|Answer Relevance|95|0.0263|0.0|wilcoxon_signed_rank|0.216873|descriptive only; small n; dz undefined/inflated when sd=0|
|Reliability|0|n/a|n/a|not_applicable|n/a|no paired values (judge parse failures or missing rows)|

### Static Care Profile Only

|Metric|n paired|Delta mean|Delta median|Test|p-value|Note|
|--:|--:|--:|--:|--:|--:|--:|
|Faithfulness|96|0.0052|0.0|wilcoxon_signed_rank|0.64695|descriptive only; small n; dz undefined/inflated when sd=0|
|Answer Relevance|95|0.0053|0.0|wilcoxon_signed_rank|0.97015|descriptive only; small n; dz undefined/inflated when sd=0|
|Reliability|0|n/a|n/a|not_applicable|n/a|no paired values (judge parse failures or missing rows)|

## 12. Figures

- `fig1_retrieval_ablation.png`
- `fig2_answer_quality_ablation.png`
- `fig3_reliability_ablation.png`
- `fig4_adaptive_assistance_ablation.png`

## 13. Reproducibility Information

- Console log saved to `C:\Users\chosun\Documents\ElderDocAI-System\data\evaluation_results\ablation_gold96\ablation_console_output.txt` (1173 lines).
- Generation model llama3.2 (temperature 0 / top_p 0.1 / top_k 10); judge llama3.2.
- Reproducible from `scripts/evaluate_ablation.py` with the same indexed KB and gold-QA set.

## 14. Limitations

- 96 gold questions is a curated, in-scope set; most are straightforward and high-reliability, creating ceiling effects that limit retrieval and reliability-gate differentiation.
- The records are SYNTHETIC (Synthea-derived). No clinical validity is claimed.
- A1 and A6 share the same prompt by construction and reuse the same generation outputs; they differ only in the reported label.
- A2's window-level signal is a state-only proxy of the fixed-default strategy; the production system ships no non-adaptive fallback to ablate directly.
- A5 removes the reliability section from the prompt; the reliability calculation itself is unchanged and still reported.

## 15. Interpretation

Retrieval/reranking/reliability ablations produce high and internally consistent retrieval and answer-quality scores on the 96 in-scope gold questions, so retrieval/reliability differences are small (ceiling effects) and cannot be over-interpreted. The window-level analysis demonstrates that adaptive assistance is strongly state- and transition-dependent: a fixed-default or static-patient policy matches the full system's assistance decisions on a small fraction of the 9,723 windows, whereas a per-window state-only policy matches a large share. This shows that the dynamic care-state and transition layers meaningfully change assistance behavior, while evidence grounding and reliable generation remain stable when those layers are removed. These are architectural / internal-consistency results on synthetic data, not clinical evidence.
