# ElderDocAI - External LLM Generation Baselines Report

> Synthetic longitudinal records (Synthea-derived) + 96 gold QA questions.
> Every condition uses the SAME retrieval (hybrid top-3), same reliability config, same grounded-prompt templates, and the SAME judge model/rubric (llama3.2). ONLY the generation model differs between the reference and the baseline conditions.

## 1. Design

|Condition|Generation model|Context|Reference ablation cond.|
|--:|--:|--:|--:|
|llama3.2:latest|FULL|llama3.2:latest|FULL|A0|
|llama3.2:latest|PLAIN|llama3.2:latest|PLAIN|A1|
|mistral:latest|FULL|mistral:latest|FULL|A0|
|mistral:latest|PLAIN|mistral:latest|PLAIN|A1|
|qwen3:latest|FULL|qwen3:latest|FULL|A0|
|qwen3:latest|PLAIN|qwen3:latest|PLAIN|A1|
|gemma2:9b|FULL|gemma2:9b|FULL|A0|
|gemma2:9b|PLAIN|gemma2:9b|PLAIN|A1|
|phi3:mini|FULL|phi3:mini|FULL|A0|
|phi3:mini|PLAIN|phi3:mini|PLAIN|A1|

- FULL = complete ElderDocAI grounded prompt (dynamic care state + adaptive assistance plan + reliability gate); byte-identical to ablation A0.
- PLAIN = static profile + plain RAG grounded prompt; byte-identical to ablation A1.
- Reference generator (`llama3.2:latest`) is not re-run: FULL/PLAIN records are reused from ablation conditions A0/A1 (verified prompt-identical at build time via SHA-256). Baseline models are run fresh.

## 2. Per-condition results

|Condition|Faithfulness|Relevance|Reliability|Gen s (mean)|Judge s (mean)|Errors|Judge parse failures|
|--:|--:|--:|--:|--:|--:|--:|--:|
|llama3.2:latest|FULL|0.9089|0.8974|0.9059|0.5968|0.8518|0|1|
|llama3.2:latest|PLAIN|0.9141|0.8958|0.9059|0.5528|0.7942|0|0|
|mistral:latest|FULL|0.9323|0.9661|0.9059|1.5327|0.8351|0|0|
|mistral:latest|PLAIN|0.9453|0.9427|0.9059|1.0609|0.8191|0|0|
|qwen3:latest|FULL|0.9271|0.9714|0.9059|14.6625|0.8828|0|-4|
|qwen3:latest|PLAIN|0.9141|0.9789|0.9059|8.9254|0.8005|0|-1|
|gemma2:9b|FULL|0.8947|0.9342|0.9059|2.1676|0.905|1|2|
|gemma2:9b|PLAIN|0.9036|0.9688|0.9059|1.115|0.7884|0|0|
|phi3:mini|FULL|0.9167|0.9427|0.9059|1.1684|0.8786|0|0|
|phi3:mini|PLAIN|0.8776|0.9349|0.9059|0.8395|0.7884|0|0|

## 3. Delta vs reference FULL (llama3.2 FULL)

|Condition|Faithfulness delta|Relevance delta|
|--:|--:|--:|
|llama3.2:latest|PLAIN|0.0052|-0.0016|
|mistral:latest|FULL|0.0234|0.0687|
|mistral:latest|PLAIN|0.0364|0.0453|
|qwen3:latest|FULL|0.0182|0.074|
|qwen3:latest|PLAIN|0.0052|0.0815|
|gemma2:9b|FULL|-0.0142|0.0368|
|gemma2:9b|PLAIN|-0.0053|0.0714|
|phi3:mini|FULL|0.0078|0.0453|
|phi3:mini|PLAIN|-0.0313|0.0375|

## 4. Reliability / grounding note

Because all conditions use the same hybrid retrieval (top-3, reranked) and the same reliability configuration, the retrieved evidence, grounding metrics, and reliability/decision scores are identical across every condition (any difference would indicate a data-integrity problem). Reliability decision distribution: ["{'ACCEPT': 88, 'REFINE': 8}"].

## 5. Statistical analysis (paired Wilcoxon vs reference FULL)

## context_effect|gemma2:9b (context effect: PLAIN minus FULL)

|Metric|n paired|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|95|0.0105|0.0|wilcoxon_signed_rank|0.670339|0.0496|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|95|0.0342|0.0|wilcoxon_signed_rank|0.048568|0.2066|descriptive only; small n; dz undefined/inflated when sd=0|

## context_effect|llama3.2:latest (context effect: PLAIN minus FULL)

|Metric|n paired|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|96|0.0052|0.0|wilcoxon_signed_rank|0.64695|0.03|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|95|0.0053|0.0|wilcoxon_signed_rank|0.97015|0.0234|descriptive only; small n; dz undefined/inflated when sd=0|

## context_effect|mistral:latest (context effect: PLAIN minus FULL)

|Metric|n paired|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|96|0.013|0.0|wilcoxon_signed_rank|0.38909|0.0713|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|96|-0.0234|0.0|wilcoxon_signed_rank|0.235408|-0.1178|descriptive only; small n; dz undefined/inflated when sd=0|

## context_effect|phi3:mini (context effect: PLAIN minus FULL)

|Metric|n paired|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|96|-0.0391|0.0|wilcoxon_signed_rank|0.153566|-0.1739|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|96|-0.0078|0.0|wilcoxon_signed_rank|0.634526|-0.0476|descriptive only; small n; dz undefined/inflated when sd=0|

## context_effect|qwen3:latest (context effect: PLAIN minus FULL)

|Metric|n paired|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|96|-0.013|0.0|wilcoxon_signed_rank|0.420282|-0.0604|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|95|0.0053|0.0|wilcoxon_signed_rank|0.637352|0.0482|descriptive only; small n; dz undefined/inflated when sd=0|

### gemma2:9b|FULL vs reference FULL

|Metric|n|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|95|-0.0132|0.0|wilcoxon_signed_rank|0.516412|-0.0781|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|94|0.0372|0.0|wilcoxon_signed_rank|0.174726|0.1532|descriptive only; small n; dz undefined/inflated when sd=0|

### gemma2:9b|PLAIN vs reference FULL

|Metric|n|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|96|-0.0052|0.0|wilcoxon_signed_rank|0.881497|-0.0287|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|95|0.0711|0.0|wilcoxon_signed_rank|0.001002|0.3358|descriptive only; small n; dz undefined/inflated when sd=0|

### llama3.2:latest|PLAIN vs reference FULL

|Metric|n|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|96|0.0052|0.0|wilcoxon_signed_rank|0.64695|0.03|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|95|0.0053|0.0|wilcoxon_signed_rank|0.97015|0.0234|descriptive only; small n; dz undefined/inflated when sd=0|

### mistral:latest|FULL vs reference FULL

|Metric|n|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|96|0.0234|0.0|wilcoxon_signed_rank|0.12819|0.1564|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|95|0.0684|0.0|wilcoxon_signed_rank|0.004317|0.3108|descriptive only; small n; dz undefined/inflated when sd=0|

### mistral:latest|PLAIN vs reference FULL

|Metric|n|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|96|0.0365|0.0|wilcoxon_signed_rank|0.019984|0.2144|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|95|0.0447|0.0|wilcoxon_signed_rank|0.087527|0.1965|descriptive only; small n; dz undefined/inflated when sd=0|

### phi3:mini|FULL vs reference FULL

|Metric|n|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|96|0.0078|0.0|wilcoxon_signed_rank|0.61209|0.0516|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|95|0.0474|0.0|wilcoxon_signed_rank|0.035956|0.2286|descriptive only; small n; dz undefined/inflated when sd=0|

### phi3:mini|PLAIN vs reference FULL

|Metric|n|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|96|-0.0312|0.0|wilcoxon_signed_rank|0.251811|-0.1451|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|95|0.0368|0.0|wilcoxon_signed_rank|0.142357|0.1474|descriptive only; small n; dz undefined/inflated when sd=0|

### qwen3:latest|FULL vs reference FULL

|Metric|n|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|96|0.0182|0.0|wilcoxon_signed_rank|0.179712|0.0946|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|95|0.0737|0.0|wilcoxon_signed_rank|0.001173|0.3472|descriptive only; small n; dz undefined/inflated when sd=0|

### qwen3:latest|PLAIN vs reference FULL

|Metric|n|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|96|0.0052|0.0|wilcoxon_signed_rank|0.639412|0.03|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|94|0.0824|0.0|wilcoxon_signed_rank|0.000203|0.3952|descriptive only; small n; dz undefined/inflated when sd=0|

## 6. Figures

- `fig1_baseline_quality.png`
- `fig2_context_effect.png`
- `fig3_generation_latency.png`

## 7. Limitations

- All generation models are small local models (<=9B); no large external API model is included.
- The judge is llama3.2 (same as RQ2/ablation) for cross-condition comparability; it may be biased toward the reference generator's style.
- 96 curated in-scope gold questions; ceiling effects apply.
- The longitudinal records are SYNTHETIC (Synthea-derived); no clinical validity is claimed.