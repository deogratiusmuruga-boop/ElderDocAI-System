# ElderDocAI - External LLM Generation Baselines Report

> Synthetic longitudinal records (Synthea-derived) + 16 gold QA questions.
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
|llama3.2:latest|FULL|0.9375|0.8906|0.9394|1.0469|0.875|0|0|
|llama3.2:latest|PLAIN|0.9688|0.9531|0.9394|0.8592|1.0169|0|0|
|mistral:latest|FULL|0.8906|0.9844|0.9394|2.0137|1.0366|0|0|
|mistral:latest|PLAIN|0.9531|0.9844|0.9394|1.0431|0.8035|0|0|
|qwen3:latest|FULL|0.9062|1.0|0.9394|5.2461|0.9438|0|0|
|qwen3:latest|PLAIN|0.9375|0.9531|0.9394|6.2507|1.1441|0|0|
|gemma2:9b|FULL|0.9844|0.9531|0.9394|3.3898|0.9407|0|0|
|gemma2:9b|PLAIN|0.8906|0.9375|0.9394|1.6389|0.7405|0|0|
|phi3:mini|FULL|0.8906|0.9688|0.9394|1.7425|2.3153|0|0|
|phi3:mini|PLAIN|0.8438|0.9375|0.9394|0.584|0.7988|0|0|

## 3. Delta vs reference FULL (llama3.2 FULL)

|Condition|Faithfulness delta|Relevance delta|
|--:|--:|--:|
|llama3.2:latest|PLAIN|0.0313|0.0625|
|mistral:latest|FULL|-0.0469|0.0938|
|mistral:latest|PLAIN|0.0156|0.0938|
|qwen3:latest|FULL|-0.0313|0.1094|
|qwen3:latest|PLAIN|0.0|0.0625|
|gemma2:9b|FULL|0.0469|0.0625|
|gemma2:9b|PLAIN|-0.0469|0.0469|
|phi3:mini|FULL|-0.0469|0.0782|
|phi3:mini|PLAIN|-0.0937|0.0469|

## 4. Reliability / grounding note

Because all conditions use the same hybrid retrieval (top-3, reranked) and the same reliability configuration, the retrieved evidence, grounding metrics, and reliability/decision scores are identical across every condition (any difference would indicate a data-integrity problem). Reliability decision distribution: ["{'ACCEPT': 16}"].

## 5. Statistical analysis (paired Wilcoxon vs reference FULL)

## context_effect|gemma2:9b (context effect: PLAIN minus FULL)

|Metric|n paired|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|16|-0.0938|0.0|wilcoxon_signed_rank|0.033895|-0.6057|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|16|-0.0156|0.0|wilcoxon_signed_rank|1.0|-0.0673|descriptive only; small n; dz undefined/inflated when sd=0|

## context_effect|llama3.2:latest (context effect: PLAIN minus FULL)

|Metric|n paired|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|16|0.0312|0.0|wilcoxon_signed_rank|0.414216|0.2019|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|16|0.0625|0.0|wilcoxon_signed_rank|0.234194|0.2919|descriptive only; small n; dz undefined/inflated when sd=0|

## context_effect|mistral:latest (context effect: PLAIN minus FULL)

|Metric|n paired|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|16|0.0625|0.0|wilcoxon_signed_rank|0.271396|0.2685|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|16|0.0|0.0|wilcoxon_signed_rank|None|None|identical per-question values; test not meaningful|

## context_effect|phi3:mini (context effect: PLAIN minus FULL)

|Metric|n paired|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|16|-0.0469|0.0|wilcoxon_signed_rank|0.558009|-0.1691|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|16|-0.0312|0.0|wilcoxon_signed_rank|0.157299|-0.366|descriptive only; small n; dz undefined/inflated when sd=0|

## context_effect|qwen3:latest (context effect: PLAIN minus FULL)

|Metric|n paired|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|16|0.0312|0.0|wilcoxon_signed_rank|0.414216|0.2019|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|16|-0.0469|0.0|wilcoxon_signed_rank|0.083265|-0.4651|descriptive only; small n; dz undefined/inflated when sd=0|

### gemma2:9b|FULL vs reference FULL

|Metric|n|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|16|0.0469|0.0|wilcoxon_signed_rank|0.179712|0.3447|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|16|0.0625|0.0|wilcoxon_signed_rank|0.317311|0.2685|descriptive only; small n; dz undefined/inflated when sd=0|

### gemma2:9b|PLAIN vs reference FULL

|Metric|n|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|16|-0.0469|0.0|wilcoxon_signed_rank|0.317311|-0.25|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|16|0.0469|0.0|wilcoxon_signed_rank|0.580712|0.1533|descriptive only; small n; dz undefined/inflated when sd=0|

### llama3.2:latest|PLAIN vs reference FULL

|Metric|n|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|16|0.0312|0.0|wilcoxon_signed_rank|0.414216|0.2019|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|16|0.0625|0.0|wilcoxon_signed_rank|0.234194|0.2919|descriptive only; small n; dz undefined/inflated when sd=0|

### mistral:latest|FULL vs reference FULL

|Metric|n|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|16|-0.0469|0.0|wilcoxon_signed_rank|0.379537|-0.2059|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|16|0.0938|0.0|wilcoxon_signed_rank|0.130797|0.3917|descriptive only; small n; dz undefined/inflated when sd=0|

### mistral:latest|PLAIN vs reference FULL

|Metric|n|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|16|0.0156|0.0|wilcoxon_signed_rank|0.654721|0.1089|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|16|0.0938|0.0|wilcoxon_signed_rank|0.130797|0.3917|descriptive only; small n; dz undefined/inflated when sd=0|

### phi3:mini|FULL vs reference FULL

|Metric|n|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|16|-0.0469|0.0|wilcoxon_signed_rank|0.405381|-0.2059|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|16|0.0781|0.0|wilcoxon_signed_rank|0.197466|0.3302|descriptive only; small n; dz undefined/inflated when sd=0|

### phi3:mini|PLAIN vs reference FULL

|Metric|n|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|16|-0.0938|0.0|wilcoxon_signed_rank|0.248213|-0.3114|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|16|0.0469|0.0|wilcoxon_signed_rank|0.496242|0.1691|descriptive only; small n; dz undefined/inflated when sd=0|

### qwen3:latest|FULL vs reference FULL

|Metric|n|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|16|-0.0312|0.0|wilcoxon_signed_rank|0.414216|-0.2019|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|16|0.1094|0.0|wilcoxon_signed_rank|0.0656|0.4904|descriptive only; small n; dz undefined/inflated when sd=0|

### qwen3:latest|PLAIN vs reference FULL

|Metric|n|Delta mean|Delta median|Test|p-value|dz|Note|
|--:|--:|--:|--:|--:|--:|--:|--:|
|faithfulness|16|0.0|0.0|wilcoxon_signed_rank|1.0|0.0|descriptive only; small n; dz undefined/inflated when sd=0|
|answer_relevance|16|0.0625|0.0|wilcoxon_signed_rank|0.380455|0.2348|descriptive only; small n; dz undefined/inflated when sd=0|

## 6. Figures

- `fig1_baseline_quality.png`
- `fig2_context_effect.png`
- `fig3_generation_latency.png`

## 7. Limitations

- All generation models are small local models (<=9B); no large external API model is included.
- The judge is llama3.2 (same as RQ2/ablation) for cross-condition comparability; it may be biased toward the reference generator's style.
- 16 curated in-scope gold questions; ceiling effects apply.
- The longitudinal records are SYNTHETIC (Synthea-derived); no clinical validity is claimed.