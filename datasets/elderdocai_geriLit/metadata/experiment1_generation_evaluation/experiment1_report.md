# Experiment 1 - Final-Stack GeriLit-Gold v1.1 Generation-Quality Evaluation - Report

**Status:** PASS

## 1. Dataset

- Benchmark: `data/geri_lit_gold_v1_1.json` (GeriLit-Gold v1.1, FROZEN)
- n = 121 (121 reviewed, 121 accepted, 0 rejected, 0 pending)
- Benchmark SHA-256: `1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72`
- Unique PMCIDs in benchmark: 45 - Unique gold chunks: 116

## 2. System configuration (frozen, read-only)

- Retrieval backend: `geri_lit` (dense BGE top-5 + BM25 top-5 -> 0.6/0.4 hybrid -> CrossEncoder rerank -> top-3)
- Embedding model: `BAAI/bge-base-en-v1.5`
- CrossEncoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- LLM: Ollama `llama3.2:latest` (3.2B Q4_K_M, local)
- Generation params: temperature 0, top_p 0.1, top_k 10
- Judge model: `llama3.2:latest` with deterministic sampler (temperature 0, top_p 0.1, top_k 10); prompts reused verbatim from `scripts/evaluate_gold_qa.py`
- Reliability: `0.3*authority + 0.3*relevance + 0.2*support + 0.1*coverage + 0.1*consistency`; ACCEPT >= 0.80, REFINE >= 0.65, RE-RETRIEVE >= 0.45, REJECT < 0.45
- Gate: Task 1 programmatic gate (MAX_REFINE=1, MAX_RETRIEVE=1); Task 2 relevance semantics (similarity_score = dense cosine)

## 3. Evaluation methodology (fixed before generation)

- Faithfulness: `FAITHFULNESS_PROMPT` LLM judge, 0-1 scale; unsupported-claim flag = faithfulness <= 0.50
- Answer relevance: `RELEVANCE_PROMPT` LLM judge, 0-1 scale
- Evidence support: deterministic `span_token_coverage(answer, final_evidence_text)`
- Hallucination/unsupported: faithfulness <= 0.50; contradiction = faithfulness == 0.0
- Refusal: actual gate `refused` flag; refusal quality not re-judged (no ground-truth labels)

## 4. Generation results (n=121)

| Metric | n | Mean | Median | Std | Min | Max |
| Faithfulness | 121 | 0.7541 | 1.0000 | 0.3966 | 0.0000 | 1.0000 |
| Answer relevance | 121 | 0.6715 | 1.0000 | 0.4496 | 0.0000 | 1.0000 |
| Evidence support | 121 | 0.8528 | 0.9500 | 0.2408 | 0.0000 | 1.0000 |
| Unsupported-claim rate (faith <= 0.50) | 28 | 23.14% | - | - | - | - |
| Contradiction rate (faith == 0.0) | 22 | 18.18% | - | - | - | - |

## 5. Reliability / generation behavior

| Decision | Count | Percentage |
|----------|-------:|-----------:|
| ACCEPT | 37 | 30.6% |
| REFINE | 84 | 69.4% |
| RE-RETRIEVE | 0 | 0.0% |
| REJECT | 0 | 0.0% |

- Mean final reliability: 0.7884
- Refinement cases: 84 - Re-retrieval cases: 0 - Refusal count: 0 - Generation permitted: 121
- Evidence count: mean 2.9917 - median 3
- Empty-evidence cases: 0 - Error cases: 0
- Gold chunk present in final evidence: 33/121


## 6. Topic-level results (descriptive, not ranked)

| Topic | n | mean rel. score | mean relevance factor | mean faith. | mean ans. relv. | mean evid. support |
|-------|---|-----------------|----------------------|------------|----------------|-------------------|
| C01 | 11 | 0.7879 | 0.5443 | 0.8864 | 0.8636 | 0.9597 |
| C02 | 15 | 0.7828 | 0.5320 | 0.8167 | 0.5000 | 0.8196 |
| C03 | 13 | 0.7929 | 0.5371 | 0.5577 | 0.7500 | 0.7945 |
| C04 | 13 | 0.7786 | 0.5253 | 0.8269 | 0.6731 | 0.8447 |
| C05 | 14 | 0.8023 | 0.6037 | 0.8214 | 0.8571 | 0.8804 |
| C06 | 16 | 0.7824 | 0.5402 | 0.7812 | 0.6406 | 0.8131 |
| C07 | 14 | 0.7756 | 0.5349 | 0.6786 | 0.3929 | 0.8086 |
| C08 | 9 | 0.8017 | 0.5897 | 0.8889 | 0.7500 | 0.9237 |
| C09 | 10 | 0.7925 | 0.5644 | 0.6750 | 0.6500 | 0.9692 |
| C10 | 6 | 0.8016 | 0.6085 | 0.5000 | 0.7917 | 0.7288 |

## 7. Reproducibility

- run 1 signature: `01879a90399ec50694bb59400d3b4e3968343f5ffc7bf7e6e6b10620d9452223`
- run 2 signature: `01879a90399ec50694bb59400d3b4e3968343f5ffc7bf7e6e6b10620d9452223`
- signatures identical: **True**
- deterministic: **True**
- mismatched question IDs: []

| Substantive field | identical |
|-------------------|-----------|
| retrieved_evidence_ids | True |
| reliability | True |
| decision | True |
| refinement_attempts | True |
| retrieval_attempts | True |
| refused | True |
| generated | True |
| answer | True |
| faithfulness | True |
| answer_relevance | True |
| evidence_support | True |
| hallucination | True |

## 8. Validation

- checks passed: 63/63
- benchmark SHA-256: `1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72`

## 9. Scientific interpretation (descriptive only)

This report measures the current frozen pipeline; it makes no claim that any configuration is 'better' and does not compare legacy-stack results as if they were the same experiment. Unsupported-claim/hallucination flags are derived from the frozen faithfulness rubric (<= 0.50).
