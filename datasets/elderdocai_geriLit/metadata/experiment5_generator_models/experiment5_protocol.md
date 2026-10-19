# Experiment 5 — Generator-Model Ablation on GeriLit-Gold v1.1

**Status:** PROTOCOL (frozen before first generation — do not edit after running)
**Date:** 2026-10-16
**Checkpoint:** `cd99b30` (release tag `v1.1-paper`; branch `release/geri-lit-v1.1`)
**Frozen benchmark:** `data/geri_lit_gold_v1_1.json`, SHA-256
`1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72`, n = 121.
**Judge (DECLARED BY THE PROTOCOL before ANY generation — identical everywhere):**
`llama3.2:latest`
**Generators under test (five; ONLY the `model=` parameter differs):**
`llama3.1:8b`, `qwen2.5:7b`, `mistral:7b`, `gemma2:9b`, `phi3.5`.

---

## 1. Research question

Does the final ElderDocAI generation pipeline produce materially different
faithfulness / relevance / evidence-support answer quality when the generator
LLM is swapped while everything else — retrieval, reliability gating, grounded
prompt, decoding options (temperature 0, top_p 0.1, top_k 10), and the judge —
remains byte-identical to the frozen Experiment-1 protocol?

## 2. System under test (all frozen, read-only; identical to Experiment 1)

| Component | Value | Source (frozen) |
|---|---|---|
| Retrieval backend | `geri_lit` (`ELDERDOCAI_RETRIEVAL_BACKEND=geri_lit`) | `scripts/retrieval_router.py`, `scripts/geri_lit_retriever.py` |
| Dense | BGE `BAAI/bge-base-en-v1.5`, FAISS IndexFlatIP, top-5 | `datasets/elderdocai_geriLit/geri_lit_retriever.py` |
| Sparse | BM25Okapi (stored index), top-5 | same |
| Fusion | `0.6 * dense + 0.4 * sparse` | same |
| Rerank | CrossEncoder `cross-encoder/ms-marco-MiniLM-L-6-v2`, top-5 -> top-3 final | same |
| Reliability | `0.3*authority + 0.3*relevance + 0.2*support + 0.1*coverage + 0.1*consistency` | `scripts/reliability_evaluation.py` + `config/reliability_config.json` |
| Thresholds | ACCEPT >= 0.80, REFINE >= 0.65, RE-RETRIEVE >= 0.45, REJECT < 0.45 | `config/reliability_config.json` |
| Gate | ACCEPT/REFINE/RE-RETRIEVE/REJECT enforced; MAX_REFINE=1, MAX_RETRIEVE=1 | `scripts/rag_chat.py::_run_gated_generation` |
| Grounded prompt | `build_grounded_prompt` (from frozen Exp-1) | `scripts/rag_chat.py` |
| Generation options | `{"temperature": 0, "top_p": 0.1, "top_k": 10}` (identical to Exp-1; `num_ctx` NOT set, matching Exp-1) | `scripts/rag_chat.py::generate_answer` |
| Judge | `llama3.2:latest`, options `{"temperature": 0, "top_p": 0.1, "top_k": 10}` | frozen `scripts/rag_chat.py::_llm_judge` |

Every prompt, model weight, threshold, chunking, index, judge system-prompt
and benchmark record is byte-identical to Experiment 1. **The ONLY change is the
generator model string passed to the production generation call.**

### num_ctx record (honest, read directly from the frozen Exp-1 call)

`git grep num_ctx -- scripts datasets/elderdocai_geriLit` returns **no hits**;
the frozen Exp-1 production options dict is byte-exact
`{"temperature": 0, "top_p": 0.1, "top_k": 10}`. Therefore Experiment 1 (and this
experiment) run with the **Ollama server default context window**; no explicit
`num_ctx` value is set by this experiment either. This is recorded rather than
invented; the server-default context was already in effect for the frozen
Experiment-1 judge and generator.

## 3. Judge (literally the same function as Exp-1, llama3.2 judge)

- Faithfulness: `FAITHFULNESS_PROMPT` + `llama3.2:latest`, temperature 0 /
  top_p 0.1 / top_k 10.
- Answer relevance: `RELEVANCE_PROMPT` + same judge.
- Evidence support: deterministic `span_token_coverage(answer, evidence)`.
- Hallucination flag: `faithfulness <= 0.50` (bool); aggregated
  `unsupported_claim_rate` / `contradiction_rate` (faithfulness == 0.0).
- Judge model, judge sampler, judge prompts: **all reused as constants from the
  frozen Experiment-1 judge** so the Exp-5 judge is the same judge. Only the
  generator model differs.

## 4. Paired generator-model conditions

For each of the five generators, run the ENTIRE 121-question frozen benchmark
**twice** (run-id 1 then run-id 2) with identical inputs and identical sampler
options. All substantive per-question fields (evidence IDs, reliability
factors, decision, refinement/retrieval attempts, refused, generated, answer,
faithfulness, answer_relevance, evidence_support, hallucination) are compared
run-to-run for each condition and reported per-field, exactly like Exp-1's
reproducibility manifest.

## 5. Integrity (frozen, unchanged from Exp-1)

Record SHA-256 of the frozen benchmark and frozen GeriLit artifacts (v1.1
benchmark SHA `1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72`,
chunks, embeddings, FAISS, BM25, row mapping, reliability config, Task 10D–10H,
Experiment-1 artifacts) before and after every run → any unexpected change →
STOP. No production-code, index, benchmark, threshold or prompt modification;
no downloads beyond the five generator + one judge models already pulled.

## 6. Outputs (all under this directory)

`experiment5_generator_models.py`, `experiment5_per_question_<model>_run{1,2}.json`,
`experiment5_summary_<model>_run{1,2}.json`, `experiment5_agent_models.py`
(agent list + `ollama show` digest), `experiment5_validate.py`,
`experiment5_validation.{json,md}`, `experiment5_reproducibility.{json,md}`,
`experiment5_frozen_hashes.json`, `experiment5_run_manifest.json`,
`experiment5_report.md`, `experiment5_protocol.md`.
