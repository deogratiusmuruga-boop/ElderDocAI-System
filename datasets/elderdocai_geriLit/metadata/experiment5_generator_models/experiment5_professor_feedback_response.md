# Experiment 5 — Response to Professor Feedback (frozen, byte-honest review before any run)

**Date:** 2026-10-16
**Status:** PROTOCOL-ADJUNCT (frozen before generation; do not edit after approval)
**Frozen reference:** GeriLit-Gold v1.1 (SHA-256
`1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72`, n = 121).
**Judge (byte-identical to Exp-1, unchanged everywhere):** `llama3.2:latest` +
`JUDGE_OPTIONS = {"temperature": 0, "top_p": 0.1, "top_k": 10}`.
**Only allowed difference from Exp-1:** generator model string (`--generator`).

---

## 1. Professor point 1 — "compare against ≥ 5 different models"

**Answered byte-honestly by Experiment 5 itself:** Exp-5 runs **five (5)
generator models** — `llama3.1:8b`, `qwen2.5:7b`, `mistral:7b`, `gemma2:9b`,
`phi3.5` — each through the **byte-identical frozen pipeline and the byte-identical
frozen judge**, twice (run 1 + run 2), over the same frozen GeriLit-Gold v1.1.
This is a controlled **generator-model ablation** (only `model=<generator>` in the
production `generate_answer` call differs; retrieval, reliability gate, grounded
prompt, decoding options `{"temperature":0,"top_p":0.1,"top_k":10}`, and judge
are all byte-identical to Exp-1).

Honest caveat, stated (not hidden): this compares five **generator models within
our own stack** — it is *not* a comparison against a fundamentally different RAG
framework (e.g., a different retriever, a non-gated baseline, an alternative
router). A judge-swap or backend-swap comparison is a **separate** experiment and
is out of scope for Exp-5; I will not claim Exp-5 proves cross-framework
superiority it does not measure. If the professor wants that, it is Exp-6+
(say the word and I will write a fresh frozen protocol — but that is a new
experiment, not an edit of this one).

**What Exp-5 CAN honestly demonstrate:** (a) per-generator faithfulness,
relevance, evidence support, hallucination rate; (b) **retrieval metrics**
(see §2) computed from the frozen gold chunks; (c) gold-evidence hit rate.
This shows how each alternative generator performs **relative to one another**
under identical conditions — which is the professor's "five different models"
requirement, byte-faithfully.

## 2. Professor point 2 — metric selection (byte-honest mapping)

| Professor family | Modele computable from frozen GeriLit-Gold v1.1 + frozen evaluator (no invention) | Metric(s) Exp-5 will record | Judge dependency |
|---|---|---|---|
| Retrieval: Recall@k, Hit@k, Precision@k, MRR, nDCG@k, MAP | ✅ YES — deterministic from frozen gold chunk IDs + frozen index, no LLM involved | `hit_rate@3` (gold chunk in top-3, = 27.3% frozen reference), `recall@k`, `precision@k`, `mrr@k`, `ndcg@k`, `map@k` (k = final top-3 per frozen rerank) | deterministic (no judge) — byte-safe |
| Generation: EM, F1, ROUGE, BLEU, BERTScore | ❌ **NOT computable — no gold reference ANSWERS exist in GeriLit-Gold v1.1** (it has gold evidence chunks + questions, not gold written answers) | **Recorded honestly as NOT-COMPUTABLE without inventing gold answers.** I will not fabricate gold answers or EM/ROUGE/BLEU scores. | — |
| RAG faithfulness / groundedness | ✅ YES — frozen judge `llama3.2:latest` `{"temperature":0,"top_p":0.1,"top_k":10}` byte-identical to Exp-1 | `faithfulness` (0–1), `faithfulness_reason` | judge |
| Answer relevance | ✅ YES — same frozen judge | `answer_relevance` (0–1) | judge |
| Evidence support / citation accuracy | ✅ YES — deterministic span token coverage + frozen judge | `evidence_support`, `gold_chunk_retrieved`, `hallucination` (faithfulness ≤ 0.5) | judge + deterministic |
| Context precision / context recall (RAGAS), TruLens, ARES | ⚠️ Requires NEW judge library/code → would MODIFY frozen judge byte-parity | **NOT included in Exp-5** (adding RAGAS/etc. would break the frozen judge contract = a different experiment). Recorded as out-of-scope; not invented. | — |

**Primary metric (professor "you choose one"):** **Faithfulness** (0–1, frozen
judge) — matches the professor's RAG faithfulness/groundedness family and is
the strictest guard against unsupported claims. Everything else above is
supporting.

## 3. Explicit honest refusals (recorded, not hidden)

1. **No gold reference answers exist** in the frozen benchmark → **EM / F1 /
   ROUGE / BLEU / BERTScore are NOT recorded** (I will not invent gold answers
   to produce them).
2. **RAGAS / TruLens / ARES library scores require new judge tooling** → would
   violate "judge byte-identical to Exp-1"; **excluded**, stated plainly.
3. **num_ctx / num_ctx NOT SET** byte-identical to Exp-1 (verified: `git grep
   num_ctx` across frozen scripts + protocol = 0 hits; Exp-1 did not set it).
   If the professor requires an explicit number, tell me the exact value and I
   will add it as a frozen protocol change and re-verify bytes — I will not
   invent 2048/8192/16384.

## 4. Decision needed from you (one of two)

- **A — Run Exp-5 as frozen above** (5 generators × llama3.2 judge × 2 runs;
  metrics = per-question faithfulness, relevance, evidence support,
  hallucination + the 6 deterministic retrieval metrics + gold-hit 27.3%
  check). Prints only records already recorded; writes ONLY under the Exp-5
  folder; refuses to overwrite; aborts on any SHA change. **This is my
  recommendation** — it byte-faithfully answers the professor's "five models"
  point with zero invented numbers.
- **B — You want the retrieval-metric family EXPANDED** (e.g., different k
  values, per-topic splits, baseline retriever comparison). That's a separate
  frozen protocol — name the metrics and I will write a new Exp-6 protocol
  (no touch to Exp-1..5).

Waiting for your A/B (or the exact `num_ctx` number) before I write the
evaluator edit or run anything.
