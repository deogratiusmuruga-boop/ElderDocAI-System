# ElderDocAI — Phase 3 Task 10A
## Integration Architecture Audit (read-only) + Task 10B Implementation Plan

> **Scope**: READ-ONLY audit. No production/source code, dataset, index, or frozen
> artifact was modified. No human review, no benchmark results, no API/frontend
> change, no commit/push.

---

## 1. Repository state (verified)
- Branch: `feature/mimic-pmc-migration`, HEAD `8b0696c` (unchanged).
- `.gitignore` modified (pre-existing raw-ignore rules only); untracked
  `datasets/elderdocai_geriLit/`, `datasets/mimic_demo/`, GeriLit-Gold candidate/review
  files, and previously excluded files. Nothing staged/committed/pushed.
- Frozen artifacts verified unchanged (SHA-256):
  - chunks `62bfdde3c7e77cf56112e79a71bc79bdea70767f6b2625701e392bbb6581c6b3`
  - embeddings `b0905d2f96903dadc3cf5b4a737f330853656f846955de706f21bd714ce2dacf`
  - FAISS `b7016b9dabbab08ee68f875007b6db2b013af4472b1a3c0958af8a566b59f2b3`
  - BM25 `17f503240f2f9a13abf58c94359884b03b06bec1a5a59f63f921328e2a2c70c9`
  - row_mapping `4d649f81d554c41ec7b63c65dce887466dfa6245745c568315735f1b93a1872b`
  - Gold16 `e44788926afe351d30bad2e253cfe0712ad83a4ee3b3d9abe6a63c2af0baf544`
  - Gold96 `d4e1b6e6c48cd84fc95fcbb7bf0df375365cd354ec8cc31f16dde2b3881b5299`
  - accepted_manifest `f80cd457e5e2be55ed3a3de09e2554e9d585e76cef8dc44800270dfa66a99a49`

## 2. Existing ElderDocAI retrieval flow (actual call graph)
Derived from code (NOT assumed):

```
POST /ask                          (api/main.py, ask endpoint)
 → load user_profile (DB)          (optional)
 → scripts.carebuddy_service.answer_question
    → scripts.rag_chat.generate_answer
       → extract_patient_id(user_profile)
       → get_adaptive_context(patient_id)          # Synthea adaptive context (JSON)
       → prepare_adaptive_context(...)
       → get_assistance_plan(patient_id, window)   # assistance plan lookup
       → prepare_assistance_plan(...)
       → scripts.hybrid_retriever.hybrid_search(query)   ← RETRIEVAL ENTRY POINT
            → embedding_model.encode(query, normalize)   # BAAI/bge-base-en-v1.5
            → faiss_index.search(top_k=5)                # legacy IndexFlatIP
            → bm25.get_scores(lower().split()) top-5     # legacy BM25Okapi
            → merge by chunk_id; hybrid = 0.6*d_norm + 0.4*s_norm
            → CrossEncoder(ms-marco-MiniLM-L-6-v2) rerank top-5
            → final top-3 chunks
       → prepare_evidence(retrieved_chunks)         # → evidence_items
       → scripts.reliability_evaluation.evaluate_reliability(query, evidence_items)
       → scripts.adaptive_decision_controller.make_reliability_decision(reliability)
       → print_reliability_report(...)
       → scripts.build_grounded_prompt.build_grounded_prompt(...)
       → ollama.chat(model="llama3.2:latest",
                      options={temperature:0, top_p:0.1, top_k:10}, ...)
       → return (answer, evidence)                   # via carebuddy_service
    → evaluate_reliability again (service layer)
    → make_reliability_decision again (service layer)
    → sources = [item.source_document for evidence]
    → care_context = prepare_care_context(adaptive_context) + assistance plan
 → return {answer, sources, reliability, decision, care_context, profile_used}
```

Colocated with this flow:
- `scripts/reranker.py` (standalone CrossEncoder helper; not used by rag_chat, which
  uses hybrid_retriever's inline rerank).
- `scripts/evidence_aggregation.py` (FAISS→evidence with authority mapping; not used by
  rag_chat, which uses `prepare_evidence`).

Key observations:
- **Retrieval entry point**: `hybrid_search(query)` in `rag_chat.generate_answer`.
- **Retrieved chunk type**: list of dicts with keys
  `{category, chunk_id, document_id, document_type, language, last_updated,
    organization, source_document, text, title}` (legacy), then normalized by
  `prepare_evidence` into `{chunk_id, source_document, document_category,
    authority_score, similarity_score, text}`.
- **Reliability** consumes exactly `{text, similarity_score, authority_score}` per item
  (see `evaluate_reliability` required_fields). `similarity_score` in practice is the
  legacy `rerank_score`.
- **Decision** thresholds: ACCEPT ≥ 0.80, REFINE ≥ 0.65, RE-RETRIEVE ≥ 0.45, else REJECT
  (config file).
- **Prompt**: `build_grounded_prompt` renders `source_document` + `text` per item,
  reliability block, profile, language instruction, assistance plan.
- **LLM**: Ollama `llama3.2:latest`, temp 0 / top_p 0.1 / top_k 10.
## 3. GeriLit retriever interface (read-only; frozen)
File: `datasets/elderdocai_geriLit/geri_lit_retriever.py` (Task 6, validated).

- **Class**: `GeriLitRetriever(base_dir=None, offline=True)`.
- **Init**: loads `chunks/chunks.jsonl`, `index/row_mapping.json`,
  `index/faiss_index.bin`, `index/bm25.pkl`; loads `BAAI/bge-base-en-v1.5` and
  `cross-encoder/ms-marco-MiniLM-L-6-v2` with `local_files_only=offline` and
  HF offline env vars.
- **Public methods**: `embed_query(query)`, `dense_retrieve(query, k=5)`,
  `sparse_retrieve(query, k=5)`, `hybrid_retrieve(query, dense_k=5, sparse_k=5)`,
  `rerank(query, candidates)`, `row_to_chunk(row)`, `retrieve(query)`.
- **Config**: DENSE_K=5, SPARSE_K=5, RERANK_K=5, FINAL_K=3, DENSE_W=0.6, SPARSE_W=0.4.
- **Returned evidence item** (from `retrieve`):
  `{row_id, chunk_id, pmcid, pmid, doi, title, journal, publication_year, region,
    section_id, section_title, content_type, evidence_type, source_locator,
    text, dense_score, sparse_score, hybrid_score, rerank_score, source_document}`
  where `source_document = "PMCID (Journal, Year)"`.
- **Determinism**: fusion and rerank use deterministic tie-breaks; validated in Task 6/7.
- **Offline**: requires only local caches; no downloads (validated).
- **Error behavior**: raises on missing files; asserts row mapping consistency.

## 4. Evidence schema compatibility (field-by-field)
| Field | Existing (legacy → prepare_evidence) | GeriLit retriever | Compatible? | Adapter/conversion needed |
|---|---|---|---|---|
| `chunk_id` | string (per-doc unique) | string (globally unique) | YES | none (superset) |
| `source_document` | PDF filename | `"PMCID (Journal, Year)"` | YES (string) | none; semantics change |
| `document_category` | legacy `category`/`document_category` | absent | PARTIAL | synthesize e.g. `"PMC"` / `"gerilit"` |
| `authority_score` | from authority_mapping (org-based) | absent | NO | MUST supply (reliability requires key) |
| `similarity_score` / `rerank_score` | `rerank_score` | `rerank_score` | YES | map to `similarity_score` |
| `text` | str | str | YES | none |
| `pmcid` | absent | present | YES (extra) | none |
| `pmid` / `doi` | absent | present | YES (extra) | none |
| `section_id`/`section_title`/`region` | absent | present | YES (extra) | none |
| `content_type` / `evidence_type` | absent | present | YES (extra) | none |
| `source_locator` | absent | present | YES (extra) | none |
| `dense_score`/`sparse_score`/`hybrid_score` | absent | present | YES (extra) | none |
| `row_id` | absent | present | YES (extra) | none |

Conclusion: a thin adapter that injects `authority_score` + `document_category` and
maps `rerank_score → similarity_score` is sufficient to feed GeriLit output into
`prepare_evidence`/`evaluate_reliability`/`build_grounded_prompt` **without changing
any frozen retriever or legacy code**.

## 5. Integration options (evaluated)
| Option | Files affected | Arch impact | Compatibility risk | Repro risk | Legacy behavior | Eval scripts | Rollback |
|---|---|---|---|---|---|---|---|
| A. Replace legacy retriever | `rag_chat.py`; retire legacy | High (core path changed) | High — Gold16/Gold96 source_document expectations break | High | Changed-to-GeriLit by default | Broken (legacy source_document labels mismatch) | Hard (revert commit) |
| B. Run GeriLit alongside + merge | `rag_chat.py`, merge util | Medium | Medium — cross-corpus score fusion unvalidated | Medium | Additive | Unaffected if legacy branch intact | Moderate |
| C. **Controlled routing** (config/ENV selects backend: `legacy` default, `geri_lit` opt-in) | `rag_chat.py` (few lines) + new router/adapter | Low | Low — legacy default unchanged | Low | **Unchanged by default** | Unaffected | Easy (toggle) |
| D. GeriLit as additional evidence after legacy | `rag_chat.py` evidence assembly | Medium | Medium — mixing sources in one prompt | Medium | Additive | Partially affected | Moderate |

**Recommended: OPTION C (controlled routing).** A tiny routing seam in
`rag_chat.generate_answer` selects the retrieval backend from
`ELDERDOCAI_RETRIEVAL_BACKEND` (default `legacy`; values `legacy | geri_lit | both`).
Legacy path remains byte-identical by default; GeriLit becomes an opt-in controlled
configuration for evaluation-only runs.

## 6. Care-state interaction audit
- Current profile/care-state inputs: `user_profile` (age, chronic_conditions,
  medications, preferred_language), adaptive context JSON (Synthea-derived),
  assistance plan. These are read-only inputs to prompt personalization (never
  evidence).
- **CURRENT**: no query→care-state routing exists; retrieval is corpus-wide.
- **POTENTIAL FUTURE (not in Task 10B)**: `context_status`/assistance `mode` could, in a
  later controlled task, bias source routing (e.g., prefer preventive-care evidence for
  `LIGHT_SUPPORT` states). This is explicitly future work; Task 10B does NOT implement
  care-state-conditioned routing.

## 7. Reliability/gating compatibility
- `evaluate_reliability` requires per-item `text, similarity_score, authority_score`.
  - **Consumes GeriLit unchanged**: `text`, `similarity_score` (mapped from
    `rerank_score`).
  - **Requires adapter**: `authority_score` (absent in GeriLit output) — assign a
    deterministic metadata-derived authority (e.g., PMC journal/`article_type`-based)
    without touching `authority_mapping.py`.
  - `document_category` is informational only; supply `"PMC"`/`"gerilit"` so downstream
    code reading `document_category` won't break.
- Decision controller (`make_reliability_decision`) consumes only the reliability dict;
  unchanged.
- `prepare_evidence` already defaults `authority_score` to `1.0` if absent — but for
  scientific integrity GeriLit evidence should carry a real, documented authority value,
  not the silent default.
## 8. Offline / model-loading risks
- `hybrid_retriever.py` loads BGE + CrossEncoder at import time; `GeriLitRetriever`
  loads the *same* models again. Both point at the same HuggingFace local cache → no new
  network calls, but **two model instances in memory** may cause slower startup and
  higher RAM. Task 10B should ensure models load once (shared loader or lazy init).
- `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` are set only inside the GeriLit module; the
  legacy module relies on cache. Setting these at process start is recommended.
- Ollama dependency: only needed at generation time, not retrieval — unchanged.
- Index reloads: FAISS/BM25/chunks/row_mapping load once per `GeriLitRetriever`
  instance; use a module-level singleton to avoid reload per query.

## 9. Backward-compatibility requirements for Task 10B
1. Legacy retrieval must be the DEFAULT and unchanged (`ELDERDOCAI_RETRIEVAL_BACKEND=legacy`).
2. Gold16/Gold96 evaluation scripts must run unmodified (they call `hybrid_search`/
   `rag_chat` internals; default legacy preserves behavior).
3. `/ask` contract unchanged: response fields `answer, sources, reliability, decision,
   care_context, profile_used`.
4. Frontend unchanged (it consumes only `/ask` response).
5. GeriLit enabled only via explicit env/config; evaluation uses a deterministic config.
6. Clean disable: removing/empty ENV path returns to legacy; no orphaned imports.
7. A GeriLit retrieval failure must not corrupt the legacy path (wrap in try/except →
   fallback to legacy or REJECT/empty-evidence).
8. Provenance preserved end-to-end (pmcid/section/source_locator) through the adapter.
9. Deterministic evaluation remains possible (fixed config, frozen artifacts).
10. Frozen GeriLit artifacts read-only; no writes from runtime.

## 10. Task 10B implementation plan (next task only)
Files to CREATE:
- `scripts/retrieval_router.py` — thin routing/backends + GeriLit evidence adapter.
- `scripts/geri_lit_adapter.py` — `to_legacy_evidence(geri_item)` mapping
  (`rerank_score→similarity_score`, inject `authority_score`, `document_category="PMC"`),
  pure function.
- `tests/test_retrieval_router.py` — routing + adapter + fallback tests.
- `metadata/validation/phase3_task10b_validation.{json,md}` after run.

Files to MODIFY (minimal):
- `scripts/rag_chat.py`: replace `from scripts.hybrid_retriever import hybrid_search`
  with a router call `retrieve(query, backend=os.getenv("ELDERDOCAI_RETRIEVAL_BACKEND","legacy"))`;
  keep `prepare_evidence` as-is (adapter output satisfies its fields).
- `api/main.py`: NO change (prefer not exposing backend config in the response).
- `scripts/carebuddy_service.py`: NO functional change (may pass through config).

Functions/classes to change:
- `rag_chat.generate_answer` — retrieval entry point only.
- New `GeriLitEvidenceAdapter.to_legacy_evidence`.
- New `RetrievalRouter.retrieve(query, backend)`.

Routing mechanism: env `ELDERDOCAI_RETRIEVAL_BACKEND ∈ {legacy, geri_lit, both}`; default
`legacy`. `both` = legacy only in Task 10B (GeriLit merge deferred until validated; do
not implement cross-corpus fusion in 10B).

Evidence normalization: adapter returns legacy-shaped dicts; `prepare_evidence` remains.

Fallback behavior: if GeriLit backend raises → log, use legacy results; if both fail →
empty evidence → existing REJECT path.

Validation tests: unit tests for adapter field mapping; router backend selection with
default legacy; integration smoke on `/ask` in legacy mode (unchanged); GeriLit-mode smoke
on 2–3 queries verifying provenance.

Rollback: revert `rag_chat.py` diff (few lines) + delete new router/adapter/tests.

## 11. Validation of this task
Checks (see `metadata/validation/phase3_task10a_validation.{json,md}`): source files
inspected, call graph documented, GeriLit interface documented, schemas compared, seam
identified, options evaluated, reliability compatibility evaluated, offline risks
evaluated, backward-compat documented, Task 10B plan documented, frozen artifacts
unchanged, no production files modified. All TRUE.

## 12. Git status after audit
- Branch `feature/mimic-pmc-migration`, HEAD `8b0696c`; only Task 10A metadata files
  added (untracked); no production/source modifications; nothing staged/committed/pushed.