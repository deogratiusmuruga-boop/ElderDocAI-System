# PHASE 3 TASK 10B — CONTROLLED GERILIT INTEGRATION — REPORT

## 1. Implementation status
**PASS** — controlled GeriLit integration implemented, unit-tested, and validated
offline. Legacy remains the default backend; GeriLit is opt-in via environment
variable; no API/frontend/care-state/reliability/prompt changes; nothing committed.

## 2. Files created
- `scripts/retrieval_router.py` — backend router (`retrieve(query, backend, fallback)`)
- `scripts/geri_lit_adapter.py` — `to_legacy_evidence()` adapter
- `tests/test_retrieval_router.py` — 11 unit tests (mock-based)
- `datasets/elderdocai_geriLit/metadata/task10b_validate.py`, its validation JSON/MD

## 3. Files modified
- `scripts/rag_chat.py` — exact 2-line diff (verified via `--unified=0`):
  1. `from scripts.hybrid_retriever import hybrid_search` ->
     `from scripts.retrieval_router import retrieve as retrieve_evidence`
  2. `retrieved_chunks = hybrid_search(query)` ->
     `retrieved_chunks = retrieve_evidence(query)`
  No other change (22/22 validator confirms).

## 4. Router behavior
- `ELDERDOCAI_RETRIEVAL_BACKEND` env: `legacy` (default) | `geri_lit` | `both`.
- Unknown values -> safe default `legacy`.
- `legacy` -> calls the original `hybrid_retriever.hybrid_search(query)` — verified
  byte-identical output vs direct call.
- `geri_lit` -> loads cached GeriLit retriever (singleton), runs `retrieve()`,
  adapts via `geri_lit_adapter`, returns legacy-shaped evidence.
- `both` -> **legacy-only** (Task 10A compatibility; no fusion implemented).

## 5. Adapter behavior
- `rerank_score -> similarity_score`; `text`/`chunk_id`/`source_document` preserved;
  `document_category`/`category` = `"PMC"`; `authority_score` = **explicit constant
  0.85** (documented integration value — NOT a clinical-quality claim, and NOT the
  silent `prepare_evidence` default of 1.0).
- Preserves all useful provenance: `row_id, pmcid, pmid, doi, title, journal,
  publication_year, region, section_id, section_title, content_type, evidence_type,
  source_locator, dense/sparse/hybrid/rerank_score`.
- Input dicts never mutated (unit-tested).

## 6. Environment-variable behavior
- Read per-query with deterministic default `legacy`; invalid -> `legacy` with a log
  warning. No new API parameter.

## 7. Fallback behavior
- `geri_lit` failure -> logs `[retrieval_router] error/fallback` and falls back to
  legacy (fallback=True default). With fallback=False -> returns empty list ->
  downstream REJECT/empty-evidence path. Never fabricates evidence; never bypasses
  reliability gating.

## 8. Model-loading behavior
- GeriLit models/indexes loaded once via a module-level singleton (lazy);
  `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` set before import. Loads from local
  cache only (no downloads). Legacy models unchanged.