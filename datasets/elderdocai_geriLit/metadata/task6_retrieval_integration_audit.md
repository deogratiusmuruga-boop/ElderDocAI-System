# ElderDocAI-GeriLit — Phase 3 Task 6
## Retrieval Integration Audit (existing implementation, read-only)

> Objective of this audit: document the **existing** retrieval architecture and the
> minimum integration surface required to connect the frozen GeriLit FAISS/BM25
> indexes into ElderDocAI's retrieval flow **reversibly**, without touching the
> legacy six-PDF path.

## 1. Existing retrieval files (audited, NOT modified)

| File | Role |
|---|---|
| `scripts/hybrid_retriever.py` | Dense + sparse + hybrid + rerank orchestrator (legacy KB) |
| `scripts/reranker.py` | Standalone CrossEncoder rerank helper (legacy KB) |
| `scripts/evidence_aggregation.py` | FAISS results → evidence objects (authority-mapped) |
| `scripts/authority_mapping.py` | Organization-based authority classification |
| `scripts/rag_chat.py` | `prepare_evidence()` converts retriever chunks → evidence dicts; LLM call |
| `scripts/carebuddy_service.py` | API service layer; consumes evidence `source_document` |
| `api/main.py` | FastAPI `/ask` endpoint (unchanged) |

## 2. Existing pipeline (as implemented)

```
Query
  → embedding_model.encode(query, normalize_embeddings=True)   # BAAI/bge-base-en-v1.5
  → FAISS IndexFlatIP.search(top_k=5)                          # dense candidates
  → BM25Okapi.get_scores(query.lower().split()) top-5          # sparse candidates
  → merge by chunk_id; hybrid = 0.6*(d/s_max) + 0.4*(b/b_max)
  → sort by hybrid_score desc → take top-5
  → CrossEncoder(ms-marco-MiniLM-L-6-v2).predict(pairs) → rerank
  → final top-3                                                  # FINAL_RESULTS=3
```

Configuration observed:
- `EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"`
- `RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"`
- `TOP_K_VECTOR = 5`, `TOP_K_BM25 = 5`, `TOP_K_RERANK = 5`, `FINAL_RESULTS = 3`
- Hybrid weights `0.6 * dense_norm + 0.4 * sparse_norm` (max-normalized)
- Tokenizer: `text.lower().split()` (BM25)
- Query embedding: `encode(..., normalize_embeddings=True).astype(np.float32)`

## 3. Existing assumptions / coupling points

1. **Row identity**: FAISS row `i` == `chunks[i]` == BM25 position `i` (legacy build
   aligned all three by construction). `hybrid_retriever.semantic_search` indexes
   results by `chunk_id` and later by list index.
2. **Chunk schema (legacy)**: `{category, chunk_id, document_id, document_type,
   language, last_updated, organization, source_document, text, title}`.
   Required evidence keys after `prepare_evidence`: `chunk_id`, `source_document`,
   `category`/`document_category`, `authority_score`, `rerank_score`/`dense_score`,
   `text`.
3. **Path coupling**: `hybrid_retriever.py` hard-codes `data/chunks/...` and
   `data/vector_db/...`; `rag_chat.py` imports `hybrid_search` directly.
4. **Evidence schema** (`prepare_evidence` output item):
   `{chunk_id, source_document, document_category, authority_score,
     similarity_score, text}` — the reliability evaluator and prompt builder
   consume exactly these keys.
5. **Chunk-ID uniqueness**: legacy `chunk_id` values restart per document (known
   collision risk). GeriLit chunk IDs are globally unique (Task 4 validated).
6. **Models are loaded at module import time** in `hybrid_retriever.py` — a
   legacy-regression import will also load models (acceptable for a test).

## 4. Minimum integration surface for GeriLit (reversible, additive)

1. New standalone module `geri_lit_retriever.py` under `datasets/elderdocai_geriLit/`
   (no legacy file edits required for Task 6).
2. Load frozen indexes: `index/faiss_index.bin`, `index/bm25.pkl`,
   `index/row_mapping.json`, `chunks/chunks.jsonl`, plus manifest hashes.
3. Query embed: same `BAAI/bge-base-en-v1.5`, `normalize_embeddings=True`,
   float32 → 768-d → `faiss.search(top_k=5)`.
4. BM25: `rank_bm25.BM25Okapi` with `text.lower().split()` (same tokenizer) → top-5.
5. Row mapping: `FAISS row i → row_mapping[i] → chunk_id → chunks.jsonl record`;
   identical mapping for BM25 (`bm25 chunk_ids[i]`). Never `FAISS row == chunk_id`.
6. Hybrid fusion: replicate legacy max-normalized `0.6*d + 0.4*s` fusion, with a
   **deterministic tie-break** (stable sort by chunk_id) — deterministic adaptation
   is required because raw FAISS IP and BM25 magnitude scales differ.
7. CrossEncoder: locally cached `cross-encoder/ms-marco-MiniLM-L-6-v2` (revision
   `233902d25c440f23af6f7d6e94d2946bac0bee0a`), offline/local-only load; rerank
   candidate set (union of top-5 dense + top-5 sparse), return final top-3.
8. Provenance: every result returns `chunk_id, pmcid, source_locator, section_id,
   section_title, region, content_type, evidence_type, text` from the frozen chunk
   record (no reconstruction from chunk-id string parsing).
9. Evidence-compat adapter: emit `source_document` (e.g., `PMCID (Journal, year)`)
   and `rerank_score`/`similarity_score` so `prepare_evidence`-style consumers can
   read GeriLit results without modification.
10. Legacy path untouched: `scripts/hybrid_retriever.py`, `rag_chat.py`,
    `carebuddy_service.py`, `api/` remain as-is (Task 6 adds a parallel retriever).

## 5. Reversibility

- No legacy file is modified in Task 6; removal of GeriLit integration = delete the
  new module + its reports. The six-PDF default path is untouched.

## 6. Not in scope (confirmed)

- No Gold96 / PubMedQA / TREC-CDS / BioASQ / MIMIC evaluation.
- No reliability gating, care-state, adaptive-assistance, `api/main.py`,
  frontend, manuscript, or frozen Task 2–5 artifact changes.
- No benchmark-quality metrics computed or claimed (smoke tests only).