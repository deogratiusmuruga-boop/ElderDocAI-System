# ElderDocAI-GeriLit dev-v0.1 — Phase 3 Task 5
## Embedding Generation + FAISS/BM25 Index Construction — Technical Report

> **Scope:** standalone index construction only. No CrossEncoder run, no hybrid
> retrieval integration, no LLM, no benchmark evaluation, no reliability/gating,
> no production RAG changes, no Task 2/3/4 modifications.

## 1. Inputs
- Corpus: `ElderDocAI-GeriLit-dev-v0.1` (frozen), 500 accepted PMC OA articles.
- Chunks: `chunks/chunks.jsonl` — 17,930 chunks (SHA-256
  `62bfdde3c7e77cf56112e79a71bc79bdea70767f6b2625701e392bbb6581c6b3`, unchanged).
- Evidence metadata: Task-3 artifacts unchanged (jats_enriched_metadata,
  jats_section_inventory, jats_structural_audit, evidence_type_enrichment).

## 2. Environment
- host: Windows / venv `C:\Users\chosun\.venv\Scripts\python.exe`
- numpy 2.5.2, torch 2.11.0+cu128 (CUDA available), sentence-transformers,
  faiss 1.15.0, rank_bm25
- Embedding model `BAAI/bge-base-en-v1.5` — resolved from the **local Hugging
  Face cache** (offline-capable; verified load with `local_files_only=True`),
  199 transformer layers-load, max_seq_length 512, output dim 768.

## 3. Bare pipeline (commands, in order)
```
python build_embeddings.py
python build_faiss_index.py   # "build_faiss_index.py"
python build_bm25_index.py    # "build_bm25_index.py"
python validate_task5.py
```
Artifacts and manifests are written to `index/`.

## 4. Embeddings
- 17,930 vectors x 768 float32 (shape (17930, 768)).
- L2-normalized (norms: min 0.9999999, max 1.0000001, mean 1.0).
- Text embedded = exact chunk `text` (whitespace-normalized at chunking time).
- Embedding time (CUDA, batch 128): ~2 min (120–124 s incl. load). GPU: cuda:0.

```
embeddings.npy SHA-256 = b0905d2f96903dadc3cf5b4a737f330853656f846955de706f21bd714ce2dacf
row_mapping.json SHA-256         4d649f81d554c41ec7b63c65dce887466dfa6245745c568315735f1b93a1872b
```

Determinism (two runs): byte-identical hashes across two runs (Determinism check run-2
confirms). Statistics reproducibility validated: chunks_total 17930.

| Manifest file SHA-256:
```
faiss_index.bin  b7016b9dabbab08ee68f875007b6db2b013af4472b1a3c0958af8a566b59f2b3
bm25.pkl         17f503240f2f9a13abf58c94359884b03b06bec1a5a59f63f921328e2a2c70c9
```

## 5. FAISS
- IndexFlatIP(768), INNER_PRODUCT (= cosine on L2-normalized vectors).
- ntotal = 17,930. Determinism: byte-identical across runs (`b7016b9…` twice).
- IP ordering verified equivalent to cosine ordering on sample (check pass).

## 6. BM25
- rank_bm25.BM25Okapi; tokenizer `text.lower().split()` (legacy convention).
- doc_count 17,930; chunk_ids 17,930 unique. Determinism: byte-identical
  (`17f50324…` twice).

## 7. Row alignment (= indexed row == chunk manifest position)
- row_mapping.json[ i ] == chunks.jsonl[ i ] == bm25 chunk_ids[ i ] for all i
  (0..17929). First mismatch: none.
- FAISS index i built from embeddings.npy row i; row alignment therefore holds
  for FAISS row i ↔ BM25 position i (documented single source of ordering.

## 8. Provenance recovery
- For every chunk, recovered from indexed row i → chunk_id → pmcid, source_locator,
  section_title/section_id, region, evidence_type. Full-coverage check pass (31/31).
  1,946 chunks from untitled `<sec>` elements carry pmcid+source_locator;
  provenance-complete (diagnostic, not failure).

## 9. Smoke test (technical only, not quality)
- query "anticoagulation management in frail elderly patients"
  FAISS returns 3 in-range rows; scores sorted desc; BM25 returns 3 in-range rows;
  all finite. No accuracy / benchmark claims made.

## 10. Determinism summary
- embeddings, faiss index, bm25 all byte-identical across two runs.
- validation output deterministic: 31 checks pass on both runs.

## 10b. Timing
- Embedding ~120s/run (CUDA). FAISS build <1s; BM25 build ~2s.

## 11. Integrity
- Task-2 manifest, Task-3 4 json artifacts, Task-4 chunk_statistics + chunk QA —
  hashes verified unchanged.

## 12. Files
Build/validation scripts (new, untracked).
Index artifacts under index/ (new, untracked).
Validation outputs metadata/phase3_task5_validation.{json,md} and manifests.

No commits, no push.