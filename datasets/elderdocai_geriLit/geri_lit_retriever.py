#!/usr/bin/env python3
"""ElderDocAI-GeriLit dev-v0.1 - Phase 3 Task 6: GeriLit retriever.

Standalone, reversible retrieval layer over the FROZEN GeriLit indexes:

    Query
      -> BGE query embedding (768-d, L2-normalized)
      -> FAISS IndexFlatIP top-5 (dense)
      -> BM25Okapi top-5 (sparse)
      -> hybrid fusion (0.6 dense + 0.4 sparse, max-normalized, legacy style)
      -> CrossEncoder rerank (top-5 candidates)
      -> final top-3 evidence with complete provenance

Frozen artifacts (Task 5) are READ ONLY and are never rebuilt or rewritten:

    chunks/chunks.jsonl
    index/embeddings.npy       (source vectors; not reloaded here)
    index/faiss_index.bin
    index/bm25.pkl
    index/row_mapping.json
    index/*_manifest.json

Row mapping (critical):  FAISS row i == BM25 position i == row_mapping[i]
== chunks.jsonl[i] (verified in Task 5 for all 17,930 rows). Results are
resolved via row_mapping -> chunk_id -> chunk record. NEVER assume
FAISS row == chunk_id.

Design notes
- Query embedding: same model and normalization as legacy
  (BAAI/bge-base-en-v1.5, normalize_embeddings=True).
- BM25: rank_bm25.BM25Okapi with legacy tokenizer text.lower().split().
  The stored BM25 index is used directly; it is NOT rebuilt.
- Fusion replicates the legacy max-normalized 0.6/0.4 scheme with a
  deterministic tie-break (stable sort by chunk_id).
- CrossEncoder: locally cached cross-encoder/ms-marco-MiniLM-L-6-v2,
  loaded offline (HF_HUB_OFFLINE / local_files_only). No downloads.
- No MIMIC context is joined here; that is a later task.

Scope: smoke-test / integration functionality ONLY. No benchmark metrics
(accuracy/recall/ndcg/etc.) are computed or claimed.
"""
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import faiss
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

BASE = Path(__file__).resolve().parent

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DENSE_K = 5
SPARSE_K = 5
RERANK_K = 5
FINAL_K = 3
DENSE_W = 0.6
SPARSE_W = 0.4
class GeriLitRetriever:
    """Load-once retriever over the frozen GeriLit FAISS + BM25 indexes."""

    def __init__(self, base_dir=None, offline=True):
        base = Path(base_dir) if base_dir else BASE
        self.base = base
        self.chunks_file = base / "chunks" / "chunks.jsonl"
        self.faiss_file = base / "index" / "faiss_index.bin"
        self.bm25_file = base / "index" / "bm25.pkl"
        self.row_mapping_file = base / "index" / "row_mapping.json"

        self.chunks = self._load_chunks()
        self.row_mapping = json.loads(
            self.row_mapping_file.read_text(encoding="utf-8"))
        self.faiss_index = faiss.read_index(str(self.faiss_file))
        with open(self.bm25_file, "rb") as f:
            payload = pickle.load(f)
        self.bm25 = payload["bm25"]
        self.bm25_ids = payload["chunk_ids"]

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL, local_files_only=offline)
        self.reranker = CrossEncoder(
            RERANK_MODEL, local_files_only=offline)
        try:
            self.dim = int(self.embedding_model.get_embedding_dimension())
        except AttributeError:  # older sentence-transformers
            self.dim = int(
                self.embedding_model.get_sentence_embedding_dimension())

        assert len(self.row_mapping) == len(self.chunks)
        assert self.bm25.corpus_size == len(self.chunks)

    # ---------------------------------------------------------------
    # Loading
    # ---------------------------------------------------------------
    def _load_chunks(self):
        recs = []
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        return recs

    # ---------------------------------------------------------------
    # Query embedding
    # ---------------------------------------------------------------
    def embed_query(self, query):
        vec = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        return vec

    # ---------------------------------------------------------------
    # Dense retrieval
    # ---------------------------------------------------------------
    def dense_retrieve(self, query, k=DENSE_K):
        vec = self.embed_query(query)
        scores, indices = self.faiss_index.search(vec, k)
        out = []
        for score, row in zip(scores[0], indices[0]):
            row = int(row)
            if row == -1:
                continue
            out.append((float(score), row))
        return out

    # ---------------------------------------------------------------
    # Sparse retrieval
    # ---------------------------------------------------------------
    def sparse_retrieve(self, query, k=SPARSE_K):
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        order = np.argsort(scores)[::-1][:k]
        out = []
        for row in order:
            row = int(row)
            out.append((float(scores[row]), row))
        return out

    # ---------------------------------------------------------------
    # Resolve row -> chunk record
    # ---------------------------------------------------------------
    def row_to_chunk(self, row):
        chunk_id = self.row_mapping[row]
        return chunk_id, self.chunks[row]

    # ---------------------------------------------------------------
    # Hybrid fusion (legacy-style max-normalized 0.6/0.4, deterministic)
    # ---------------------------------------------------------------
    def hybrid_retrieve(self, query, dense_k=DENSE_K, sparse_k=SPARSE_K):
        dense = self.dense_retrieve(query, dense_k)
        sparse = self.sparse_retrieve(query, sparse_k)

        dense_max = max((s for s, _ in dense), default=1.0) or 1.0
        sparse_max = max((s for s, _ in sparse), default=1.0) or 1.0

        item_map = {}
        for s, row in dense:
            dense_norm = s / dense_max
            item_map.setdefault(row, {"dense": 0.0, "sparse": 0.0})
            item_map[row]["dense"] = dense_norm
        for s, row in sparse:
            sparse_norm = s / sparse_max
            item_map.setdefault(row, {"dense": 0.0, "sparse": 0.0})
            item_map[row]["sparse"] = sparse_norm

        combos = []
        for row, sc in item_map.items():
            hybrid = DENSE_W * sc["dense"] + SPARSE_W * sc["sparse"]
            combos.append((hybrid, -row, row, sc["dense"], sc["sparse"]))
        # hybrid desc, then row asc via (-row) for deterministic ordering
        combos.sort(key=lambda x: (x[0], x[1]))
        top = combos[:RERANK_K]
        return [
            {
                "row": row,
                "hybrid_score": float(h),
                "dense_score": float(dn),
                "sparse_score": float(sn),
            }
            for h, _nr, row, dn, sn in top
        ]

    # ---------------------------------------------------------------
    # CrossEncoder rerank
    # ---------------------------------------------------------------
    def rerank(self, query, candidates):
        if not candidates:
            return []
        pairs = [
            (query, self.chunks[c["row"]]["text"])
            for c in candidates
        ]
        scores = self.reranker.predict(pairs)
        for c, sc in zip(candidates, scores):
            c["rerank_score"] = float(sc)
        order = sorted(
            candidates,
            key=lambda c: (c["rerank_score"], -c["row"]),
            reverse=True,
        )
        return order[:FINAL_K]
# ---------------------------------------------------------------
    # Full pipeline: query -> top-3 evidence
    # ---------------------------------------------------------------
    def retrieve(self, query):
        """Return final top-K evidence items with complete provenance."""
        hybrid = self.hybrid_retrieve(query)
        final = self.rerank(query, hybrid)
        evidence = []
        for c in final:
            row = c["row"]
            chunk_id, chunk = self.row_to_chunk(row)
            evidence.append(
                {
                    "row_id": row,
                    "chunk_id": chunk_id,
                    "pmcid": chunk.get("pmcid"),
                    "pmid": chunk.get("pmid"),
                    "doi": chunk.get("doi"),
                    "title": chunk.get("title"),
                    "journal": chunk.get("journal"),
                    "publication_year": chunk.get("publication_year"),
                    "region": chunk.get("region"),
                    "section_id": chunk.get("section_id"),
                    "section_title": chunk.get("section_title"),
                    "content_type": chunk.get("content_type"),
                    "evidence_type": chunk.get("evidence_type"),
                    "source_locator": chunk.get("source_locator"),
                    "text": chunk.get("text"),
                    "dense_score": c.get("dense_score"),
                    "sparse_score": c.get("sparse_score"),
                    "hybrid_score": c.get("hybrid_score"),
                    "rerank_score": c.get("rerank_score"),
                    "source_document": (
                        f"{chunk.get('pmcid')} "
                        f"({chunk.get('journal', '')}, "
                        f"{chunk.get('publication_year', '')})"
                        .strip()
                    ),
                }
            )
        return evidence


if __name__ == "__main__":
    r = GeriLitRetriever()
    print("GeriLit retriever loaded:",
          f"chunks={len(r.chunks)} dim={r.dim} "
          f"faiss_ntotal={r.faiss_index.ntotal} "
          f"bm25_docs={r.bm25.corpus_size}")
    for q in [
        "What factors are associated with falls in older adults?",
        "How can physical activity benefit older adults?",
    ]:
        print("\nQUERY:", q)
        for e in r.retrieve(q):
            print(" -", e["chunk_id"], "|", e["pmcid"], "|",
                  e["region"], "|", (e["section_title"] or "untitled"),
                  "| rerank=%.4f" % e["rerank_score"])