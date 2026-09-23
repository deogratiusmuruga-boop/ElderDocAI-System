#!/usr/bin/env python3
"""ElderDocAI-GeriLit dev-v0.1 - Phase 3 Task 5: BM25 index construction.

Builds a BM25Okapi sparse index over the frozen chunks.jsonl corpus,
reusing the legacy tokenizer convention (text.lower().split()).

Row order preserved so that BM25 position == FAISS row == chunk-manifest
position.

Outputs (under index/):
  bm25.pkl                   pickle {"bm25": BM25Okapi, "chunk_ids": [...]}
  bm25_manifest.json         config/hashes/statistics
"""
import hashlib
import json
import pickle
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

BASE = Path(__file__).resolve().parent
INDEX_DIR = BASE / "index"
CHUNKS = BASE / "chunks" / "chunks.jsonl"
BM25_PKL = INDEX_DIR / "bm25.pkl"
BM25_MANIFEST = INDEX_DIR / "bm25_manifest.json"
REF_MANIFEST = BASE / "metadata" / "bm25_manifest_reference.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tokenize(text):
    return text.lower().split()


def main():
    t0 = time.time()
    chunks = []
    with open(CHUNKS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    chunk_ids = [c["chunk_id"] for c in chunks]
    print(f"Loaded {len(chunks)} chunks")

    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_PKL, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)

    manifest = {
        "corpus_version": "ElderDocAI-GeriLit-dev-v0.1",
        "task": "phase3_task5_bm25",
        "implementation": "rank_bm25.BM25Okapi",
        "tokenizer": "text.lower().split() (legacy convention)",
        "document_count": len(chunk_ids),
        "unique_chunk_ids": len(set(chunk_ids)),
        "bm25_pkl_sha256": sha256_file(BM25_PKL),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    with open(BM25_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    if not REF_MANIFEST.exists():
        BASE.joinpath("metadata").mkdir(parents=True, exist_ok=True)
        with open(REF_MANIFEST, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

    print(json.dumps(manifest, indent=2))
    print("BM25 index written to", BM25_PKL)


if __name__ == "__main__":
    main()