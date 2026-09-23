#!/usr/bin/env python3
"""ElderDocAI-GeriLit dev-v0.1 - Phase 3 Task 5: FAISS index construction.

Builds a cosine-compatible FAISS index (IndexFlatIP over L2-normalized
embeddings) from embedding_manifest + embeddings.npy produced by
build_embeddings.py.

Strictly reuses the legacy convention (IndexFlatIP on normalized vectors),
with a deterministic row mapping preserved from the chunk manifest.

Outputs (under index/):
  faiss_index.bin     serialized IndexFlatIP (float32, N x D)
  faiss_manifest.json config/hashes/statistics
"""
import hashlib
import json
import time
from pathlib import Path

import faiss
import numpy as np

BASE = Path(__file__).resolve().parent
INDEX_DIR = BASE / "index"
EMBED_NPY = INDEX_DIR / "embeddings.npy"
FAISS_BIN = INDEX_DIR / "faiss_index.bin"
FAISS_MANIFEST = INDEX_DIR / "faiss_manifest.json"
REF_MANIFEST = BASE / "metadata" / "faiss_manifest_reference.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    t0 = time.time()
    embeddings = np.load(EMBED_NPY).astype(np.float32)
    dim = embeddings.shape[1]
    n = embeddings.shape[0]
    print(f"Embeddings: {embeddings.shape}, dim={dim}")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_BIN))

    manifest = {
        "corpus_version": "ElderDocAI-GeriLit-dev-v0.1",
        "task": "phase3_task5_faiss",
        "index_type": "IndexFlatIP",
        "metric": "INNER_PRODUCT (cosine because embeddings are L2-normalized)",
        "dimension": int(dim),
        "vector_count": int(index.ntotal),
        "dtype": "float32",
        "source_embeddings_npy": EMBED_NPY.name,
        "embeddings_npy_sha256": sha256_file(EMBED_NPY),
        "faiss_index_bin_sha256": sha256_file(FAISS_BIN),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    with open(FAISS_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    if not REF_MANIFEST.exists():
        BASE.joinpath("metadata").mkdir(parents=True, exist_ok=True)
        with open(REF_MANIFEST, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

    print(json.dumps(manifest, indent=2))
    print("FAISS index written to", FAISS_BIN, "ntotal=", index.ntotal)


if __name__ == "__main__":
    main()