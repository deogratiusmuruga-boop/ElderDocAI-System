#!/usr/bin/env python3
"""ElderDocAI-GeriLit dev-v0.1 - Phase 3 Task 5: Embedding generation.

Builds L2-normalized BGE-base-en-v1.5 embeddings for the frozen
chunks.jsonl corpus (17,930 chunks / 500 PMCIDs), preserving exact row
order so that FAISS row == BM25 position == chunk-manifest position.

Outputs (under index/):
  embeddings.npy        float32 (N x D), L2-normalized
  row_mapping.json      list[N] of chunk_id in row order
  embedding_manifest.json  model/config/hashes/statistics

Deterministic by construction (single source of ordering); embeds the
exact chunk["text"] field (same convention as the legacy pipeline).
Overly long texts are truncated to the model's max_seq_length at embed
time (512 tokens for bge-base-en-v1.5); BM25 uses the full text.

DO NOT modify any Task 2/3/4 artifacts.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parent
CHUNKS = BASE / "chunks" / "chunks.jsonl"
INDEX_DIR = BASE / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBED_NPY = INDEX_DIR / "embeddings.npy"
ROW_MAPPING = INDEX_DIR / "row_mapping.json"
MANIFEST = INDEX_DIR / "embedding_manifest.json"
REF_MANIFEST = BASE / "metadata" / "embedding_manifest_reference.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_chunks():
    recs = []
    with open(CHUNKS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def main():
    t0 = time.time()
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS.name}")

    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]
    assert len(set(chunk_ids)) == len(chunk_ids), "non-unique chunk_ids"

    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    print(f"Model={MODEL_NAME} dim={dim} max_seq_length={model.max_seq_length}")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=128,
    )
    embeddings = embeddings.astype(np.float32)
    assert embeddings.shape == (len(chunks), dim)

    np.save(EMBED_NPY, embeddings)
    with open(ROW_MAPPING, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f)

    norms = np.linalg.norm(embeddings, axis=1)
    finite = bool(np.isfinite(embeddings).all())
    manifest = {
        "corpus_version": "ElderDocAI-GeriLit-dev-v0.1",
        "task": "phase3_task5_embeddings",
        "model": MODEL_NAME,
        "model_revision": getattr(model, "model_card_data", None) and
                          getattr(model, "_model_card_text", None) and None,
        "dimension": int(dim),
        "embedding_count": int(embeddings.shape[0]),
        "dtype": str(embeddings.dtype),
        "normalization": "L2 (normalize_embeddings=True)",
        "device": str(getattr(model, "device", "auto")),
        "batch_size": 128,
        "max_seq_length": int(model.max_seq_length),
        "text_source": "chunk['text'] exact (whitespace-normalized at chunking)",
        "embedding_norm_min": float(norms.min()),
        "embedding_norm_max": float(norms.max()),
        "embedding_norm_mean": float(norms.mean()),
        "all_finite": finite,
        "embeddings_npy_sha256": sha256_file(EMBED_NPY),
        "row_mapping_sha256": sha256_file(ROW_MAPPING),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    BASE.joinpath("metadata").mkdir(parents=True, exist_ok=True)
    if not REF_MANIFEST.exists():
        with open(REF_MANIFEST, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

    print(json.dumps(manifest, indent=2))
    print("Embeddings written to", EMBED_NPY)


if __name__ == "__main__":
    main()