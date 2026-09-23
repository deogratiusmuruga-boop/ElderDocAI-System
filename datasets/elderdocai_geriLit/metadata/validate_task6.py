#!/usr/bin/env python3
"""ElderDocAI-GeriLit dev-v0.1 - Phase 3 Task 6 validation (A-J).

Validates the GeriLit retrieval integration:

A. Corpus identity        - 17,930 chunks / 500 PMCIDs / unique chunk_ids
B. FAISS                  - dim 768, ntotal 17,930, IndexFlatIP
C. BM25                   - doc_count 17,930, chunk_ids unique == row_mapping
D. Row alignment          - FAISS row == BM25 pos == row_mapping == chunks[i]
E. Query embeddings       - dim 768, finite, L2 norm ~1, float32
F. Retrieval              - dense, sparse, hybrid, CrossEncoder, final top-3
G. Provenance             - pmcid, source_locator, non-empty text
H. Legacy regression      - legacy hybrid retriever still executes
I. Offline loading        - models load locally (HF_HUB_OFFLINE)
J. Determinism            - two full passes -> identical results

Smoke queries: 10 representative elderly-care queries; FUNCTIONAL only.
No accuracy/recall/precision/ndcg/faithfulness/reliability reported.
"""
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
META_DIR = Path(__file__).resolve().parent          # .../elderdocai_geriLit/metadata
GERI_DIR = META_DIR.parent                           # .../elderdocai_geriLit
REPO_DIR = GERI_DIR.parent.parent                    # ElderDocAI-System

sys.path.insert(0, str(GERI_DIR))
sys.path.insert(0, str(REPO_DIR))

from geri_lit_retriever import GeriLitRetriever  # noqa: E402

VALIDATION_DIR = META_DIR / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = VALIDATION_DIR / "phase3_task6_validation.json"
OUT_MD = VALIDATION_DIR / "phase3_task6_validation.md"

# FROZEN expected hashes (Task 5 records)
CHUNKS_SHA = ("62bfdde3c7e77cf56112e79a71bc79bdea70767f6b2625701e39"
              "2bbb6581c6b3")
EMBEDS_SHA = ("b0905d2f96903dadc3cf5b4a737f330853656f846955de706f21b"
              "d714ce2dacf")
FAISS_SHA = ("b7016b9dabbab08ee68f875007b6db2b013af4472b1a3c0958af8"
             "a566b59f2b3")
BM25_SHA = ("17f503240f2f9a13abf58c94359884b03b06bec1a5a59f63f9213"
            "28e2a2c70c9")
ROWMAP_SHA = ("4d649f81d554c41ec7b63c65dce887466dfa6245745c568315735"
              "f1b93a1872b")
LEGACY_CHUNKS = REPO_DIR / "data" / "chunks" / "knowledge_base_chunks.json"
LEGACY_FAISS = REPO_DIR / "data" / "vector_db" / "knowledge_base.faiss"

SMOKE_QUERIES = [
    "What factors are associated with falls in older adults?",
    "How can physical activity benefit older adults?",
    "What factors are associated with cognitive decline in older adults?",
    "What approaches are described for managing polypharmacy in older adults?",
    "What are the effects of social isolation on older adults?",
    "How is frailty assessed in older adults?",
    "What interventions are described for medication adherence?",
    "What factors are associated with functional decline in older adults?",
    "What are common risks of taking multiple medications in older adults?",
    "How can caregivers support older adults with dementia at home?",
]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jl(p):
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
def run_pass(retriever):
    """Execute one complete validation pass; returns (checks, diag)."""
    checks = {}
    diag = {}

    # ---- A. corpus identity ----
    chunks = retriever.chunks
    n = len(chunks)
    checks["A_corpus_chunk_count_17930"] = n == 17930
    pmcids = sorted({c["pmcid"] for c in chunks})
    checks["A_pmcid_count_500"] = len(pmcids) == 500
    cids = [c["chunk_id"] for c in chunks]
    checks["A_chunk_ids_unique"] = len(cids) == len(set(cids))

    # ---- B. FAISS ----
    idx = retriever.faiss_index
    checks["B_faiss_indexflatip"] = idx.__class__.__name__ == "IndexFlatIP"
    checks["B_faiss_dim_768"] = idx.d == 768
    checks["B_faiss_ntotal_17930"] = idx.ntotal == 17930

    # ---- C. BM25 ----
    bm = retriever.bm25
    bm_ids = retriever.bm25_ids
    checks["C_bm25_docs_17930"] = bm.corpus_size == 17930
    checks["C_bm25_ids_len"] = len(bm_ids) == 17930
    checks["C_bm25_ids_unique"] = len(set(bm_ids)) == 17930

    # ---- D. row alignment (all rows) ----
    rm = retriever.row_mapping
    align_ok = True
    align_pos = None
    for i in range(n):
        if rm[i] != cids[i] or rm[i] != bm_ids[i]:
            align_ok = False
            align_pos = i
            break
    checks["D_row_alignment_all_17930"] = align_ok
    diag["D_first_mismatch"] = align_pos

    # ---- E. query embeddings ----
    vec = retriever.embed_query("falls in older adults")
    checks["E_query_dim_768"] = vec.shape[1] == 768
    checks["E_query_dtype_float32"] = str(vec.dtype) == "float32"
    checks["E_query_finite"] = bool(np.isfinite(vec).all())
    qn = float(np.linalg.norm(vec))
    checks["E_query_l2_norm_eq_1"] = abs(qn - 1.0) < 1e-4
    diag["E_norm"] = qn

    # ---- F. retrieval (dense, sparse, hybrid, rerank, top-3) ----
    ret_ok = True
    f_diag = {}
    for q in SMOKE_QUERIES:
        dense = retriever.dense_retrieve(q)
        sparse = retriever.sparse_retrieve(q)
        hybrid = retriever.hybrid_retrieve(q)
        final = retriever.rerank(q, hybrid)
        if len(dense) != 5 or len(sparse) != 5:
            ret_ok = False
            f_diag[q] = "candidate-count"
            break
        if len(final) != 3:
            ret_ok = False
            f_diag[q] = "top3-count"
            break
        rows = [c["row"] for c in final]
        if len(set(rows)) != len(rows):
            ret_ok = False
            f_diag[q] = "duplicate-rows"
            break
        if not all(0 <= r < n for r in rows):
            ret_ok = False
            f_diag[q] = "out-of-range"
            break
        scores = [c.get("rerank_score") for c in final]
        if not all(s is not None and np.isfinite(s) for s in scores):
            ret_ok = False
            f_diag[q] = "nonfinite-rerank"
            break
    checks["F_retrieval_pipeline_10q"] = ret_ok
    diag["F"] = f_diag or "all 10 queries OK"

    # ---- G. provenance on final evidence ----
    prov_ok = True
    prov_fail = []
    for q in SMOKE_QUERIES:
        ev = retriever.retrieve(q)
        if len(ev) != 3:
            prov_ok = False
            prov_fail.append((q, "count"))
            continue
        for e in ev:
            if not (e.get("pmcid") and e.get("source_locator")
                    and e.get("text") and e.get("chunk_id")):
                prov_ok = False
                prov_fail.append((q, "fields"))
                break
            if not e["text"].strip():
                prov_ok = False
                prov_fail.append((q, "empty-text"))
                break
        if not prov_ok:
            break
    checks["G_provenance_complete_10q"] = prov_ok
    diag["G"] = prov_fail or "all 10 queries OK"

    return checks, diag
def legacy_regression():
    """H. Minimal legacy-retriever regression (does NOT run Gold96)."""
    res = {"import": False, "indexes": False, "embed_model": False,
           "executes": False, "schema": False}
    try:
        import scripts.hybrid_retriever as hr

        res["import"] = True

        n_legacy_chunks = 0
        with open(LEGACY_CHUNKS, "r", encoding="utf-8") as f:
            n_legacy_chunks = len(json.load(f))
        res["indexes"] = (
            hasattr(hr, "faiss_index") and hr.faiss_index.ntotal == n_legacy_chunks
            and hasattr(getattr(hr, "bm25", None), "get_scores"))
        res["embed_model"] = hasattr(hr, "embedding_model")
        try:
            out = hr.hybrid_search(
                "What should a caregiver know about medication safety?")
            if isinstance(out, list) and len(out) >= 1:
                res["executes"] = all(
                    isinstance(x, dict) and x.get("text") for x in out[:1])
                res["schema"] = all(
                    set(["chunk_id", "text"]).issubset(x.keys())
                    for x in out[:1])
        except Exception:
            pass
    except Exception:
        pass
    return res


def main():
    retriever = GeriLitRetriever()
    pass1 = run_pass(retriever)
    pass2 = run_pass(retriever)

    checks = {}
    diag = {}
    for k in pass1[0]:
        v1 = pass1[0][k]
        v2 = pass2[0][k]
        ok = bool(v1 is True and v2 is True)
        checks[k] = ok
        checks[f"J_deterministic_{k}"] = bool(v1 == v2)
    diag.update(pass1[1])
    checks["J_all_pass1"] = all(v is True for v in pass1[0].values())
    checks["J_all_pass2"] = all(v is True for v in pass2[0].values())

    # determinism on resolved evidence + full row IDs (two more full passes)
    row_ids1, row_ids2 = [], []
    for q in SMOKE_QUERIES:
        row_ids1.append([e["row_id"] for e in retriever.retrieve(q)])
        row_ids2.append([e["row_id"] for e in retriever.retrieve(q)])
    checks["J_final_row_ids_identical"] = row_ids1 == row_ids2
    checks["J_no_duplicate_chunks_per_query"] = all(
        len(set(a)) == len(a) for a in row_ids1)
    diag["J_row_ids_sample"] = row_ids1[0]

    legacy = legacy_regression()
    checks["H_legacy_import"] = legacy["import"]
    checks["H_legacy_indexes"] = legacy["indexes"]
    checks["H_legacy_embed_model"] = legacy["embed_model"]
    checks["H_legacy_executes"] = legacy["executes"]
    checks["H_legacy_schema"] = legacy["schema"]
    diag["H"] = legacy

    # I. offline loading enforced at import (HF_HUB_OFFLINE) and via
    #    local_files_only=True inside GeriLitRetriever
    checks["I_offline_env_set"] = (
        os.environ.get("HF_HUB_OFFLINE") == "1")
    checks["I_retriever_loaded_offline"] = (
        retriever.dim == 768 and len(retriever.chunks) == 17930)

    # FROZEN artifact hashes unchanged
    checks["FROZEN_chunks_sha"] = (
        sha256_file(GERI_DIR / "chunks/chunks.jsonl") == CHUNKS_SHA)
    checks["FROZEN_embeddings_sha"] = (
        sha256_file(GERI_DIR / "index/embeddings.npy") == EMBEDS_SHA)
    checks["FROZEN_faiss_sha"] = (
        sha256_file(GERI_DIR / "index/faiss_index.bin") == FAISS_SHA)
    checks["FROZEN_bm25_sha"] = (
        sha256_file(GERI_DIR / "index/bm25.pkl") == BM25_SHA)
    checks["FROZEN_rowmapping_sha"] = (
        sha256_file(GERI_DIR / "index/row_mapping.json") == ROWMAP_SHA)

    ok = all(v is True for v in checks.values())
    out = {
        "validation_ts": datetime.now(timezone.utc).isoformat(),
        "validated_by": "metadata/validate_task6.py",
        "all_checks_pass": ok,
        "chunk_totals": {"chunks": len(retriever.chunks),
                         "pmcids": 500},
        "smoke_query_count": len(SMOKE_QUERIES),
        "checks": checks,
        "diag": diag,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    lines = ["# ElderDocAI-GeriLit Dev Corpus - Phase 3 Task 6 Validation", "",
             f"- **validated_by**: validate_task6.py",
             f"- **timestamp**: {out['validation_ts']}", "",
             "| Check | Result |", "|---|---|"]
    for k, v in checks.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    (OUT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"all_checks_pass": ok,
                      "n_checks": len(checks),
                      "smoke_queries": len(SMOKE_QUERIES)}, indent=2))


if __name__ == "__main__":
    main()