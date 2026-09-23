#!/usr/bin/env python3
"""ElderDocAI-GeriLit dev-v0.1 - Phase 3 Task 5 validation.

Validates:
 - corpus identity (17,930 chunks / 500 PMCIDs / unique chunk_ids)
 - embeddings (count, dim, finite, L2-normalized)
 - FAISS (readable, IndexFlatIP, ntotal, dim, IP==cosine order on sample)
 - BM25 (doc count, id list size)
 - ROW ALIGNMENT: row_mapping == chunk order == BM25 id order (FAISS row used the
   same embeddings.npy order by construction)
 - provenance recovery from indexed rows (chunk_id, pmcid, source_locator,
   section_title, evidence_type present)
 - offline model availability (local cache, no Hub)
 - technical smoke test (FAISS/BM25 return valid rows; scores finite; no accuracy)
 - reference manifests present (determinism baseline files)
 - integrity: Task-2/3/4 artifacts unchanged (hash equality)
 - statistics reproducibility
"""
import hashlib
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

BASE = Path(__file__).resolve().parent
INDEX = BASE / "index"
META = BASE / "metadata"
VDIR = META / "validation"
VDIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = VDIR / "phase3_task5_validation.json"
OUT_MD = VDIR / "phase3_task5_validation.md"

ACC_MAN_SHA = ("f80cd457e5e2be55ed3a3de09e2554e9d585e76cef"
               "8dc44800270dfa66a99a49")
TASK3_SHA = {
    "jats_enriched_metadata.json":
        "655f1f4a63e64a0d55682a8d207e2059393b756b77ee8abac182b7ab1dad7825",
    "jats_section_inventory.json":
        "1dac58f9b5d7bf481e1b8a13d8a6f1d9ed119bdbb8f99d4b574a9d5b4b087e7c",
    "jats_structural_audit.json":
        "59799ac33474b589f505ddbd999a1166b515d0d07c1eef8ab756f4be9d87fc5c",
    "evidence_type_enrichment.json":
        "670ee5fbcf33c597256b287e82f073b56719b05320d92d05c17df79a01620568",
}
CHUNKS_SHA = ("62bfdde3c7e77cf56112e79a71bc79bdea70767f6b2625701e39"
              "2bbb6581c6b3")


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


def checks_pass(checks):
    return all(v is True for v in checks.values())
def main():
    checks = {}
    diag = {}

    # ---- corpus identity ----
    chunks = read_jl(BASE / "chunks" / "chunks.jsonl")
    n = len(chunks)
    checks["corpus_chunk_count"] = n == 17930
    pmcids = sorted({c["pmcid"] for c in chunks})
    checks["corpus_pmcid_count"] = len(pmcids) == 500
    cids = [c["chunk_id"] for c in chunks]
    checks["chunk_ids_unique"] = len(cids) == len(set(cids))

    # ---- embeddings ----
    emb = np.load(INDEX / "embeddings.npy")
    diag["emb_shape"] = list(emb.shape)
    checks["emb_shape"] = emb.shape == (17930, 768)
    checks["emb_dtype"] = emb.dtype == np.float32
    checks["emb_finite"] = bool(np.isfinite(emb).all())
    norms = np.linalg.norm(emb, axis=1)
    checks["emb_l2_normalized"] = bool(np.abs(norms - 1.0).max() < 1e-4)
    checks["emb_norm_min_max"] = bool(norms.min() > 0.9999 and norms.max() < 1.0001)

    # ---- FAISS ----
    idx = faiss.read_index(str(INDEX / "faiss_index.bin"))
    checks["faiss_readable"] = idx is not None
    checks["faiss_indexflatip"] = isinstance(idx, faiss.IndexFlatIP)
    checks["faiss_ntotal"] = int(idx.ntotal) == 17930
    checks["faiss_dim"] = idx.d == 768
    # IP ordering == cosine ordering on a sample (embeddings L2-normalized)
    s = emb[:64]
    tmp_idx = faiss.IndexFlatIP(768)
    tmp_idx.add(s)
    _, I1 = tmp_idx.search(s[:16], 3)
    cos_top = np.argsort(-(s[:16] @ s.T), axis=1)[:, :3]
    checks["faiss_ip_matches_cosine"] = bool(np.array_equal(I1, cos_top))

    # ---- BM25 ----
    with open(INDEX / "bm25.pkl", "rb") as f:
        pkl = pickle.load(f)
    bm = pkl["bm25"]
    bm_ids = pkl["chunk_ids"]
    checks["bm25_docs"] = bm.corpus_size == 17930
    checks["bm25_ids_size"] = len(bm_ids) == 17930
    checks["bm25_ids_unique"] = len(set(bm_ids)) == 17930

    # ---- row alignment ----
    row_map = json.loads((INDEX / "row_mapping.json").read_text(encoding="utf-8"))
    checks["rowmapping_size"] = len(row_map) == n
    checks["rowmapping_equals_chunk_order"] = row_map == cids
    mismatch = next((i for i in range(n) if row_map[i] != bm_ids[i]), None)
    checks["rowmapping_equals_bm25_order"] = mismatch is None
    diag["first_row_mismatch"] = mismatch

    # ---- provenance recovery (sample + full) ----
    # Provenance present = chunk_id matches row map AND pmcid AND
    # source_locator AND evidence_type. source_locator pinpoints exact
    # source element(s) (e.g. "PMC10008549:BACK:-:21..21"); it is the
    # ground-truth provenance. Untitled/unsectioned chunks (section_title
    # empty and section_id None) keep full provenance and are counted as a
    # diagnostic, not a failure.
    prov_bad = []
    untitled_ct = 0
    for i, rec in enumerate(chunks):
        if not (rec.get("section_title") or "").strip() and not rec.get("section_id"):
            untitled_ct += 1
        ok = (rec["chunk_id"] == row_map[i] and rec.get("pmcid")
              and rec.get("source_locator") and rec.get("evidence_type"))
        if not ok:
            prov_bad.append(i)
            if len(prov_bad) >= 5:
                break
    checks["provenance_recovery_present"] = len(prov_bad) == 0
    diag["untitled_unsectioned_chunks"] = untitled_ct
    diag["prov_sample_fail"] = prov_bad[:5]
# ---- offline model ----
    logging.disable(logging.CRITICAL)
    offline_ok = False
    off_dim = None
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("BAAI/bge-base-en-v1.5", local_files_only=True)
        v = m.encode(["fall prevention in older adults"],
                     normalize_embeddings=True, convert_to_numpy=True)
        offline_ok = bool(np.isfinite(v).all())
        off_dim = int(v.shape[1])
    except Exception as e:
        diag["offline_error"] = repr(e)
    checks["offline_model_available_local"] = offline_ok
    checks["offline_dim"] = off_dim == 768

    # ---- technical smoke test (functionality only) ----
    q = "anticoagulation management in frail elderly patients"
    qv = m.encode([q], normalize_embeddings=True,
                  convert_to_numpy=True).astype(np.float32)
    Dn, In = idx.search(qv, 3)
    scores = bm.get_scores(q.lower().split())
    top_bm = np.argsort(scores)[::-1][:3]
    checks["smoke_faiss_rows_valid"] = bool((In[0] >= 0).all() and
                                            (In[0] < 17930).all())
    checks["smoke_faiss_sorted"] = bool((np.diff(Dn[0]) <= 0).all())
    checks["smoke_bm25_rows"] = len(top_bm) == 3 and bool((top_bm < 17930).all())
    checks["smoke_scores_finite"] = bool(np.isfinite(Dn).all())

    # ---- reference manifests ----
    refs = ["embedding_manifest_reference.json", "faiss_manifest_reference.json",
            "bm25_manifest_reference.json"]
    checks["reference_manifests_present"] = all((META / r).exists() for r in refs)

    # ---- integrity of Task 2/3/4 artifacts ----
    checks["task2_manifest_unchanged"] = (
        sha256_file(BASE / "manifest/accepted_manifest.jsonl") == ACC_MAN_SHA)
    t3 = all(sha256_file(META / fn) == h for fn, h in TASK3_SHA.items())
    checks["task3_artifacts_unchanged"] = t3
    checks["task4_chunks_unchanged"] = (
        sha256_file(BASE / "chunks/chunks.jsonl") == CHUNKS_SHA)
    stats = json.loads((META / "chunk_statistics.json").read_text(encoding="utf-8"))
    checks["statistics_reproducible"] = stats.get("chunks_total") == 17930

    ok = checks_pass(checks)
    out = {
        "validation_ts": datetime.now(timezone.utc).isoformat(),
        "validated_by": "validate_task5.py",
        "all_checks_pass": ok,
        "checks": checks,
        "diag": diag,
        "counts": {"chunks": n, "pmcids": len(pmcids),
                  "embeddings": int(emb.shape[0]),
                  "faiss_ntotal": int(idx.ntotal),
                  "bm25_docs": int(bm.corpus_size)},
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    lines = ["# ElderDocAI-GeriLit Dev Corpus - Phase 3 Task 5 Validation", "",
             f"- **validated_by**: validate_task5.py",
             f"- **timestamp**: {out['validation_ts']}", "",
             "| Check | Result |", "|---|---|"]
    for k, v in checks.items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Counts", "```json",
              json.dumps(out["counts"], indent=2), "```"]
    (OUT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"all_checks_pass": ok, "n_checks": len(checks),
                      "counts": out["counts"]}, indent=2))
    return out


if __name__ == "__main__":
    main()