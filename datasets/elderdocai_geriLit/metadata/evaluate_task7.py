#!/usr/bin/env python3
"""ElderDocAI-GeriLit dev-v0.1 - Phase 3 Task 7: retrieval evaluation.

STANDALONE evaluation. Frozen GeriLit artifacts are READ ONLY.

RELEVANCE-LABEL SITUATION (determined during inspection)
---------------------------------------------------------
No legitimate relevance judgments exist for the GeriLit corpus in this
repository:
  - Gold96 / gold_qa_extended.json : 96 questions; labels are integer chunk
    ids + 6 PDF filenames of the LEGACY six-PDF KB. No PMCID / GeriLit chunk
    identifiers. Not mappable to GeriLit.
  - evaluation/evaluation_queries.json : 5 queries; expected_source = legacy
    PDF filenames (not GeriLit PMCIDs).
  - PubMedQA / TREC-CDS / BioASQ : PENDING_BENCHMARK_ACQUISITION (not
    downloaded; protected_ids empty).
No qrels / relevance / judgment files exist in the repository.

Therefore Recall@k, Precision@k, MRR, nDCG@k, Hit-Rate@k are
NOT COMPUTABLE (insufficient/absent ground truth). This script does NOT
manufacture labels. It performs a strict FUNCTIONAL + DETERMINISTIC
multi-stage evaluation (dense / BM25 / hybrid / CrossEncoder / final top-3)
and records unavailable metrics and their reasons explicitly.

Stages evaluated functionally:
  A. Dense       : FAISS IndexFlatIP top-5
  B. BM25        : BM25Okapi top-5
  C. Hybrid      : 0.6*d + 0.4*s (Task 6 fusion), candidate pool
  D. CrossEncoder: reranked ranking of hybrid candidates
  E. Final top-3 : GeriLitRetriever.retrieve() output
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

META_DIR = Path(__file__).resolve().parent          # .../elderdocai_geriLit/metadata
GERI_DIR = META_DIR.parent                           # .../elderdocai_geriLit
REPO_DIR = GERI_DIR.parent.parent                    # ElderDocAI-System

sys.path.insert(0, str(GERI_DIR))
sys.path.insert(0, str(REPO_DIR))

from geri_lit_retriever import GeriLitRetriever  # noqa: E402

VALIDATION_DIR = META_DIR / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

# Frozen artifact hashes (Task 6 report §19; verified unchanged)
FROZEN = {
    "chunks.jsonl": "62bfdde3c7e77cf56112e79a71bc79bdea70767f6b2625701e392bbb6581c6b3",
    "embeddings.npy": "b0905d2f96903dadc3cf5b4a737f330853656f846955de706f21bd714ce2dacf",
    "faiss_index.bin": "b7016b9dabbab08ee68f875007b6db2b013af4472b1a3c0958af8a566b59f2b3",
    "bm25.pkl": "17f503240f2f9a13abf58c94359884b03b06bec1a5a59f63f921328e2a2c70c9",
    "row_mapping.json": "4d649f81d554c41ec7b63c65dce887466dfa6245745c568315735f1b93a1872b",
}
LEGACY_SCRIPTS = ["scripts/hybrid_retriever.py", "scripts/rag_chat.py",
                  "scripts/reranker.py", "scripts/evidence_aggregation.py",
                  "scripts/authority_mapping.py", "scripts/carebuddy_service.py"]

# Controlled functional query set: 10 Task-6 smoke queries + 5 legacy
# evaluation queries (legacy expected_source labels are NOT used as
# GeriLit relevance labels; they only existed for the six-PDF KB).
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
LEGACY_QUERIES = [
    "What are safe ways for older adults to take medicine?",
    "How can family members help care for an older adult?",
    "What are common signs of memory loss in older adults?",
    "What kinds of exercise are recommended for older adults?",
    "What does a healthy eating pattern include?",
]
ALL_QUERIES = SMOKE_QUERIES + LEGACY_QUERIES
N_QUERIES = len(ALL_QUERIES)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def frozen_hashes_ok(retriever):
    here = {
        "chunks.jsonl": sha256_file(GERI_DIR / "chunks/chunks.jsonl"),
        "embeddings.npy": sha256_file(GERI_DIR / "index/embeddings.npy"),
        "faiss_index.bin": sha256_file(GERI_DIR / "index/faiss_index.bin"),
        "bm25.pkl": sha256_file(GERI_DIR / "index/bm25.pkl"),
        "row_mapping.json": sha256_file(GERI_DIR / "index/row_mapping.json"),
    }
    return here, all(here[k] == FROZEN[k] for k in FROZEN)


def legacy_hashes():
    return {s: sha256_file(REPO_DIR / s) for s in LEGACY_SCRIPTS}
def evaluate_stages(retriever):
    """Return per-query stage results with functional validity flags."""
    results = []
    for q in ALL_QUERIES:
        row = {"query": q}
        # A. dense top-5
        dense = retriever.dense_retrieve(q)
        row["dense_rows"] = [int(r) for _, r in dense]
        row["dense_scores_finite"] = all(
            np.isfinite(s) for s, _ in dense)
        # B. BM25 top-5
        sparse = retriever.sparse_retrieve(q)
        row["sparse_rows"] = [int(r) for _, r in sparse]
        row["sparse_scores_finite"] = all(
            np.isfinite(s) for s, _ in sparse)
        # C. hybrid candidate pool
        hybrid = retriever.hybrid_retrieve(q)
        row["hybrid_rows"] = [c["row"] for c in hybrid]
        row["hybrid_scores_finite"] = all(
            np.isfinite(c["hybrid_score"]) for c in hybrid)
        # D. CrossEncoder rerank (ordered)
        reranked = retriever.rerank(q, hybrid)
        row["rerank_rows_order"] = [c["row"] for c in reranked]
        row["rerank_scores_finite"] = all(
            np.isfinite(c["rerank_score"]) for c in reranked)
        # E. final top-3
        final = retriever.retrieve(q)
        row["final_rows"] = [e["row_id"] for e in final]
        row["final_chunk_ids"] = [e["chunk_id"] for e in final]
        row["final_pmcids"] = [e["pmcid"] for e in final]
        row["final_provenance_complete"] = all(
            e.get("pmcid") and e.get("source_locator") and e.get("text")
            and e["text"].strip() for e in final)
        row["final_no_duplicates"] = (
            len(set(row["final_chunk_ids"])) == len(row["final_chunk_ids"]))
        row["final_rows_valid"] = all(
            0 <= r < len(retriever.chunks) for r in row["final_rows"])
        row["final_all_provenance_keys"] = all(
            set(["chunk_id", "pmcid", "source_locator", "section_id",
                 "region", "content_type", "evidence_type", "text"]).issubset(
                     set(e.keys()) if e.get("section_id") is not None
                     else set(e.keys()) | {"section_id"})
            for e in final)
        results.append(row)
    return results


def validate_functional(results, n_expected):
    """Aggregate functional checks across queries."""
    ok = True
    summary = {}
    for key in ["dense_scores_finite", "sparse_scores_finite",
                "hybrid_scores_finite", "rerank_scores_finite",
                "final_provenance_complete", "final_no_duplicates",
                "final_rows_valid"]:
        vals = [r[key] for r in results]
        summary[key] = all(vals)
        ok = ok and all(vals)
    summary["dense_k"] = all(len(r["dense_rows"]) == 5 for r in results)
    summary["sparse_k"] = all(len(r["sparse_rows"]) == 5 for r in results)
    summary["final_k"] = all(len(r["final_rows"]) == 3 for r in results)
    summary["n_queries"] = len(results) == n_expected
    ok = ok and all(summary.values())
    return ok, summary
def main():
    retriever = GeriLitRetriever()
    n = len(retriever.chunks)

    # pass 1 and pass 2 (identical inputs, two independent executions)
    results1 = evaluate_stages(retriever)
    results2 = evaluate_stages(retriever)

    # determinism: compare everything that determines the ranking
    comp_keys = ["dense_rows", "sparse_rows", "hybrid_rows",
                 "rerank_rows_order", "final_rows", "final_chunk_ids",
                 "final_pmcids"]
    det_diffs = []
    for a, b in zip(results1, results2):
        for k in comp_keys:
            if a[k] != b[k]:
                det_diffs.append((a["query"], k))
    deterministic = len(det_diffs) == 0

    func_ok, func_summary = validate_functional(results1, N_QUERIES)
    here_hashes, frozen_ok = frozen_hashes_ok(retriever)
    legacy_before = legacy_hashes()

    # (legacy retriever executes check: reuse Task-6 regression style)
    legacy_exec = False
    try:
        import scripts.hybrid_retriever as hr  # noqa: F401
        out = hr.hybrid_search("What is medication safety advice?")
        legacy_exec = isinstance(out, list) and len(out) >= 1
    except Exception:
        legacy_exec = False
    legacy_after = legacy_hashes()
    legacy_unchanged = legacy_before == legacy_after

    all_checks = dict(func_summary)
    all_checks["legacy_scripts_hash_unchanged"] = legacy_unchanged
    all_checks["frozen_artifacts_unchanged"] = frozen_ok
    all_checks["deterministic_two_passes"] = deterministic
    all_checks["offline_models_loaded"] = (
        retriever.dim == 768 and len(retriever.chunks) == 17930)

    pass_all = all(v is True for v in all_checks.values())

    metrics = {
        "Recall@1": "NOT COMPUTABLE",
        "Recall@3": "NOT COMPUTABLE",
        "Recall@5": "NOT COMPUTABLE",
        "Recall@10": "NOT COMPUTABLE",
        "Precision@k": "NOT COMPUTABLE",
        "MRR": "NOT COMPUTABLE",
        "nDCG@k": "NOT COMPUTABLE",
        "Hit-Rate@k": "NOT COMPUTABLE",
        "reason": (
            "No legitimate relevance judgments mapped to GeriLit chunks/PMCIDs "
            "exist in the repository (Gold96/evaluation_queries reference the "
            "legacy six-PDF KB; external benchmarks not acquired). Labels were "
            "NOT manufactured."),
    }

    out = {
        "task": "phase3_task7",
        "validation_ts": datetime.now(timezone.utc).isoformat(),
        "evaluated_by": "metadata/evaluate_task7.py",
        "configuration": {
            "dense_k": 5, "sparse_k": 5, "rerank_candidates": 5,
            "final_k": 3, "dense_weight": 0.6, "sparse_weight": 0.4,
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        },
        "corpus_identity": {"chunks": n, "pmcids": 500},
        "query_counts": {"smoke": len(SMOKE_QUERIES),
                         "legacy_query_text": len(LEGACY_QUERIES),
                         "total": N_QUERIES},
        "relevance_labels": {
            "available": False,
            "detail": ("No GeriLit-mapped relevance judgments available; "
                       "Gold96 & evaluation_queries.json reference legacy "
                       "six-PDF KB only; external benchmarks pending."),
        },
        "metrics": metrics,
        "functional_validation": func_summary,
        "determinism": {"two_passes_identical": deterministic,
                        "diffs": det_diffs},
        "stage_evals": results1,
        "frozen_hashes": here_hashes,
        "legacy_scripts_hashes": legacy_after,
        "all_checks_pass": pass_all,
        "checks": all_checks,
    }
    OUT_RESULTS = META_DIR / "phase3_task7_evaluation_results.json"
    OUT_RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # validation artifacts (convention: validation/*.json + *.md)
    vjson = {
        "validation_ts": out["validation_ts"],
        "validated_by": "metadata/evaluate_task7.py",
        "all_checks_pass": pass_all,
        "checks": all_checks,
        "metrics_not_computable": True,
    }
    (VALIDATION_DIR / "phase3_task7_validation.json").write_text(
        json.dumps(vjson, indent=2), encoding="utf-8")
    lines = ["# ElderDocAI-GeriLit - Phase 3 Task 7 Validation", "",
             f"- **timestamp**: {out['validation_ts']}", "",
             "| Check | Result |", "|---|---|"]
    for k, v in all_checks.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    (VALIDATION_DIR / "phase3_task7_validation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "all_checks_pass": pass_all,
        "functional_ok": func_ok,
        "deterministic": deterministic,
        "n_queries": N_QUERIES,
        "metrics": "NOT COMPUTABLE (no legitimate GeriLit relevance labels)",
    }, indent=2))


if __name__ == "__main__":
    main()