#!/usr/bin/env python3
"""ElderDocAI-GeriLit - Phase 3 Task 10H: v1.1 quantitative retrieval evaluation.

Controlled replication/extension of Task 10D against the FROZEN GeriLit-Gold
v1.1 benchmark (121 records). The retrieval pipeline, indexes, models, K
settings and metric definitions are EXACTLY those of Task 10D; the only new
input is data/geri_lit_gold_v1_1.json.

Configurations (unchanged from Task 10D):
  A. dense   BGE/FAISS top-5
  B. sparse  BM25Okapi top-5
  C. hybrid  existing 0.6*dense + 0.4*sparse (top-5)
  D. rerank  hybrid followed by CrossEncoder rerank (final top-3)

Metrics (per query; binary gold from gold_relevant_chunk_ids):
  Recall@1/3/5, MRR, nDCG@5 (identical to Task 10D formulas).

Mode --v10-equivalence re-runs the SAME code on v1.0 and compares every
per-query ranking/metric against the frozen Task 10D results (methodological
equivalence check). NO retrieval code is modified; offline only.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

META_DIR = Path(__file__).resolve().parent
GERI_DIR = META_DIR.parent
REPO_DIR = GERI_DIR.parent.parent
if str(GERI_DIR) not in sys.path:
    sys.path.insert(0, str(GERI_DIR))

from geri_lit_retriever import GeriLitRetriever  # noqa: E402
from geri_lit_retriever import (DENSE_K, SPARSE_K, FINAL_K,
                                DENSE_W, SPARSE_W,
                                EMBEDDING_MODEL, RERANK_MODEL)  # noqa: E402

HYBRID_K = DENSE_K  # Task 10D: hybrid candidate cap == top-5 (same as 10D)

GOLD_11_FILE = REPO_DIR / "data" / "geri_lit_gold_v1_1.json"
GOLD_10_FILE = REPO_DIR / "data" / "geri_lit_gold.json"
RESULTS_FILE = META_DIR / "task10h_retrieval_results.json"
SUMMARY_FILE = META_DIR / "task10h_summary.json"
RESULTS_EQUIV = META_DIR / "task10h_v10_equivalence.json"

V11_EXPECTED_SHA = ("1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72")
V10_EXPECTED_SHA = ("28ef54faeec3b9a1cb8c1c79c9a478d8ea3618574b2787a82a74c1460f209c62")

K_EVAL = [1, 3, 5]
NDCG_K = 5
CONFIGS = ("dense", "sparse", "hybrid", "rerank")
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def metrics_for(ranking, gold_set):
    """Ranking: list of chunk_ids (ordered). gold_set: set of gold chunk ids.
    EXACTLY the Task 10D formulas."""
    m = {}
    for k in K_EVAL:
        m["recall@%d" % k] = 1.0 if any(
            c in gold_set for c in ranking[:k]) else 0.0
    rr = 0.0
    for idx, c in enumerate(ranking):
        if c in gold_set:
            rr = 1.0 / (idx + 1)
            break
    m["mrr"] = rr
    dcg = 0.0
    for idx, c in enumerate(ranking[:NDCG_K]):
        if c in gold_set:
            dcg += 1.0 / (idx + 2)
    n_rel = min(NDCG_K, len(gold_set))
    idcg = sum(1.0 / (i + 2) for i in range(n_rel))
    m["ndcg@5"] = dcg / idcg if idcg > 0 else 0.0
    return m


def first_gold_rank(ranking, gold_set):
    for idx, c in enumerate(ranking):
        if c in gold_set:
            return idx + 1
    return None


def fmt(v):
    return round(float(v), 6)


def run_retrieval(retriever, query):
    """Run all four frozen configurations; return rankings per config.
    IDENTICAL call signature/order to Task 10D."""
    ranks = {}
    dense = retriever.dense_retrieve(query, DENSE_K)
    ranks["dense"] = [retriever.row_to_chunk(row)[0] for _, row in dense]
    sparse = retriever.sparse_retrieve(query, SPARSE_K)
    ranks["sparse"] = [retriever.row_to_chunk(row)[0] for _, row in sparse]
    hybrid = retriever.hybrid_retrieve(query, dense_k=DENSE_K, sparse_k=SPARSE_K)
    ranks["hybrid"] = [retriever.row_to_chunk(c["row"])[0] for c in hybrid]
    reranked = retriever.rerank(query, hybrid)
    ranks["rerank"] = [retriever.row_to_chunk(c["row"])[0] for c in reranked]
    return ranks


def build_row(rec, retriever, id_field):
    q = rec["question"].strip()
    gold_set = set(rec["gold_relevant_chunk_ids"])
    ranks = run_retrieval(retriever, q)
    row = {
        "final_benchmark_id": rec[id_field],
        "original_candidate_id": rec.get("original_candidate_id"),
        "candidate_id": rec.get("candidate_id"),
        "original_v1_0_id": rec.get("original_v1_0_id"),
        "pmcid": rec.get("pmcid"),
        "primary_topic": rec.get("primary_topic")
        if "primary_topic" in rec else rec.get("topic"),
        "topic": rec.get("topic"),
        "question": q,
        "gold_relevant_chunk_ids": sorted(gold_set),
        "rankings": {c: ranks[c] for c in CONFIGS},
        "metrics": {c: metrics_for(ranks[c], gold_set) for c in CONFIGS},
        "hit_rank_first_gold": {
            c: first_gold_rank(ranks[c], gold_set) for c in CONFIGS},
    }
    for c in CONFIGS:
        row["top1_" + c] = ranks[c][:1]
        row["top3_" + c] = ranks[c][:3]
        row["top5_" + c] = ranks[c][:5]
    return row
def aggregate(results):
    agg = {}
    for cfg in CONFIGS:
        d = {"recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0,
             "mrr": 0.0, "ndcg@5": 0.0}
        for r in results:
            m = r["metrics"][cfg]
            for k in d:
                d[k] += m[k]
        n = len(results)
        agg[cfg] = {k: fmt(v / n) for k, v in d.items()}
        agg[cfg]["n"] = n
    return agg


def topic_aggregate(results):
    by_topic = defaultdict(list)
    for r in results:
        t = r["primary_topic"] or r["topic"]
        by_topic[t].append(r)
    topics = {}
    for t, rows in sorted(by_topic.items()):
        topics[t] = aggregate(rows)
    return topics


def failure_distribution(results):
    dist = {}
    for cfg in CONFIGS:
        top1 = sum(1 for r in results if r["hit_rank_first_gold"][cfg] == 1)
        top3 = sum(1 for r in results
                   if r["hit_rank_first_gold"][cfg] is not None
                   and r["hit_rank_first_gold"][cfg] <= 3)
        top5 = sum(1 for r in results
                   if r["hit_rank_first_gold"][cfg] is not None
                   and r["hit_rank_first_gold"][cfg] <= 5)
        dist[cfg] = {"top1_hits": top1, "top3_hits": top3, "top5_hits": top5,
                     "no_top5": len(results) - top5}
    dist["any_config_top5"] = sum(
        1 for r in results
        if any(r["hit_rank_first_gold"][c] is not None
               and r["hit_rank_first_gold"][c] <= 5 for c in CONFIGS))
    dist["never_top5"] = len(results) - dist["any_config_top5"]
    return dist


def v10_v11_comparison(overall):
    """v1.1 measured vs v1.0 control from the frozen Task 10D summary."""
    t10d = json.loads(
        (META_DIR / "task10d_summary.json").read_text(encoding="utf-8"))
    v10 = t10d["overall"]
    comp = {}
    for cfg in CONFIGS:
        mkeys = ("recall@1", "recall@3", "recall@5", "mrr", "ndcg@5")
        comp[cfg] = {
            "v1.0": {k: v10[cfg][k] for k in mkeys},
            "v1.1": {k: overall[cfg][k] for k in mkeys},
            "diff_v11_minus_v10": {
                k: fmt(overall[cfg][k] - v10[cfg][k]) for k in mkeys},
        }
    return {"control_source": "task10d_summary.json (frozen)",
            "per_config": comp}


def deterministic_signature(results):
    canon = []
    for r in results:
        canon.append({
            "id": r["final_benchmark_id"],
            "metrics": {c: r["metrics"][c] for c in CONFIGS},
            "rankings": {c: r["rankings"][c] for c in CONFIGS},
        })
    return hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()


def check_v10_equivalence(retriever):
    """Re-run the SAME code on v1.0 and compare with frozen Task 10D."""
    g10 = json.loads(GOLD_10_FILE.read_text(encoding="utf-8"))
    recs10 = g10["records"]
    frozen = json.loads(
        (META_DIR / "task10d_retrieval_results.json")
        .read_text(encoding="utf-8"))["results"]
    assert len(recs10) == 134 and len(frozen) == 134
    mism = []
    for rec, fr in zip(recs10, frozen):
        row = build_row(rec, retriever, "final_benchmark_id")
        for c in CONFIGS:
            if row["rankings"][c] != fr["rankings"][c]:
                mism.append((row["final_benchmark_id"], c, "ranking"))
            if row["metrics"][c] != fr["metrics"][c]:
                mism.append((row["final_benchmark_id"], c, "metrics"))
    eq = {
        "methodological_equivalence_ok": len(mism) == 0,
        "n_mismatches": len(mism),
        "mismatches": mism[:20],
        "v10_records_rerun": len(recs10),
    }
    RESULTS_EQUIV.write_text(json.dumps(eq, indent=2), encoding="utf-8")
    return eq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v10-equivalence", action="store_true",
                    help="re-run v1.0 with the same code and compare with the "
                         "frozen Task 10D results")
    ap.add_argument("--run-name", default="run1")
    args = ap.parse_args()
    ts = datetime.now(timezone.utc).isoformat()

    # --- protect inputs -------------------------------------------------
    assert sha256(GOLD_11_FILE) == V11_EXPECTED_SHA, "v1.1 SHA mismatch: STOP"
    g11 = json.loads(GOLD_11_FILE.read_text(encoding="utf-8"))
    assert g11.get("status") == "FROZEN" and g11.get("version") == "1.1"
    recs = g11["records"]
    assert len(recs) == 121
    assert all(r["review_status"] == "ACCEPTED" for r in recs)
    assert sha256(GOLD_10_FILE) == V10_EXPECTED_SHA, "v1.0 SHA mismatch: STOP"

    h_art = {
        "v1.0": sha256(GOLD_10_FILE),
        "v1.1": sha256(GOLD_11_FILE),
        "chunks": sha256(GERI_DIR / "chunks/chunks.jsonl"),
        "embeddings": sha256(GERI_DIR / "index/embeddings.npy"),
        "faiss": sha256(GERI_DIR / "index/faiss_index.bin"),
        "bm25": sha256(GERI_DIR / "index/bm25.pkl"),
        "row_mapping": sha256(GERI_DIR / "index/row_mapping.json"),
    }

    retriever = GeriLitRetriever()
    eq = None
    if args.v10_equivalence:
        eq = check_v10_equivalence(retriever)
        print("methodological_equivalence_ok:", eq["methodological_equivalence_ok"])

    results = [build_row(r, retriever, "final_benchmark_id") for r in recs]
    overall = aggregate(results)
    topics = topic_aggregate(results)
    failure = failure_distribution(results)
    sig = deterministic_signature(results)

    out = {
        "task": "phase3_task10h",
        "run_name": args.run_name,
        "generated_at_utc": ts,
        "generated_by": "metadata/task10h_retrieval_evaluation.py",
        "benchmark": {"version": "1.1", "status": "FROZEN",
                      "n": len(results),
                      "unique_pmcid": len({r["pmcid"] for r in results}),
                      "unique_gold_chunks": len(
                          {c for r in results
                           for c in r["gold_relevant_chunk_ids"]}),
                      "expected_sha256": V11_EXPECTED_SHA},
        "config_k": {"dense_k": DENSE_K, "sparse_k": SPARSE_K,
                     "hybrid_k": HYBRID_K, "final_k": FINAL_K},
        "weights": {"dense_weight": DENSE_W, "sparse_weight": SPARSE_W},
        "models": {"embedding_model": EMBEDDING_MODEL,
                   "rerank_model": RERANK_MODEL},
        "protected_hashes_before": h_art,
        "results": results,
        "signature": sig,
    }
    RESULTS_FILE.write_text(json.dumps(out, indent=2), encoding="utf-8")

    summary = {
        "task": "phase3_task10h",
        "run_name": args.run_name,
        "generated_at_utc": ts,
        "generated_by": "metadata/task10h_retrieval_evaluation.py",
        "benchmark": out["benchmark"],
        "overall": overall,
        "topics": topics,
        "failure_distribution": failure,
        "v10_v11_comparison": v10_v11_comparison(overall),
        "results_signature": sig,
        "v10_equivalence": eq,
        "v10_control_source": "task10d_summary.json (frozen Task 10D results)",
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({
        "n_questions": len(results),
        "overall": overall,
        "topics_covered": len(topics),
        "signature": sig,
        **failure,
    }, indent=2))


if __name__ == "__main__":
    main()