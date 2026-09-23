#!/usr/bin/env python3
"""ElderDocAI-GeriLit - Phase 3 Task 10D: quantitative retrieval evaluation.

Evaluates the frozen GeriLit retrieval pipeline against the human-verified
GeriLit-Gold benchmark (134 questions).

Four configurations, using ONLY the existing frozen retriever behavior:
  A. Dense             BGE/FAISS top-5
  B. Sparse            BM25Okapi top-5
  C. Hybrid            existing 0.6*dense + 0.4*sparse (top-5 candidates)
  D. Hybrid + Rerank   C followed by CrossEncoder rerank (final top-3)

Metrics (per query, binary relevance strictly from gold_relevant_chunk_ids):
  Recall@1/3/5 : 1 if >=1 gold chunk in top K else 0
  MRR          : 1/rank of first gold chunk in the ranking (0 if absent)
  nDCG@5       : binary DCG@5 / IDCG@5

No retrieval algorithm is modified; no production code is touched; offline.

Outputs:
  metadata/task10d_retrieval_results.json   (per-query, all configs)
  metadata/task10d_summary.json             (aggregate summaries)
"""
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

GOLD_FILE = REPO_DIR / "data" / "geri_lit_gold.json"
RESULTS_FILE = META_DIR / "task10d_retrieval_results.json"
SUMMARY_FILE = META_DIR / "task10d_summary.json"

DENSE_K = 5
SPARSE_K = 5
HYBRID_K = 5
FINAL_K = 3
K_EVAL = [1, 3, 5]
NDCG_K = 5

CONFIGS = ("dense", "sparse", "hybrid", "rerank")


def metrics_for(ranking, gold_set):
    """ranking: list of chunk_ids (ordered). gold_set: set of gold chunk ids."""
    m = {}
    for k in K_EVAL:
        m["recall@%d" % k] = 1.0 if any(
            c in gold_set for c in ranking[:k]) else 0.0
    # MRR
    rr = 0.0
    for idx, c in enumerate(ranking):
        if c in gold_set:
            rr = 1.0 / (idx + 1)
            break
    m["mrr"] = rr
    # nDCG@5 (binary)
    dcg = 0.0
    for idx, c in enumerate(ranking[:NDCG_K]):
        if c in gold_set:
            dcg += 1.0 / (idx + 2)  # log2(idx+2)
    n_rel = min(NDCG_K, len(gold_set))
    idcg = sum(1.0 / (i + 2) for i in range(n_rel))
    m["ndcg@5"] = dcg / idcg if idcg > 0 else 0.0
    return m


def fmt(v):
    return round(float(v), 6)
def run_retrieval(retriever, query):
    """Run all four frozen configurations; return rankings per config."""
    ranks = {}

    dense = retriever.dense_retrieve(query, DENSE_K)
    ranks["dense"] = [retriever.row_to_chunk(row)[0]
                      for _, row in dense]

    sparse = retriever.sparse_retrieve(query, SPARSE_K)
    ranks["sparse"] = [retriever.row_to_chunk(row)[0]
                       for _, row in sparse]

    hybrid = retriever.hybrid_retrieve(query,
                                       dense_k=DENSE_K, sparse_k=SPARSE_K)
    ranks["hybrid"] = [retriever.row_to_chunk(c["row"])[0]
                       for c in hybrid]

    reranked = retriever.rerank(query, hybrid)
    ranks["rerank"] = [retriever.row_to_chunk(c["row"])[0]
                       for c in reranked]
    return ranks


def main():
    gold = json.loads(GOLD_FILE.read_text(encoding="utf-8"))
    recs = gold["records"]
    assert len(recs) == 134, "expected 134 gold questions"

    retriever = GeriLitRetriever()
    queries = [r["question"].strip() for r in recs]
    assert all(queries), "all questions non-empty"

    results = []
    for rec in recs:
        q = rec["question"].strip()
        gold_set = set(rec["gold_relevant_chunk_ids"])
        ranks = run_retrieval(retriever, q)
        qrow = {
            "final_benchmark_id": rec["final_benchmark_id"],
            "original_candidate_id": rec.get("original_candidate_id"),
            "question": q,
            "primary_topic": rec.get("primary_topic"),
            "secondary_topics": rec.get("secondary_topics", []),
            "gold_relevant_chunk_ids": sorted(gold_set),
            "rankings": {c: ranks[c] for c in CONFIGS},
            "metrics": {c: metrics_for(ranks[c], gold_set)
                        for c in CONFIGS},
        }
        results.append(qrow)

    out = {
        "task": "phase3_task10d",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "metadata/evaluate_task10d.py",
        "config_k": {"dense_k": DENSE_K, "sparse_k": SPARSE_K,
                     "hybrid_k": HYBRID_K, "final_k": FINAL_K},
        "n_questions": len(results),
        "results": results,
    }
    RESULTS_FILE.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---- aggregate summary ----
    summary = {}
    for cfg in CONFIGS:
        agg = {"recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0,
               "mrr": 0.0, "ndcg@5": 0.0}
        for r in results:
            met = r["metrics"][cfg]
            agg["recall@1"] += met["recall@1"]
            agg["recall@3"] += met["recall@3"]
            agg["recall@5"] += met["recall@5"]
            agg["mrr"] += met["mrr"]
            agg["ndcg@5"] += met["ndcg@5"]
        n = len(results)
        summary[cfg] = {k: fmt(v / n) for k, v in agg.items()}
        summary[cfg]["n"] = n

    topics = {}
    by_topic = defaultdict(list)
    for r in results:
        by_topic[r["primary_topic"]].append(r)
    for t, rows in sorted(by_topic.items()):
        topics[t] = {}
        for cfg in CONFIGS:
            agg = {"recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0,
                   "mrr": 0.0, "ndcg@5": 0.0}
            for r in rows:
                met = r["metrics"][cfg]
                for k in agg:
                    agg[k] += met[k]
            n = len(rows)
            topics[t][cfg] = {k: fmt(v / n) for k, v in agg.items()}
            topics[t][cfg]["n"] = n

    summ = {
        "task": "phase3_task10d",
        "generated_at_utc": out["generated_at_utc"],
        "overall": summary,
        "topics": topics,
        "topics_covered": sorted(by_topic.keys()),
        "n_topics": len(by_topic),
    }
    SUMMARY_FILE.write_text(json.dumps(summ, indent=2), encoding="utf-8")

    print(json.dumps({"n_questions": len(results),
                      "overall": summary,
                      "topics_covered": len(by_topic)}, indent=2))


if __name__ == "__main__":
    main()