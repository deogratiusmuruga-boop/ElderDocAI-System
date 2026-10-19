#!/usr/bin/env python3
"""Experiment 5 - frozen retrieval-layer metrics (deterministic, no judge).

The retrieval layer is byte-identical across all five generator conditions, so
retrieval quality is computed once and reported separately (manuscript Table 8).

Reads the frozen Task 10H v1.1 retrieval results (the ordered rerank top-3 per
question plus each question's gold chunk IDs) and computes, for k in {1,2,3}:
  Hit@k  (= Recall@k here, because every question has exactly one gold chunk)
  Precision@k
  nDCG@k
and the rank-cut metrics MRR and MAP (MAP == MRR for single-gold-chunk queries).

Writes only experiment5_retrieval_metrics.json next to this script. No judge,
no model, no network.
"""
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "task10h_retrieval_results.json"
OUT = HERE / "experiment5_retrieval_metrics.json"

KS = [1, 2, 3]


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    results = data["results"]
    n = len(results)

    hit = {k: 0 for k in KS}
    prec = {k: [] for k in KS}
    ndcg = {k: [] for k in KS}
    rr = []

    for row in results:
        gold = set(row["gold_relevant_chunk_ids"])
        ranking = row["rankings"]["rerank"]  # ordered top-3 after rerank
        rel = []
        first_rank = None
        for i, chunk_id in enumerate(ranking, start=1):
            r = 1.0 if chunk_id in gold else 0.0
            rel.append(r)
            if r == 1.0 and first_rank is None:
                first_rank = i

        # perfect DCG for a single relevant item at rank 1
        idcg = (2 ** 1 - 1) / math.log2(1 + 1)
        for k in KS:
            if any(rel[:k]):
                hit[k] += 1
            prec[k].append(sum(rel[:k]) / k)
            dcg = sum((2 ** r - 1) / math.log2(i + 1)
                      for i, r in enumerate(rel[:k], start=1))
            ndcg[k].append(dcg / idcg if idcg else 0.0)

        rr.append(1.0 / first_rank if first_rank else 0.0)

    table = {
        "n": n,
        "source": SRC.name,
        "metric": {
            "Hit@k / Recall@k": {("k=%d" % k): round(hit[k] / n, 4) for k in KS},
            "Precision@k": {("k=%d" % k): round(statistics.mean(prec[k]), 4)
                            for k in KS},
            "nDCG@k": {("k=%d" % k): round(statistics.mean(ndcg[k]), 4)
                       for k in KS},
            "MRR": round(statistics.mean(rr), 4),
            "MAP": round(statistics.mean(rr), 4),
        },
    }

    OUT.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT.name)
    for m, v in table["metric"].items():
        print("  %-18s %s" % (m, v))


if __name__ == "__main__":
    main()
