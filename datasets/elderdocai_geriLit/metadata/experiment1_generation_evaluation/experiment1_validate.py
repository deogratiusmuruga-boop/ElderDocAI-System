#!/usr/bin/env python3
"""Experiment 1 - validation script.

Validates the per-question records, aggregate recomputation, benchmark
integrity, frozen-artifact integrity, and reproducibility comparison.
Read-only with respect to production code and frozen artifacts.
"""
import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[4]
OUT_DIR = Path(__file__).resolve().parent

BENCHMARK_FILE = BASE_DIR / "data" / "geri_lit_gold_v1_1.json"
EXPECTED_BENCHMARK_SHA256 = (
    "1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72"
)
GERE_DIR = BASE_DIR / "datasets" / "elderdocai_geriLit"

FROZEN_FILES = {
    "benchmark_v1_0": BASE_DIR / "data" / "geri_lit_gold.json",
    "benchmark_v1_1": BENCHMARK_FILE,
    "chunks": GERE_DIR / "chunks" / "chunks.jsonl",
    "embeddings": GERE_DIR / "index" / "embeddings.npy",
    "faiss": GERE_DIR / "index" / "faiss_index.bin",
    "bm25": GERE_DIR / "index" / "bm25.pkl",
    "row_mapping": GERE_DIR / "index" / "row_mapping.json",
    "reliability_config": BASE_DIR / "config" / "reliability_config.json",
}
FROZEN_EVAL_FILES = {
    "task10d_results": GERE_DIR / "metadata" / "task10d_retrieval_results.json",
    "task10d_summary": GERE_DIR / "metadata" / "task10d_summary.json",
    "task10e_analysis": GERE_DIR / "metadata" / "task10e_failure_analysis.json",
    "task3_summary": GERE_DIR
    / "metadata"
    / "task3_reliability_evaluation"
    / "task3_summary.json",
}
TOPIC_CODES = ["C01", "C02", "C03", "C04", "C05",
               "C06", "C07", "C08", "C09", "C10"]

REQUIRED_FIELDS = [
    "question_id", "topic", "question", "gold_pmcid", "gold_chunk_ids",
    "retrieved_evidence_ids", "retrieved_evidence_texts", "retrieval_backend",
    "evidence_count", "initial_reliability", "initial_decision", "reliability",
    "decision", "retrieval_attempts", "refinement_attempts", "refused",
    "generated", "answer", "latency_seconds", "faithfulness",
    "answer_relevance", "evidence_support", "hallucination", "error",
]


def sha256(path):
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(name, ok, detail=""):
    return {"check": name, "passed": bool(ok), "detail": str(detail)}


def _row_value(row, key):
    """Read a numeric row field, falling back to the nested reliability dict."""
    v = row.get(key)
    if v is None:
        rel = row.get("reliability") or {}
        v = rel.get(key)
    return v


def recompute_summary(rows):
    """Recompute aggregates directly from per-question rows (protocol 6)."""
    vals = {}
    for key in ["overall_reliability", "authority", "relevance", "support",
                "coverage", "consistency", "faithfulness", "answer_relevance",
                "evidence_support", "latency_seconds"]:
        nums = [float(_row_value(r, key)) for r in rows
                if _row_value(r, key) is not None]
        vals[key] = ({
            "n": len(nums),
            "mean": round(statistics.mean(nums), 6) if nums else None,
            "median": round(statistics.median(nums), 6) if nums else None,
            "std": round(statistics.stdev(nums), 6) if len(nums) > 1 else 0.0,
            "min": round(min(nums), 6) if nums else None,
            "max": round(max(nums), 6) if nums else None,
        })
    dec = {}
    for r in rows:
        d = r.get("decision") or "MISSING"
        dec[d] = dec.get(d, 0) + 1
    return vals, dec


def validate(run1_path, run2_path, summary_path):
    results = []

    # ---- 1/2/3. Benchmark integrity ----
    bench_sha = sha256(BENCHMARK_FILE)
    results.append(check("benchmark_sha256_matches",
                         bench_sha == EXPECTED_BENCHMARK_SHA256, bench_sha))
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        bench = json.load(f)["records"]
    results.append(check("benchmark_121_records", len(bench) == 121))
    results.append(check("benchmark_all_accepted",
                         all(r.get("review_status") == "ACCEPTED"
                             for r in bench)))
    bids = [r["final_benchmark_id"] for r in bench]
    results.append(check("benchmark_ids_unique", len(set(bids)) == len(bids)))

    # ---- 4. Run1 per-question completeness ----
    if not os.path.exists(run1_path):
        out = {"passed": False, "total_checks": 1, "passed_checks": 0,
               "failed_checks": 1,
               "checks": [check("run1_exists", False, run1_path)]}
        with open(OUT_DIR / "experiment1_validation.json", "w",
                  encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(json.dumps({"summary": "run1 file missing", "passed": False},
                         indent=2))
        return

    with open(run1_path, "r", encoding="utf-8") as f:
        rows1 = json.load(f)
    results.append(check("run1_121_records", len(rows1) == 121,
                         "n=%d" % len(rows1)))
    r1_ids = [r["question_id"] for r in rows1]
    results.append(check("run1_ids_unique", len(set(r1_ids)) == len(r1_ids)))
    results.append(check("run1_all_benchmark_ids_represented_exactly_once",
                         sorted(r1_ids) == sorted(bids)))
    for field in REQUIRED_FIELDS:
        missing = [r["question_id"] for r in rows1 if field not in r]
        results.append(check("field_present_%s" % field, not missing,
                             "missing in %r" % missing))
    # gold chunks traceable to benchmark records
    by_id = {r["final_benchmark_id"]: r for r in bench}
    bad_gold = [r["question_id"] for r in rows1
                if set(r.get("gold_chunk_ids") or [])
                != set(by_id.get(r["question_id"], {}).get(
                    "gold_relevant_chunk_ids") or [])]
    results.append(check("gold_chunk_ids_consistent_with_benchmark",
                         not bad_gold, "mismatch %r" % bad_gold))

    # ---- 5. Aggregate recomputation ----
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        recomputed, dec_recomp = recompute_summary(rows1)
        mismatch = []
        for key in ["overall_reliability", "faithfulness",
                    "answer_relevance", "evidence_support"]:
            if summary.get(key) != recomputed.get(key):
                mismatch.append(key)
        results.append(check(
            "aggregate_statistics_recompute", not mismatch,
            "mismatches: %r" % mismatch))
        results.append(check(
            "decision_counts_recompute",
            summary.get("decision_counts") == dec_recomp))
    else:
        results.append(check("summary_present", False, summary_path))

    # ---- 6. Frozen artifact integrity (post-run snapshot) ----
    for name, path in FROZEN_FILES.items():
        s = sha256(path)
        results.append(check("frozen_%s_present" % name, s is not None, s))
    for name, path in FROZEN_EVAL_FILES.items():
        s = sha256(path)
        results.append(check("frozen_eval_%s_present" % name,
                             s is not None, s))

    # ---- 7. Model/prompt config untouched ----
    rag = (BASE_DIR / "scripts" / "rag_chat.py").read_text(encoding="utf-8")
    results.append(check(
        "production_generation_options_frozen",
        '"temperature": 0,' in rag and '"top_p": 0.1,' in rag
        and '"top_k": 10,' in rag))
    results.append(check(
        "production_llm_frozen", "llama3.2:latest" in rag))
    results.append(check(
        "gate_budgets_frozen",
        "MAX_REFINE_ATTEMPTS = 1" in rag
        and "MAX_RETRIEVE_ATTEMPTS = 1" in rag))

    # ---- 8. Reproducibility ----
    if os.path.exists(run2_path):
        with open(run2_path, "r", encoding="utf-8") as f:
            rows2 = json.load(f)
        results.append(check("run2_121_records", len(rows2) == 121,
                             "n=%d" % len(rows2)))
        r2_ids = [r["question_id"] for r in rows2]
        results.append(check("run2_ids_unique",
                             len(set(r2_ids)) == len(r2_ids)))
        fields = ["retrieved_evidence_ids", "reliability", "decision",
                  "refinement_attempts", "retrieval_attempts", "refused",
                  "generated", "answer", "faithfulness", "answer_relevance",
                  "evidence_support", "hallucination"]
        same = {}
        m1 = {r["question_id"]: r for r in rows1}
        m2 = {r["question_id"]: r for r in rows2}
        for field in fields:
            cnt = sum(1 for q in m1
                      if m1[q].get(field) == m2[q].get(field))
            same[field] = cnt
        for field in fields:
            results.append(check(
                "repro_%s_identical" % field, same[field] == 121,
                "%d/121 match" % same[field]))
    else:
        results.append(check("run2_present", False, run2_path))

    passed = sum(1 for c in results if c["passed"])
    total = len(results)
    ok = passed == total
    out = {
        "passed": bool(ok),
        "total_checks": total,
        "passed_checks": passed,
        "failed_checks": total - passed,
        "benchmark": "data/geri_lit_gold_v1_1.json",
        "benchmark_sha256": bench_sha,
        "checks": results,
    }
    with open(OUT_DIR / "experiment1_validation.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    md = ["# Experiment 1 - Validation Report",
          "",
          "**Passed:** %s/%s" % (passed, total),
          "",
          "| # | Check | Passed | Detail |",
          "|---|-------|--------|--------|"]
    for i, c in enumerate(results, 1):
        md.append("| %d | %s | %s | %s |"
                  % (i, c["check"], c["passed"], c["detail"]))
    (OUT_DIR / "experiment1_validation.md").write_text(
        "\n".join(md), encoding="utf-8")
    print(json.dumps({
        "summary": "rows=%d total_checks=%d passed=%d" % (
            len(rows1), total, passed),
        "total_checks": total, "passed_checks": passed,
        "failed": [c["check"] for c in results if not c["passed"]],
        "passed": ok}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run1", default=str(
        OUT_DIR / "experiment1_per_question_run1.json"))
    ap.add_argument("--run2", default=str(
        OUT_DIR / "experiment1_per_question_run2.json"))
    ap.add_argument("--summary", default=str(
        OUT_DIR / "experiment1_summary_run1.json"))
    args = ap.parse_args()
    validate(args.run1, args.run2, args.summary)