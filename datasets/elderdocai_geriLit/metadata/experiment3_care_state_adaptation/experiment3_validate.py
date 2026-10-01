#!/usr/bin/env python3
"""Experiment 3 - validation script.

Validates benchmark integrity, the real (non-synthetic) mechanism, paired
record structure, aggregate recomputation, reproducibility, and frozen
artifact integrity. Read-only with respect to production code and artifacts.
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
    "exp1_per_question": GERE_DIR
    / "metadata"
    / "experiment1_generation_evaluation"
    / "experiment1_per_question_run1.json",
    "exp2_per_question": GERE_DIR
    / "metadata"
    / "experiment2_gating_effectiveness"
    / "experiment2_per_question_run1.json",
}

REQUIRED_TOP = ["pair_id", "question_id", "topic", "question", "gold_pmcid",
                "gold_chunk_ids", "state_A", "state_B", "state_difference",
                "score_A", "score_B", "transition_type",
                "transition_direction", "condition_A", "condition_B",
                "adaptation_judgment", "appropriateness_judgment",
                "paired_differences"]
COND_FIELDS = ["evidence_ids", "reliability", "decision",
               "refinement_attempts", "retrieval_attempts", "refused",
               "generated", "answer", "faithfulness", "answer_relevance",
               "evidence_support", "gold_chunk_retrieved", "error"]


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


def recompute_paired_deltas(rows, key):
    vals = [r["paired_differences"][key] for r in rows
            if r["paired_differences"].get(key) is not None]
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 6) if vals else None,
    }
def validate(run1_path, run2_path):
    results = []

    bench_sha = sha256(BENCHMARK_FILE)
    results.append(check("benchmark_sha256_matches",
                         bench_sha == EXPECTED_BENCHMARK_SHA256, bench_sha))
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        bench = json.load(f)["records"]
    results.append(check("benchmark_121_records", len(bench) == 121))
    bids = [r["final_benchmark_id"] for r in bench]
    results.append(check("benchmark_ids_unique", len(set(bids)) == len(bids)))

    # ---- current-state audit + mechanism are non-synthetic ----
    results.append(check("current_state_audit_exists",
                         (OUT_DIR / "current_state_audit.md").exists()))
    results.append(check("protocol_exists",
                         (OUT_DIR / "experiment3_protocol.md").exists()))
    care_src = (BASE_DIR / "scripts" / "care_state.py").read_text(
        encoding="utf-8")
    results.append(check("care_state_module_exists",
                         "def compute_care_state" in care_src))
    # The estimator must be a pure computation module: it must not import or
    # read Synthea data (no file I/O, no synthea path/import). The word
    # "Synthea" appears only in its disclaimer docstring.
    results.append(check(
        "no_synthea_import_in_estimator",
        "import synthea" not in care_src
        and "from synthea" not in care_src
        and "datasets/synthea" not in care_src
        and "open(" not in care_src
        and "json.load" not in care_src))
    est_src = (OUT_DIR / "experiment3_evaluation.py").read_text(
        encoding="utf-8")
    results.append(check(
        "experiment_no_synthea_data",
        "import synthea" not in est_src
        and "from synthea" not in est_src
        and "datasets/synthea" not in est_src))

    # ---- frozen config ----
    rag = (BASE_DIR / "scripts" / "rag_chat.py").read_text(encoding="utf-8")
    results.append(check("generation_options_frozen",
                         '"temperature": 0,' in rag
                         and '"top_p": 0.1,' in rag
                         and '"top_k": 10,' in rag))
    cfg = json.loads((BASE_DIR / "config" / "reliability_config.json")
                     .read_text(encoding="utf-8"))
    results.append(check(
        "weights_frozen",
        cfg["reliability_weights"] == {"authority": 0.3, "relevance": 0.3,
                                       "support": 0.2, "coverage": 0.1,
                                       "consistency": 0.1}))
    results.append(check(
        "thresholds_frozen",
        cfg["decision_thresholds"] == {"accept": 0.8, "refine": 0.65,
                                       "re_retrieve": 0.45}))

    if not os.path.exists(run1_path):
        out = {"passed": False, "total_checks": 1, "passed_checks": 0,
               "failed_checks": 1,
               "checks": [check("run1_exists", False, run1_path)]}
        with open(OUT_DIR / "experiment3_validation.json", "w",
                  encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(json.dumps({"summary": "run1 missing", "passed": False},
                         indent=2))
        return

    with open(run1_path, "r", encoding="utf-8") as f:
        rows1 = json.load(f)
    results.append(check("run1_20_pairs", len(rows1) == 20,
                         "n=%d" % len(rows1)))
    ids1 = [r["pair_id"] for r in rows1]
    results.append(check("pair_ids_unique", len(set(ids1)) == len(ids1)))
    results.append(check("pairs_from_benchmark",
                         set(ids1) <= set(bids)))
    for field in REQUIRED_TOP:
        missing = [r["pair_id"] for r in rows1
                   if field not in r or (
                       r[field] is None
                       and field not in ("appropriateness_judgment",))]
        results.append(check("field_present_%s" % field, not missing))
    for cond in ("condition_A", "condition_B"):
        for field in COND_FIELDS:
            missing = [r["pair_id"] for r in rows1
                       if field not in r[cond]]
            results.append(check("%s_field_%s" % (cond, field), not missing))
# ---- controlled contrast ----
    results.append(check(
        "states_differ_all_pairs",
        all(r["state_A"] != r["state_B"] for r in rows1)))
    results.append(check(
        "evidence_identical_across_conditions",
        all(r["condition_A"]["evidence_ids"]
            == r["condition_B"]["evidence_ids"] for r in rows1)))
    results.append(check(
        "transition_is_escalation",
        all(r["transition_direction"] == "UP" for r in rows1)))

    # ---- aggregate recomputation ----
    summary_path = OUT_DIR / "experiment3_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        for key in ["faithfulness_delta", "answer_relevance_delta",
                    "evidence_support_delta"]:
            rec = recompute_paired_deltas(rows1, key)
            reported = summary["paired_deltas"][key]
            ok = (rec["mean"] is None or (
                reported.get("mean") is not None
                and abs(reported["mean"] - rec["mean"]) < 1e-6))
            results.append(check("delta_recompute_%s" % key, ok,
                                 "reported=%s rec=%s" %
                                 (reported.get("mean"), rec["mean"])))
    else:
        results.append(check("summary_present", False, str(summary_path)))

    # ---- frozen integrity ----
    for name, path in FROZEN_FILES.items():
        s = sha256(path)
        results.append(check("frozen_%s_present" % name, s is not None, s))
    for name, path in FROZEN_EVAL_FILES.items():
        s = sha256(path)
        results.append(check("frozen_eval_%s_present" % name,
                             s is not None, s))
# ---- reproducibility ----
    if os.path.exists(run2_path):
        with open(run2_path, "r", encoding="utf-8") as f:
            rows2 = json.load(f)
        results.append(check("run2_20_pairs", len(rows2) == 20))
        m1 = {r["pair_id"]: r for r in rows1}
        m2 = {r["pair_id"]: r for r in rows2}
        fields = [
            ("state_inputs", lambda r: (r["state_A"], r["state_B"],
                                        r["score_A"], r["score_B"])),
            ("questions", lambda r: r["question"]),
            ("evidence_A", lambda r: r["condition_A"]["evidence_ids"]),
            ("evidence_B", lambda r: r["condition_B"]["evidence_ids"]),
            ("reliability_A", lambda r: r["condition_A"]["reliability"]),
            ("reliability_B", lambda r: r["condition_B"]["reliability"]),
            ("decision_A", lambda r: r["condition_A"]["decision"]),
            ("decision_B", lambda r: r["condition_B"]["decision"]),
            ("answer_A", lambda r: r["condition_A"]["answer"]),
            ("answer_B", lambda r: r["condition_B"]["answer"]),
            ("adaptation", lambda r: r["adaptation_judgment"]),
        ]
        for name, fn in fields:
            cnt = sum(1 for q in m1 if fn(m1[q]) == fn(m2[q]))
            results.append(check("repro_%s_identical" % name,
                                 cnt == 20, "%d/20" % cnt))
    else:
        results.append(check("run2_present", False, run2_path))

    passed = sum(1 for c in results if c["passed"])
    total = len(results)
    ok = passed == total
    out = {
        "passed": bool(ok), "total_checks": total,
        "passed_checks": passed, "failed_checks": total - passed,
        "benchmark_sha256": bench_sha, "checks": results,
    }
    with open(OUT_DIR / "experiment3_validation.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    md = ["# Experiment 3 - Validation Report", "",
          "**Passed:** %s/%s" % (passed, total), "",
          "| # | Check | Passed | Detail |",
          "|---|-------|--------|--------|"]
    for i, c in enumerate(results, 1):
        md.append("| %d | %s | %s | %s |" % (i, c["check"], c["passed"],
                                             c["detail"]))
    (OUT_DIR / "experiment3_validation.md").write_text(
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
        OUT_DIR / "experiment3_per_question_run1.json"))
    ap.add_argument("--run2", default=str(
        OUT_DIR / "experiment3_per_question_run2.json"))
    args = ap.parse_args()
    validate(args.run1, args.run2)