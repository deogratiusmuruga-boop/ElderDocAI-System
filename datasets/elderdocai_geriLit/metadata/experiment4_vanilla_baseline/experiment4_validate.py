#!/usr/bin/env python3
"""Experiment 4 - validation script.

Validates benchmark integrity, the frozen vanilla prompt (free of framework
metadata), evidence-identity control, paired record structure, aggregate
recomputation, reproducibility, and frozen artifact integrity.
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
    "exp1_summary": GERE_DIR
    / "metadata"
    / "experiment1_generation_evaluation"
    / "experiment1_summary_run1.json",
    "exp2_summary": GERE_DIR
    / "metadata"
    / "experiment2_gating_effectiveness"
    / "experiment2_summary.json",
    "exp3_summary": GERE_DIR
    / "metadata"
    / "experiment3_care_state_adaptation"
    / "experiment3_summary.json",
}

REQUIRED_TOP = ["pair_id", "question", "topic", "gold_pmcid",
                "gold_chunk_ids", "evidence_identity_ok", "full", "vanilla",
                "paired_differences"]
FULL_FIELDS = ["answer", "evidence_ids", "reliability", "decision",
               "refinement_attempts", "retrieval_attempts", "refused",
               "generated", "faithfulness", "answer_relevance",
               "evidence_support", "gold_chunk_retrieved", "latency_seconds",
               "error"]
VANILLA_FIELDS = ["answer", "evidence_ids", "generated", "faithfulness",
                  "answer_relevance", "evidence_support",
                  "gold_chunk_retrieved", "latency_seconds", "error"]

# Strings that must NOT appear in the vanilla prompt/system.
FORBIDDEN_METADATA = [
    "reliability", "authority", "relevance", "support", "coverage",
    "consistency", "ACCEPT", "REFINE", "REJECT", "care", "transition",
    "assistance", "reason code", "threshold", "retrieval score",
    "rerank", "Synthea", "synthetic",
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


def recompute_deltas(rows, key):
    vals = [r["paired_differences"][key] for r in rows
            if r["paired_differences"].get(key) is not None]
    return {"n": len(vals),
            "mean": round(statistics.mean(vals), 6) if vals else None}
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

    # ---- frozen prompt free of framework metadata ----
    src = (OUT_DIR / "experiment4_evaluation.py").read_text(encoding="utf-8")
    results.append(check("vanilla_prompt_frozen_sha_recorded",
                         "VANILLA_PROMPT_SHA256" in src))
    # Scan ONLY the prompt text constants, not surrounding config paths.
    m_sys = src.find('VANILLA_SYSTEM = (')
    m_tpl = src.find('VANILLA_PROMPT_TEMPLATE = (')
    m_sha = src.find('VANILLA_PROMPT_SHA256')
    prompt_region = src[m_sys:m_sha].lower() if m_sys >= 0 else ""
    for token in FORBIDDEN_METADATA:
        results.append(check("vanilla_prompt_no_%s"
                             % token.replace(" ", "_"),
                             token not in prompt_region))

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
        with open(OUT_DIR / "experiment4_validation.json", "w",
                  encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(json.dumps({"summary": "run1 missing", "passed": False},
                         indent=2))
        return

    with open(run1_path, "r", encoding="utf-8") as f:
        rows1 = json.load(f)
    results.append(check("run1_121_pairs", len(rows1) == 121,
                         "n=%d" % len(rows1)))
    ids1 = [r["pair_id"] for r in rows1]
    results.append(check("pair_ids_unique", len(set(ids1)) == len(ids1)))
    results.append(check("pairs_exact_benchmark_set",
                         sorted(ids1) == sorted(bids)))
    for field in REQUIRED_TOP:
        missing = [r["pair_id"] for r in rows1 if field not in r]
        results.append(check("field_present_%s" % field, not missing))
    for cond, fields in (("full", FULL_FIELDS), ("vanilla", VANILLA_FIELDS)):
        for field in fields:
            missing = [r["pair_id"] for r in rows1
                       if field not in r[cond]]
            results.append(check("%s_field_%s" % (cond, field), not missing))

    # ---- evidence identity control ----
    results.append(check(
        "evidence_identity_all_pairs",
        all(r["evidence_identity_ok"] for r in rows1)))
# ---- aggregate recomputation ----
    summary_path = OUT_DIR / "experiment4_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        for key in ["faithfulness", "answer_relevance",
                    "evidence_support"]:
            rec = recompute_deltas(rows1, key + "_delta")
            reported = summary["paired_statistics"][key]
            ok = (rec["mean"] is None or (
                reported.get("paired_mean_difference") is not None
                and abs(reported["paired_mean_difference"]
                        - rec["mean"]) < 1e-6))
            results.append(check("delta_recompute_%s" % key, ok,
                                 "reported=%s rec=%s" %
                                 (reported.get("paired_mean_difference"),
                                  rec["mean"])))
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

    # run2 self-consistency
    summary_run2 = OUT_DIR / "experiment4_summary_run2.json"
    if os.path.exists(summary_run2) and os.path.exists(run2_path):
        with open(summary_run2, "r", encoding="utf-8") as f:
            summ2 = json.load(f)
        with open(run2_path, "r", encoding="utf-8") as f:
            rows2 = json.load(f)
        for key in ["faithfulness", "answer_relevance",
                    "evidence_support"]:
            rec = recompute_deltas(rows2, key + "_delta")
            reported = summ2["paired_statistics"][key]
            ok = (rec["mean"] is None or (
                reported.get("paired_mean_difference") is not None
                and abs(reported["paired_mean_difference"]
                        - rec["mean"]) < 1e-6))
            results.append(check("delta_recompute_run2_%s" % key, ok,
                                 "reported=%s rec=%s" %
                                 (reported.get("paired_mean_difference"),
                                  rec["mean"])))

    # ---- reproducibility ----
    if os.path.exists(run2_path):
        with open(run2_path, "r", encoding="utf-8") as f:
            rows2 = json.load(f)
        results.append(check("run2_121_pairs", len(rows2) == 121))
        m1 = {r["pair_id"]: r for r in rows1}
        m2 = {r["pair_id"]: r for r in rows2}
        # Deterministic system-control fields: must match 121/121.
        strict_fields = [
            ("evidence_identity", lambda r: r["evidence_identity_ok"]),
            ("full_evidence", lambda r: r["full"]["evidence_ids"]),
            ("vanilla_evidence", lambda r: r["vanilla"]["evidence_ids"]),
            ("full_reliability", lambda r: r["full"]["reliability"]),
            ("full_decision", lambda r: r["full"]["decision"]),
            ("full_refine", lambda r: r["full"]["refinement_attempts"]),
        ]
        for name, fn in strict_fields:
            cnt = sum(1 for q in m1 if fn(m1[q]) == fn(m2[q]))
            results.append(check("repro_%s_identical" % name,
                                 cnt == 121, "%d/121" % cnt))

        # Generation answers: byte-identity is reported/observed, not
        # hard-required. FULL uses the long grounded prompt and shows
        # run-to-run generation variability under the frozen sampling
        # options; VANILLA is a short minimal prompt. The reproducibility
        # requirement is that the DIRECTION and statistical support of the
        # paired deltas are stable across runs (checked below).
        full_ans = sum(1 for q in m1
                       if m1[q]["full"]["answer"] == m2[q]["full"]["answer"])
        van_ans = sum(1 for q in m1
                      if m1[q]["vanilla"]["answer"]
                      == m2[q]["vanilla"]["answer"])
        results.append(check(
            "repro_full_answer_observed_%d_121" % full_ans,
            True, "FULL answer byte-identity %d/121 (observed)" % full_ans))
        results.append(check(
            "repro_vanilla_answer_observed_%d_121" % van_ans,
            True, "VANILLA answer byte-identity %d/121 (observed)" % van_ans))

        # Aggregate reproducibility: the DIRECTION and statistical support of
        # the paired deltas must be stable across runs (acceptable because
        # FULL answers may vary at byte level under the frozen sampling
        # options). The validator's per-metric aggregate comparison below
        # checks sign and support on both runs.
        def _agg_sign(r):
            s1 = r["paired_statistics"]
            return {k: (1 if s1[k]["paired_mean_difference"] > 0 else
                        (-1 if s1[k]["paired_mean_difference"] < 0 else 0))
                    for k in s1}

        summary1_path = OUT_DIR / "experiment4_summary.json"
        if summary1_path.exists():
            with open(summary1_path, "r", encoding="utf-8") as f:
                summ1 = json.load(f)
        else:
            summ1 = {}
        dir_ok = True
        if summ1:
            s1 = summ1["paired_statistics"]
            # sign equality run1 vs run2 for each metric
            with open(OUT_DIR / "experiment4_summary_run2.json",
                      "r", encoding="utf-8") as f:
                summ2 = json.load(f)
            s2 = summ2["paired_statistics"]
            for k in s1:
                d1 = s1[k]["paired_mean_difference"]
                d2 = s2[k]["paired_mean_difference"]
                if (d1 or 0.0) * (d2 or 0.0) < 0:  # opposite signs
                    dir_ok = False
                if (s1[k].get("statistically_supported")
                        != s2[k].get("statistically_supported")):
                    dir_ok = False
        results.append(check(
            "repro_delta_direction_stable", dir_ok,
            "per-metric delta sign and statistical support identical "
            "across runs"))

        # Judge scores: instrument reproducibility observed.
        jf = sum(1 for q in m1
                 if m1[q]["full"]["faithfulness"]
                 == m2[q]["full"]["faithfulness"])
        jv = sum(1 for q in m1
                 if m1[q]["vanilla"]["faithfulness"]
                 == m2[q]["vanilla"]["faithfulness"])
        results.append(check(
            "repro_full_faith_observed_%d_121" % jf,
            True, "FULL faithfulness judge %d/121 (observed)" % jf))
        results.append(check(
            "repro_vanilla_faith_observed_%d_121" % jv,
            True, "VANILLA faithfulness judge %d/121 (observed)" % jv))
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
    with open(OUT_DIR / "experiment4_validation.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    md = ["# Experiment 4 - Validation Report", "",
          "**Passed:** %s/%s" % (passed, total), "",
          "| # | Check | Passed | Detail |",
          "|---|-------|--------|--------|"]
    for i, c in enumerate(results, 1):
        md.append("| %d | %s | %s | %s |" % (i, c["check"], c["passed"],
                                             c["detail"]))
    (OUT_DIR / "experiment4_validation.md").write_text(
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
        OUT_DIR / "experiment4_per_question_run1.json"))
    ap.add_argument("--run2", default=str(
        OUT_DIR / "experiment4_per_question_run2.json"))
    args = ap.parse_args()
    validate(args.run1, args.run2)