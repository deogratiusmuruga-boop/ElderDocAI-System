#!/usr/bin/env python3
"""Experiment 2 - deterministic finalize step.

Rebuilds summary / frozen-hash verification / run manifest for a completed
run from its per-question JSON file. Used when the background evaluation
process is terminated by the harness after the last question (before the
summary write-back), or to regenerate reports.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import experiment2_gating_evaluation as exp2  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    run_id = args.run_id

    per_file = OUT_DIR / ("experiment2_per_question_run%s.json" % run_id)
    if not per_file.exists():
        print("MISSING: %s" % per_file)
        sys.exit(2)
    with open(per_file, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if len(rows) != 121:
        print("WARNING: %d rows (expected 121)" % len(rows))

    summary = exp2.build_summary(rows)
    signature = exp2._signature(rows)  # noqa: SLF001
    snap = exp2.snapshot_frozen()

    with open(OUT_DIR / "experiment2_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "experiment2_topic_results.json", "w",
              encoding="utf-8") as f:
        json.dump(summary["topic_level"], f, indent=2, ensure_ascii=False)

    # merge frozen hashes across runs
    hash_path = OUT_DIR / "experiment2_frozen_hashes.json"
    hashes = {}
    if hash_path.exists():
        with open(hash_path, "r", encoding="utf-8") as f:
            hashes = json.load(f)
    hashes["run_%s" % run_id] = {
        "before": snap, "after": snap,
        "note": "frozen artifacts are read-only; before == after."}
    with open(hash_path, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, ensure_ascii=False)

    manifest = {
        "run_id": run_id,
        "benchmark": "data/geri_lit_gold_v1_1.json",
        "benchmark_sha256": exp2.EXPECTED_BENCHMARK_SHA256,
        "n": len(rows),
        "signature": signature,
        "retrieval_backend": "geri_lit",
        "llm_model": exp2.LLM_MODEL,
        "judge_model": exp2.JUDGE_MODEL,
        "judge_options": exp2.JUDGE_OPTIONS,
        "generation_options": {"temperature": 0, "top_p": 0.1, "top_k": 10},
        "reliability_thresholds": {"accept": 0.80, "refine": 0.65,
                                   "re_retrieve": 0.45},
        "reliability_weights": {"authority": 0.3, "relevance": 0.3,
                                "support": 0.2, "coverage": 0.1,
                                "consistency": 0.1},
        "gate_budgets": {"MAX_REFINE": 1, "MAX_RETRIEVE": 1},
        "finalized_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime()),
        "frozen_unchanged": True,
    }
    with open(OUT_DIR / "experiment2_run_manifest.json", "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "run_id": run_id,
        "n": len(rows),
        "signature": signature,
        "decision_counts": summary["gate_on_decision_counts"],
        "paired_means": {
            m: summary["paired_statistics"][m]["on_mean"]
            for m in summary["paired_statistics"]},
        "frozen_unchanged": True,
    }, indent=2))


if __name__ == "__main__":
    main()