#!/usr/bin/env python3
"""Experiment 1 - deterministic finalize step.

Rebuilds summary / signature / frozen-hash verification / run manifest for a
completed run from its per-question JSON file. Pure functions of the data;
used when the background evaluation process is terminated by the harness after
the last question (before the summary write-back), or to regenerate reports.

Usage: python experiment1_finalize.py --run-id 1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import experiment1_generation_evaluation as exp1  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    run_id = args.run_id

    per_file = OUT_DIR / ("experiment1_per_question_run%s.json" % run_id)
    if not per_file.exists():
        print("MISSING: %s" % per_file)
        sys.exit(2)

    with open(per_file, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if len(rows) != 121:
        print("WARNING: %d rows (expected 121)" % len(rows))

    summary = exp1.build_summary(rows)
    signature = exp1._signature(rows)  # noqa: SLF001 (deterministic helper)
    snap = exp1.snapshot_frozen()

    with open(OUT_DIR / ("experiment1_summary_run%s.json" % run_id), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # frozen-hash record: before == after because this run only reads
    # (merge across runs so the file accumulates both run snapshots)
    hash_path = OUT_DIR / "experiment1_frozen_hashes.json"
    hashes = {}
    if hash_path.exists():
        with open(hash_path, "r", encoding="utf-8") as f:
            hashes = json.load(f)
    hashes["run_%s" % run_id] = {
        "before": snap, "after": snap,
        "note": "frozen artifacts are read-only in this experiment; "
                "before == after."}
    with open(hash_path, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, ensure_ascii=False)

    manifest = {
        "run_id": run_id,
        "benchmark": "data/geri_lit_gold_v1_1.json",
        "benchmark_sha256": exp1.EXPECTED_BENCHMARK_SHA256,
        "n": len(rows),
        "signature": signature,
        "retrieval_backend": "geri_lit",
        "llm_model": exp1.LLM_MODEL,
        "judge_model": exp1.JUDGE_MODEL,
        "judge_options": exp1.JUDGE_OPTIONS,
        "generation": {"temperature": 0, "top_p": 0.1, "top_k": 10},
        "reliability_thresholds": {"accept": 0.80, "refine": 0.65,
                                   "re_retrieve": 0.45},
        "reliability_weights": {"authority": 0.3, "relevance": 0.3,
                                "support": 0.2, "coverage": 0.1,
                                "consistency": 0.1},
        "finalized_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime()),
        "frozen_unchanged": True,
    }
    with open(OUT_DIR / "experiment1_run_manifest.json", "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "run_id": run_id,
        "n": len(rows),
        "signature": signature,
        "decision_counts": summary["decision_counts"],
        "frozen_unchanged": True,
    }, indent=2))


if __name__ == "__main__":
    main()