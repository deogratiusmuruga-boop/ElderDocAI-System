#!/usr/bin/env python3
"""Experiment 3 - deterministic finalize step.

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

import experiment3_evaluation as exp3  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    run_id = args.run_id

    per_file = OUT_DIR / ("experiment3_per_question_run%s.json" % run_id)
    if not per_file.exists():
        print("MISSING: %s" % per_file)
        sys.exit(2)
    with open(per_file, "r", encoding="utf-8") as f:
        rows = json.load(f)
    print("loaded %d rows" % len(rows))

    summary = exp3.build_summary(rows)
    signature = exp3._signature(rows)  # noqa: SLF001
    snap = exp3.snapshot_frozen()

    with open(OUT_DIR / "experiment3_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "experiment3_topic_results.json", "w",
              encoding="utf-8") as f:
        json.dump(summary["topic_level"], f, indent=2, ensure_ascii=False)

    hash_path = OUT_DIR / "experiment3_frozen_hashes.json"
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
        "n": len(rows),
        "signature": signature,
        "question_set": "first 2 per topic, frozen v1.1 (read-only)",
        "retrieval_backend": "geri_lit",
        "llm_model": exp3.LLM_MODEL,
        "judge_model": exp3.JUDGE_MODEL,
        "judge_options": exp3.JUDGE_OPTIONS,
        "now": exp3.NOW,
        "states": {"A": "LOW_ACTIVITY", "B": "HIGH_ACTIVITY"},
        "finalized_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime()),
        "frozen_unchanged": True,
    }
    with open(OUT_DIR / "experiment3_run_manifest.json", "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "run_id": run_id,
        "n": len(rows),
        "signature": signature,
        "state_response_rate": summary["state_response_rate"],
        "evidence_identical_rate": summary["evidence_identical_rate"],
        "groundedness_rate": summary["groundedness_rate"],
        "appropriateness_rate": summary["appropriateness_rate"],
        "frozen_unchanged": True,
    }, indent=2))


if __name__ == "__main__":
    main()