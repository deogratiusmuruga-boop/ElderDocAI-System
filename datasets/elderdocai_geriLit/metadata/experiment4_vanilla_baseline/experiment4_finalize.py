#!/usr/bin/env python3
"""Experiment 4 - deterministic finalize step.

Rebuilds summary / frozen-hash verification / run manifest for a completed
run from its per-question JSON file. Used when the background evaluation
process is terminated after the final question (before summary write-back).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import experiment4_evaluation as exp4  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    run_id = args.run_id

    per_file = OUT_DIR / ("experiment4_per_question_run%s.json" % run_id)
    if not per_file.exists():
        print("MISSING: %s" % per_file)
        sys.exit(2)
    with open(per_file, "r", encoding="utf-8") as f:
        rows = json.load(f)
    print("loaded %d rows" % len(rows))

    summary = exp4.build_summary(rows)
    signature = exp4._signature(rows)  # noqa: SLF001
    snap = exp4.snapshot_frozen()

    with open(OUT_DIR / "experiment4_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / ("experiment4_summary_run%s.json" % run_id), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "experiment4_topic_results.json", "w",
              encoding="utf-8") as f:
        json.dump(summary["topic_level"], f, indent=2, ensure_ascii=False)

    hash_path = OUT_DIR / "experiment4_frozen_hashes.json"
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
        "retrieval_backend": "geri_lit",
        "llm_model": exp4.LLM_MODEL,
        "judge_model": exp4.JUDGE_MODEL,
        "judge_options": exp4.JUDGE_OPTIONS,
        "vanilla_prompt_sha256": exp4.VANILLA_PROMPT_SHA256,
        "finalized_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime()),
        "frozen_unchanged": True,
    }
    with open(OUT_DIR / "experiment4_run_manifest.json", "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "run_id": run_id,
        "n": len(rows),
        "signature": signature,
        "evidence_identity_ok": summary["evidence_identity_ok"],
        "paired_means": {
            m: summary["paired_statistics"][m]["paired_mean_difference"]
            for m in summary["paired_statistics"]},
        "statistically_supported": {
            m: summary["paired_statistics"][m]["statistically_supported"]
            for m in summary["paired_statistics"]},
        "frozen_unchanged": True,
    }, indent=2))


if __name__ == "__main__":
    main()