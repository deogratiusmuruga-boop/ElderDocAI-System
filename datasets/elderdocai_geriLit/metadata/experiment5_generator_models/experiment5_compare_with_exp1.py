#!/usr/bin/env python3
"""Read-only: compare an Experiment 5 per-question file with Experiment 1 run N."""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
META = HERE.parent


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp5-file", required=True,
                    help="file name inside the Experiment 5 folder")
    ap.add_argument("--exp1-run", default="1")
    a = ap.parse_args()

    hits = sorted(META.glob(
        "experiment1_*/experiment1_per_question_run%s.json" % a.exp1_run))
    if len(hits) != 1:
        sys.exit("Expected exactly one Experiment 1 file, found: %s" % hits)
    ref = {r["question_id"]: r for r in load(hits[0])}
    new = load(HERE / a.exp5_file)

    common = [r for r in new if r["question_id"] in ref]
    ev_ordered = [r["question_id"] for r in common
                  if r["retrieved_evidence_ids"]
                  != ref[r["question_id"]]["retrieved_evidence_ids"]]
    ev_set = [r["question_id"] for r in common
              if set(r["retrieved_evidence_ids"])
              != set(ref[r["question_id"]]["retrieved_evidence_ids"])]
    ans_same = sum(1 for r in common
                   if (r["answer"] or "").strip()
                   == (ref[r["question_id"]]["answer"] or "").strip())
    dec_same = sum(1 for r in common
                   if r["decision"] == ref[r["question_id"]]["decision"])
    faith_same = sum(1 for r in common
                     if r["faithfulness"]
                     == ref[r["question_id"]]["faithfulness"])
    gold_new = sum(1 for r in common if r.get("gold_chunk_retrieved"))
    gold_ref = sum(1 for r in common
                   if ref[r["question_id"]].get("gold_chunk_retrieved"))

    print("Experiment 1 reference file:", hits[0].name)
    print("records in Experiment 5 file:", len(new), "| compared:", len(common))
    print("evidence-ID mismatches (ordered):", len(ev_ordered), ev_ordered[:5])
    print("evidence-ID mismatches (as sets):", len(ev_set), ev_set[:5])
    print("identical answers: %d/%d" % (ans_same, len(common)))
    print("identical decisions: %d/%d" % (dec_same, len(common)))
    print("identical faithfulness scores: %d/%d" % (faith_same, len(common)))
    print("gold chunk retrieved: Experiment 5 %d vs Experiment 1 %d (of %d)"
          % (gold_new, gold_ref, len(common)))


if __name__ == "__main__":
    main()