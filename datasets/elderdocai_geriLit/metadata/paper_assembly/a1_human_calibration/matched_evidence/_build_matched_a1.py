#!/usr/bin/env python3
"""Build the matched-evidence A1 human-calibration artifacts (read-only source).

Derives the annotation sheet and manifest from the FROZEN Experiment 1 run-1
per-question artifact for the EXACT 20 question IDs of the original A1 pilot
selection (seed 42, stratified 11 high / 4 partial / 5 low).

Evidence matching rule: for each question the human sees
``evidence_full = "\\n".join(retrieved_evidence_texts)`` EXACTLY as recorded
(no truncation, no reordering, no added text). These are the full final
evidence texts the Experiment 1 automated judge consumed (the judge's
build_evidence_text adds presentation scaffolding only; the underlying text
is identical).

This script writes ONLY under
paper_assembly/a1_human_calibration/matched_evidence/. It performs NO
agreement computation and fills NO human annotation fields.
"""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

A1 = Path(__file__).resolve().parent.parent        # a1_human_calibration
OUT = Path(__file__).resolve().parent              # matched_evidence
META = A1.parent.parent                            # geriLit metadata root
REPO = META.parent.parent.parent                   # repository root

RUN1 = (META / "experiment1_generation_evaluation"
        / "experiment1_per_question_run1.json")
PILOT_MANIFEST = A1 / "a1_sample_manifest.json"
BENCHMARK = REPO / "data" / "geri_lit_gold_v1_1.json"

PROTOCOL_VERSION = "1.0-matched-evidence"
SEED = 42
N = 20
COLS = ["question_id", "question", "answer", "evidence_full",
        "human_faithfulness", "human_answer_relevance",
        "human_evidence_support", "human_unsupported_claim", "notes"]
SUPPORT_BINS = [[0.0, 0.125, 0.0], [0.125, 0.375, 0.25], [0.375, 0.625, 0.5],
                [0.625, 0.875, 0.75], [0.875, 1.0, 1.0]]
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    rows = json.loads(RUN1.read_text(encoding="utf-8"))
    by_id = {r["question_id"]: r for r in rows}
    pilot = json.loads(PILOT_MANIFEST.read_text(encoding="utf-8"))
    ids = [it["question_id"] for it in pilot["items"]]

    assert len(ids) == N, "pilot manifest must contain %d ids" % N
    for qid in ids:
        assert qid in by_id, "question %s missing from run1" % qid

    sheet = []
    for qid in ids:
        rec = by_id[qid]
        evidence_texts = rec.get("retrieved_evidence_texts") or []
        sheet.append({
            "question_id": qid,
            "question": rec["question"],
            "answer": rec["answer"],
            "evidence_full": "\n".join(evidence_texts),
            "human_faithfulness": "",
            "human_answer_relevance": "",
            "human_evidence_support": "",
            "human_unsupported_claim": "",
            "notes": "",
        })

    csv_path = OUT / "a1_matched_annotation_sheet.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for row in sheet:
            w.writerow(row)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": SEED,
        "n": N,
        "selected_question_ids": ids,
        "source_experiment": "Experiment 1",
        "source_run": "run1",
        "source_artifact": str(RUN1),
        "source_artifact_sha256": sha256(RUN1),
        "benchmark_sha256": sha256(BENCHMARK),
        "evidence_matching_statement": (
            "For each question the annotator is shown evidence_full = "
            "\"\\n\".join(retrieved_evidence_texts) EXACTLY as recorded in "
            "the frozen Experiment 1 run-1 artifact: the same full final "
            "evidence texts, in the same order, without truncation, "
            "summarization, paraphrase, reordering, or added text. These are "
            "the evidence texts the Experiment 1 automated judge consumed."),
        "evidence_ordering_statement": (
            "Evidence items appear in the exact recorded order of "
            "retrieved_evidence_texts in the run-1 artifact."),
        "blinding_statement": (
            "The sheet contains only question_id, question, answer, "
            "evidence_full, and blank human rating fields. It exposes NO "
            "automated judge scores, judge reasoning, stratum, topic, gate "
            "decision, reliability, gold labels, or experiment condition."),
        "support_bin_thresholds": {
            "rule": (
                "Automated continuous span-token coverage is mapped to the "
                "five-point scale; exact midpoint boundaries are assigned to "
                "the lower label."),
            "bins": [{"lo": b[0], "hi": b[1], "label": b[2]}
                     for b in SUPPORT_BINS]},
        "kappa_degeneracy_rule": (
            "If either rater has a zero-variance marginal (>= 19 of 20 "
            "observations in one category), report: 'kappa = degenerate - "
            "zero-variance marginal' plus observed agreement (Po), expected "
            "agreement (Pe), and exact agreement. A degenerate kappa is NOT "
            "interpreted as ordinary disagreement. Numerical kappa is "
            "reported only when no marginal is degenerate and Pe < 0.90. "
            "This rule applies symmetrically to all evaluated categorical "
            "metrics."),
        "pilot_relationship": (
            "The original A1 pilot remains preserved as a historical "
            "methodological pilot (abridged evidence). The matched A1 is the "
            "primary human-judge calibration analysis; it supersedes the "
            "pilot for that purpose and is not a second independent "
            "estimate."),
        "no_validation_claim": (
            "This is an evaluator-concordance/calibration analysis. It does "
            "not validate or invalidate the judge; no claim about judge "
            "accuracy, precision, bias, or clinical validity is drawn from "
            "n=20."),
        "human_fields_blank": True,
        "no_agreement_computed": True,
        "confirmation": (
            "Human annotation fields are intentionally blank and no "
            "agreement statistics have been computed."),
        "created_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
    }
    man_path = OUT / "a1_matched_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    print("wrote", csv_path.name, "rows:", len(sheet))
    print("wrote", man_path.name)
    print("sheet sha256:", sha256(csv_path))
    print("manifest sha256:", sha256(man_path))


if __name__ == "__main__":
    main()