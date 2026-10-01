#!/usr/bin/env python3
"""A1 - Human-vs-LLM-judge agreement (Cohen's kappa + exact match).

Merges the completed human annotation file (a1_human_labels.csv) with the
judge scores captured in the frozen manifest, and reports per-metric exact
match agreement and Cohen's kappa, plus the unsupported-claim confusion.

If the human labels have not yet been provided, prints a "BLOCKED" status
and exits 0. Nothing is written in the blocked state.

Reads only frozen artifacts + the researcher-provided CSV.
"""
import csv
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
MANIFEST = OUT / "a1_sample_manifest.json"
LABELS_CSV = OUT / "a1_human_labels.csv"

METRICS = ["faithfulness", "answer_relevance", "evidence_support"]


def cohens_kappa(a, b):
    n = len(a)
    if n == 0:
        return None
    labels = sorted(set(a) | set(b))
    idx = {v: i for i, v in enumerate(labels)}
    obs = [[0] * len(labels) for _ in labels]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1
    po = sum(obs[i][i] for i in range(len(labels))) / n
    row = [sum(obs[i][j] for j in range(len(labels)))
           for i in range(len(labels))]
    col = [sum(obs[i][j] for i in range(len(labels)))
           for j in range(len(labels))]
    pe = sum((row[i] / n) * (col[i] / n) for i in range(len(labels)))
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def main():
    if not MANIFEST.exists():
        print("BLOCKED: a1_sample_manifest.json missing - run "
              "a1_select_sample.py first")
        return 0
    if not LABELS_CSV.exists():
        print("BLOCKED: awaiting researcher labels. Please save the filled "
              "annotation sheet as a1_human_labels.csv in this folder.")
        print("(No outputs written in blocked state.)")
        return 0

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    judge = {item["question_id"]: item
             for item in manifest["items"]}

    with open(LABELS_CSV, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    populated = [r for r in rows
                 if r.get("human_faithfulness") not in (None, "")]
    if not populated:
        print("BLOCKED: a1_human_labels.csv exists but has no completed "
              "rows yet.")
        return 0

    results = {}
    for metric in METRICS:
        a, b = [], []
        for r in populated:
            qid = r["question_id"]
            if qid not in judge:
                continue
            human = r.get("human_%s" % metric)
            if human in (None, ""):
                continue
            try:
                a.append(float(human))
            except ValueError:
                continue
            b.append(float(judge[qid]["judge_%s" % metric]))
        if not a:
            results[metric] = None
            continue
        match = sum(1 for x, y in zip(a, b) if x == y)
        results[metric] = {
            "n": len(a),
            "exact_match_pct": round(100.0 * match / len(a), 2),
            "cohens_kappa": round(cohens_kappa(a, b), 4)
            if cohens_kappa(a, b) is not None else None,
        }

    # binary unsupported-claim agreement
    tp = fp = tn = fn = 0
    for r in populated:
        qid = r["question_id"]
        if qid not in judge:
            continue
        try:
            h = int(float(r.get("human_unsupported_claim") or 0))
        except ValueError:
            continue
        j = 1 if judge[qid]["judge_hallucination_flag"] else 0
        tp += h == 1 and j == 1
        fp += h == 1 and j == 0
        tn += h == 0 and j == 0
        fn += h == 0 and j == 1

    out = {
        "n_human_annotated": len(populated),
        "n_matched_to_manifest": sum(
            r["question_id"] in judge for r in populated),
        "per_metric": results,
        "unsupported_claim_confusion": {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
            "recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
        },
    }
    (OUT / "a1_human_judge_agreement.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# A1 - Human vs LLM Judge Agreement", ""]
    md.append("n annotated: %d" % out["n_human_annotated"])
    md.append("")
    md.append("| metric | n | exact match % | Cohen's kappa |")
    md.append("|--------|---|---------------|---------------|")
    for metric, v in results.items():
        if v is None:
            continue
        md.append("| %s | %d | %.2f | %s |"
                  % (metric, v["n"], v["exact_match_pct"],
                     v["cohens_kappa"]))
    c = out["unsupported_claim_confusion"]
    md.append("")
    md.append("Unsupported-claim (human 1 vs judge flag 1): precision %s, "
              "recall %s" % (c["precision"], c["recall"]))
    md.append("")
    md.append("*Pre-registered rubric and scale: see "
              "a1_annotation_instructions.md*")
    (OUT / "a1_human_judge_agreement.md").write_text(
        "\n".join(md), encoding="utf-8")
    print("wrote a1_human_judge_agreement.json / .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())