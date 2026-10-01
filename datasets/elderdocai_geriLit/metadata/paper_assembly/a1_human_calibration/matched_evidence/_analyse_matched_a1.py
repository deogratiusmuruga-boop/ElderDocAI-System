#!/usr/bin/env python3
"""Matched-A1 agreement analysis (deterministic; preregistered protocol).

Verifies the completed annotation sheet against the frozen matched sheet and
the Experiment 1 run-1 artifact, then computes the preregistered agreement
statistics:

  - faithfulness        : human vs Experiment 1 run-1 automated faithfulness
  - answer relevance    : human vs Experiment 1 run-1 automated relevance
  - evidence support    : human vs binned(run-1 continuous span coverage)
  - unsupported claim   : human binary vs run-1 hallucination flag

Support binned scale uses the frozen five-bin mapping; exact midpoint
boundaries go to the lower label. Cohen's kappa follows the preregistered
degeneracy rule (zero-variance marginal = >= 19/20 in one category).

Writes ONLY under paper_assembly/a1_human_calibration/matched_evidence/.
"""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
A1 = OUT.parent
META = A1.parent.parent

FROZEN_SHEET = OUT / "a1_matched_annotation_sheet.csv"
COMPLETED_SHEET = OUT / "a1_matched_annotation_sheet_completed.csv"
PROTOCOL = OUT / "A1_MATCHED_ANNOTATION_PROTOCOL.md"
MANIFEST = OUT / "a1_matched_manifest.json"
RUN1 = META / "experiment1_generation_evaluation" / "experiment1_per_question_run1.json"

GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
SUPPORT_BINS = [[0.0, 0.125, 0.0], [0.125, 0.375, 0.25], [0.375, 0.625, 0.5],
                [0.625, 0.875, 0.75], [0.875, 1.0, 1.0]]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bin_support(x):
    """Map continuous coverage to the frozen five-level bins."""
    x = float(x)
    for lo, hi, label in SUPPORT_BINS:  # noqa: B007 loop var unused
        if x < hi:
            return label
    return 1.0  # covers x == 1.0 (and clamps any upward drift)


def grid_index(v):
    return GRID.index(float(v)) if float(v) in GRID else None


def cohens_kappa(a, b):
    n = len(a)
    labels = sorted(set(a) | set(b))
    idx = {v: i for i, v in enumerate(labels)}
    obs = [[0] * len(labels) for _ in labels]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1
    po = sum(obs[i][i] for i in range(len(labels))) / n
    row = [sum(r) for r in obs]
    col = [sum(obs[i][j] for i in range(len(labels)))
           for j in range(len(labels))]
    pe = sum((row[i] / n) * (col[i] / n) for i in range(len(labels)))
    kappa = 1.0 if pe == 1.0 else (po - pe) / (1.0 - pe)
    return round(kappa, 4), round(po, 4), round(pe, 4)


def marginal(vals):
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    return counts


def run_verification():
    frozen = list(csv.DictReader(open(FROZEN_SHEET, encoding="utf-8-sig")))
    done = list(csv.DictReader(open(COMPLETED_SHEET, encoding="utf-8-sig")))
    run1 = json.loads(RUN1.read_text(encoding="utf-8"))
    by_id = {r["question_id"]: r for r in run1}

    # ---------------- verification ----------------
    check = {}
    check["exactly_20_rows"] = len(done) == 20
    check["no_duplicate_ids"] = (
        len({r["question_id"] for r in done}) == len(done))
    check["identical_id_ordering"] = (
        [r["question_id"] for r in done]
        == [r["question_id"] for r in frozen])
    for field in ["question", "answer", "evidence_full"]:
        check["identical_%s" % field] = all(
            a[field] == b[field] for a, b in zip(done, frozen))
    check["no_evidence_truncation"] = all(
        r["evidence_full"] == "\n".join(by_id[r["question_id"]]
                                        ["retrieved_evidence_texts"])
        for r in done)
    check["expected_columns_only"] = (
        list(done[0].keys())
        == ["question_id", "question", "answer", "evidence_full",
            "human_faithfulness", "human_answer_relevance",
            "human_evidence_support", "human_unsupported_claim", "notes"])
    check["human_columns_populated"] = all(
        r[k].strip() != ""
        for r in done for k in ["human_faithfulness",
                                "human_answer_relevance",
                                "human_evidence_support",
                                "human_unsupported_claim"])
    check["rating_domains_valid"] = (
        all(float(r["human_faithfulness"]) in GRID for r in done)
        and all(float(r["human_answer_relevance"]) in GRID for r in done)
        and all(float(r["human_evidence_support"]) in GRID for r in done)
        and all(r["human_unsupported_claim"] in ("0", "1") for r in done))
    check["all_ids_in_run1"] = all(r["question_id"] in by_id for r in done)
    frozen_sha = sha256(FROZEN_SHEET)
    check["frozen_sheet_sha256"] = frozen_sha
    check["frozen_sheet_unchanged"] = (
        frozen_sha
        == "613261530541d0f55ff387e866bd2afa5c93c8c4d46fd9b248a6233a34e7193f")
    check["all_pass"] = all(
        v is True for k, v in check.items()
        if k != "frozen_sheet_sha256")
    return check, frozen_sha, done, by_id
def agreement(ids, human, by_id):
    def f(metric, transform):
        return ([float(human[q]["human_%s" % metric]) for q in ids],
                [transform(by_id[q]) for q in ids])

    h_f, a_f = f("faithfulness", lambda r: float(r["faithfulness"]))
    h_r, a_r = f("answer_relevance", lambda r: float(r["answer_relevance"]))
    h_s, a_s = f("evidence_support",
                 lambda r: bin_support(r["evidence_support"]))
    h_u = [float(human[q]["human_unsupported_claim"]) for q in ids]
    a_u = [1 if by_id[q]["hallucination"] else 0 for q in ids]
    methods = {"faithfulness": (h_f, a_f),
               "answer_relevance": (h_r, a_r),
               "evidence_support": (h_s, a_s),
               "unsupported_claim": (h_u, a_u)}
    n = len(ids)
    stats = {}
    for metric, (h, a) in methods.items():
        exact = sum(1 for x, y in zip(h, a) if x == y)
        kappa, po, pe = cohens_kappa(h, a)
        deg_h = max(marginal(h).values()) >= 19
        deg_a = max(marginal(a).values()) >= 19
        degenerate = deg_h or deg_a or pe >= 0.90
        out = {"n": n,
               "human_marginal": {str(k): v
                                  for k, v in sorted(marginal(h).items())},
               "automated_marginal": {str(k): v
                                      for k, v in sorted(marginal(a).items())},
               "exact_agreement_count": exact,
               "exact_agreement_pct": round(100.0 * exact / n, 2),
               "Po": po, "Pe": pe,
               "kappa_degenerate": bool(degenerate),
               "kappa": ("degenerate - zero-variance marginal"
                         if degenerate else kappa),
               "degenerate_reasons": [r for r, c in [
                   ("human_marginal_ge19_of_20", deg_h),
                   ("automated_marginal_ge19_of_20", deg_a),
                   ("Pe>=0.90", pe >= 0.90)] if c]}
        if metric != "unsupported_claim":
            within = sum(1 for x, y in zip(h, a)
                         if abs(grid_index(x) - grid_index(y)) <= 1)
            out["within_one_bin_agreement_pct"] = round(
                100.0 * within / n, 2)
        if metric == "unsupported_claim":
            out["confusion"] = {
                "tp": sum(1 for x, y in zip(h, a) if x == 1 and y == 1),
                "fp": sum(1 for x, y in zip(h, a) if x == 1 and y == 0),
                "tn": sum(1 for x, y in zip(h, a) if x == 0 and y == 0),
                "fn": sum(1 for x, y in zip(h, a) if x == 0 and y == 1)}
        stats[metric] = out
    return stats


def render_md(a):
    lines = [
        "# A1 - Matched-Evidence Human-Judge Agreement Analysis (frozen protocol)",
        "",
        "Protocol: `%s` (v1.0-matched-evidence)" % a["protocol_version"],
        "",
        "- Source annotation sheet: `%s`" % a["source_annotation_sheet_path"],
        "- Annotation-sheet SHA-256: `%s`" % a["annotation_sheet_sha256"][:16],
        "- Experiment 1 source: `%s`" % a["experiment1_source_path"],
        "- n = %d" % a["n"],
        "- Verification all-pass: **%s**" % (a["verification_checks"]
                                             .get("all_pass")),
        "",
        "## Verification checks",
        "",
    ]
    for k, v in a["verification_checks"].items():
        if k == "frozen_sheet_sha256":
            lines.append("- %s: `%s`" % (k, v[:16]))
        else:
            lines.append("- %s: %s" % (k, v))
    lines += ["", "## Agreement statistics (n = %d)" % a["n"], "",
              "| metric | exact n | exact % | Po | Pe | within-1-bin % | "
              "Cohen's kappa |",
              "|--------|---------|---------|----|----|--------------|"
              "--------------|"]
    for m, s in a["statistics"].items():
        lines.append("| %s | %d | %.1f | %.4f | %.4f | %s | %s |"
                     % (m, s["exact_agreement_count"],
                        s["exact_agreement_pct"], s["Po"], s["Pe"],
                        s.get("within_one_bin_agreement_pct"), s["kappa"]))
    lines += ["", "### κ degeneracy note", ""]
    lines.append(
        "Per the preregistered rule, kappa is reported as degenerate when "
        "a rater marginal is zero-variance (>= 19/20 in one category) or "
        "Pe >= 0.90. Degenerate kappa is not interpreted as ordinary "
        "disagreement; exact agreement / Po / Pe are reported instead.")
    lines += ["", "### Human vs automated marginals", "",
              "| metric | human marginal | automated marginal |", ""]
    for m, s in a["statistics"].items():
        lines.append("| %s | %s | %s |"
                     % (m, s["human_marginal"], s["automated_marginal"]))
    lines += ["", "### Unsupported-claim confusion (human flag vs run-1 "
                  "hallucination flag)", ""]
    c = a["statistics"]["unsupported_claim"]["confusion"]
    lines.append("| TP | FP | TN | FN |")
    lines.append("|----|----|----|----|")
    lines.append("| %d | %d | %d | %d |" % (c["tp"], c["fp"], c["tn"],
                                            c["fn"]))
    lines += ["", "- Analysis timestamp (UTC): %s"
              % a["analysis_timestamp_utc"],
              "- Support-bin mapping: %s" % a["support_bin_mapping"], ""]
    (OUT / "a1_matched_agreement_analysis.md").write_text(
        "\n".join(lines), encoding="utf-8")


def main():
    check, frozen_sha, done, by_id = run_verification()
    ids = [r["question_id"] for r in done]
    human = {r["question_id"]: r for r in done}
    stats = agreement(ids, human, by_id)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    analysis = {
        "protocol_version": manifest.get("protocol_version"),
        "source_annotation_sheet_path": str(COMPLETED_SHEET),
        "annotation_sheet_sha256": sha256(COMPLETED_SHEET),
        "matched_protocol_path": str(PROTOCOL),
        "experiment1_source_path": str(RUN1),
        "experiment1_source_sha256": sha256(RUN1),
        "n": len(ids),
        "question_ids": ids,
        "support_bin_mapping": SUPPORT_BINS,
        "support_bin_rule": (
            "exact midpoint boundaries are assigned to the lower label"),
        "kappa_degeneracy_rule": (
            "zero-variance marginal (>= 19/20 in one category) or Pe >= 0.90 "
            "=> kappa reported as degenerate; numerical kappa only otherwise"),
        "verification_checks": check,
        "statistics": stats,
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
    }
    (OUT / "a1_matched_agreement_analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    render_md(analysis)
    print("verification all_pass:", check["all_pass"], "| frozen sha:",
          frozen_sha[:16])
    for m, s in stats.items():
        print(m, "| exact%%: %.2f | Po: %.4f | Pe: %.4f | kappa: %s"
              % (s["exact_agreement_pct"], s["Po"], s["Pe"], s["kappa"]))
    print("wrote a1_matched_agreement_analysis.json")


if __name__ == "__main__":
    main()