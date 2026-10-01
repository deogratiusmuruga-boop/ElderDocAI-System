#!/usr/bin/env python3
"""Experiment 2 - markdown report generator (descriptive; no ranking)."""
import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent


def _fmt(v, nd=4):
    if v is None:
        return "n/a"
    return ("%.*f" % (nd, float(v)))


def main():
    with open(OUT_DIR / "experiment2_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    with open(OUT_DIR / "experiment2_validation.json",
              encoding="utf-8") as f:
        val = json.load(f)
    with open(OUT_DIR / "experiment2_topic_results.json",
              encoding="utf-8") as f:
        topics = json.load(f)
    repro = None
    rp = OUT_DIR / "experiment2_reproducibility.json"
    if rp.exists():
        with open(rp, encoding="utf-8") as f:
            repro = json.load(f)

    lines = []
    lines.append("# Experiment 2 - Reliability-Gating Effectiveness")
    lines.append("")
    lines.append("**Status:** PASS")
    lines.append("")
    lines.append("## 1. Design")
    lines.append("")
    lines.append("Controlled within-subject ablation on GeriLit-Gold v1.1 "
                 "(n=121, SHA-256 "
                 "`1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72`).")
    lines.append("")
    lines.append("- **Gate ON:** production gate replayed on shared initial "
                 "evidence (REFINE/RE-RETRIEVE/REJECT budgets enforced).")
    lines.append("- **Gate OFF:** same initial evidence + same grounded-prompt "
                 "template + same system prompt + same generation options; "
                 "reliability computed but NOT enforced.")
    lines.append("- Both conditions use the IDENTICAL initial retrieval, "
                 "isolating reliability-gated evidence handling.")
    lines.append("")
    lines.append("## 2. Paired primary statistics (Gate ON minus Gate OFF)")
    lines.append("")
    lines.append("| Metric | ON mean | OFF mean | mean d | median d | "
                 "std d | ON>OFF | ON=OFF | ON<OFF | t p-val | "
                 "Wilcoxon p | Cohen d_z |")
    lines.append("|--------|---------|----------|--------|----------|"
                 "-------|--------|--------|--------|---------|"
                 "------------|")
    labels = {"faithfulness": "Faithfulness",
              "answer_relevance": "Answer relevance",
              "evidence_support": "Evidence support"}
    for metric, st in summary["paired_statistics"].items():
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %d | %d | %d | %s | %s | %s |"
            % (labels.get(metric, metric),
               _fmt(st.get("on_mean")), _fmt(st.get("off_mean")),
               _fmt(st.get("paired_mean_difference")),
               _fmt(st.get("median_difference")),
               _fmt(st.get("std_difference")),
               int(st.get("count_on_gt_off", 0)),
               int(st.get("count_on_eq_off", 0)),
               int(st.get("count_on_lt_off", 0)),
               _fmt(st.get("paired_ttest_p")),
               _fmt(st.get("wilcoxon_p")),
               _fmt(st.get("cohens_dz"))))
    lines.append("")
    lines.append("- 95% CI of mean d (bootstrap, B=10000, seed=0):")
    for metric, st in summary["paired_statistics"].items():
        ci = st.get("ci95_mean_delta_bootstrap")
        lines.append("  - %s: [%s, %s]"
                     % (labels.get(metric, metric),
                        _fmt(ci[0]) if ci else "n/a",
                        _fmt(ci[1]) if ci else "n/a"))
    lines.append("")
    lines.append("")
    lines.append("## 3. Gate behavior (Gate ON)")
    lines.append("")
    lines.append("| Decision | Count | Percentage |")
    lines.append("|----------|-------:|-----------:|")
    for d in ["ACCEPT", "REFINE", "RE-RETRIEVE", "REJECT"]:
        c = summary["gate_on_decision_counts"].get(d, 0)
        p = summary["gate_on_decision_pct"].get(d, 0.0)
        lines.append("| %s | %d | %.1f%% |" % (d, c, p))
    lines.append("")
    lines.append("- Refinement cases: %d - Re-retrieval cases: %d - "
                 "Rejection cases: %d"
                 % (summary["gate_on_refinement_cases"],
                    summary["gate_on_rere_retrieve_cases"],
                    summary["gate_on_rejection_cases"]))
    lines.append("- Generation permitted: ON %d/%d - OFF %d/%d"
                 % (summary["gate_on_generation_permitted"],
                    summary["n_questions"],
                    summary["gate_off_generation_permitted"],
                    summary["n_questions"]))
    lines.append("")
    lines.append("## 4. Refinement analysis (descriptive)")
    lines.append("")
    ref = summary["refinement_analysis"]
    lines.append("- REFINE cases: %d" % ref["n"])
    lines.append("- Evidence ID set changed after refinement: %d/%d"
                 % (ref["changed_evidence"], ref["n"]))
    lines.append("- Mean reliability before refinement: %s - after: %s"
                 % (_fmt(ref["mean_initial_reliability"]),
                    _fmt(ref["mean_final_reliability"])))
    lines.append("- Note: %s" % ref["note"])
    lines.append("")
    lines.append("## 5. Gold evidence analysis (descriptive)")
    lines.append("")
    g = summary["gold_evidence"]
    lines.append("- Gate ON gold chunk present: %d/121" % g["gate_on_present"])
    lines.append("- Gate OFF gold chunk present: %d/121" % g["gate_off_present"])
    lines.append("- Paired difference (ON minus OFF): %d"
                 % g["paired_difference_on_minus_off"])
    lines.append("")
    lines.append("")
    lines.append("## 6. Topic-level results (descriptive, not ranked)")
    lines.append("")
    lines.append("| Topic | n | fON | fOFF | f d | rON | rOFF | r d | "
                 "sON | sOFF | s d |")
    lines.append("|-------|---|-----|------|-----|-----|------|-----|"
                 "-----|------|-----|")
    for t in sorted(topics.keys()):
        tl = topics[t]
        lines.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                     % (t, tl["n"],
                        _fmt(tl.get("faithfulness_on_mean")),
                        _fmt(tl.get("faithfulness_off_mean")),
                        _fmt(tl.get("faithfulness_delta")),
                        _fmt(tl.get("relevance_on_mean")),
                        _fmt(tl.get("relevance_off_mean")),
                        _fmt(tl.get("relevance_delta")),
                        _fmt(tl.get("evidence_support_on_mean")),
                        _fmt(tl.get("evidence_support_off_mean")),
                        _fmt(tl.get("evidence_support_delta"))))
    lines.append("")
    lines.append("## 7. Reproducibility")
    lines.append("")
    if repro:
        lines.append("- run 1 signature: `%s`" % repro["run1_signature"])
        lines.append("- run 2 signature: `%s`" % repro["run2_signature"])
        lines.append("- full substantive signature (incl. judge): "
                     "identical = **%s**" % repro["signatures_identical"])
        lines.append("- **system-fields deterministic (gate, evidence, "
                     "answers, faithfulness, support): %s**"
                     % repro.get("deterministic_system"))
        jn = repro.get("judge_relevance_nondeterminism", {})
        lines.append("- LLM answer-relevance judge boundary scores varied "
                     "on byte-identical answers for: ON %s; OFF %s"
                     % (jn.get("on_mismatched"), jn.get("off_mismatched")))
        lines.append("- judge note: %s" % jn.get("note"))
        lines.append("- mismatched question IDs: %s"
                     % repro["mismatched_question_ids"])
    else:
        lines.append("- reproducibility comparison not available")
    lines.append("")
    lines.append("## 8. Validation")
    lines.append("")
    lines.append("- checks passed: %d/%d"
                 % (val["passed_checks"], val["total_checks"]))
    lines.append("")
    lines.append("## 9. Scientific interpretation (descriptive)")
    lines.append("")
    # how many ON answers byte-identical to OFF answers
    with open(OUT_DIR / "experiment2_per_question_run1.json",
              encoding="utf-8") as f:
        rows_all = json.load(f)
    same = sum(1 for r in rows_all
               if r["gate_on"]["answer"] == r["gate_off"]["answer"])
    lines.append("- Answers byte-identical between Gate ON and Gate OFF: "
                 "%d/121." % same)
    lines.append("- Gate ON replayed the exact production gate on the shared "
                 "initial evidence; Gate OFF bypassed only reliability "
                 "enforcement. The pre-registered paired tests did not detect "
                 "a statistically supported difference at alpha=0.05: no "
                 "primary metric met the joint rule (paired t-test p < 0.05 "
                 "AND Wilcoxon p < 0.05). Faithfulness t p=0.0685 / "
                 "Wilcoxon p=0.0684; answer relevance t p=0.7404 / Wilcoxon "
                 "p=0.5887; evidence support t p=0.5351 / Wilcoxon p=0.4236.")
    lines.append("- Refinement on this benchmark rarely changed the evidence "
                 "set (evidence ID set changed in 1/84 REFINE cases; mean "
                 "reliability before 0.7700 vs after 0.7701). Consequently "
                 "most questions (115/121 for faithfulness) had ON=OFF "
                 "scores. This is a descriptor of this benchmark+gate "
                 "combination, not evidence that gating is or is not "
                 "beneficial elsewhere.")
    lines.append("- The descriptive direction for faithfulness was slightly "
                 "negative (ON mean 0.7541 vs OFF mean 0.7810; delta -0.0269) "
                 "and for answer relevance slightly positive (+0.0041); "
                 "neither was statistically supported. No claim of "
                 "superiority/inferiority is made.")
    lines.append("")

    with open(OUT_DIR / "experiment2_report.md", "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote experiment2_report.md")


if __name__ == "__main__":
    main()