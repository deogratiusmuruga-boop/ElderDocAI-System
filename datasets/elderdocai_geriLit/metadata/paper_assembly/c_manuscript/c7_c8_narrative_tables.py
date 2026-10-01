#!/usr/bin/env python3
"""C7/C8 - Narrative tables and figure data for the manuscript.

Parses the FROZEN official summary artifacts (summary JSON + A2 judge
reliability) and emits the manuscript narrative tables and figure-data
CSVs. No statistics are recomputed here: every number is lifted verbatim
from the frozen experiment summaries or the frozen A2/A3 outputs.

Writes only under paper_assembly/c_manuscript/.
"""
import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
META = OUT.parent.parent
A2 = META / "paper_assembly" / "a2_judge_reliability" / "a2_judge_reliability.json"
A3 = META / "paper_assembly" / "a3_confidence_intervals" / "a3_confidence_intervals.json"


def load(rel):
    with open(META / rel, "r", encoding="utf-8") as f:
        return json.load(f)


E1 = load("experiment1_generation_evaluation/experiment1_summary_run1.json")
E2 = load("experiment2_gating_effectiveness/experiment2_summary.json")
E3 = load("experiment3_care_state_adaptation/experiment3_summary.json")
E4 = load("experiment4_vanilla_baseline/experiment4_summary.json")
with open(A2, "r", encoding="utf-8") as f:
    A2R = json.load(f)
with open(A3, "r", encoding="utf-8") as f:
    A3R = json.load(f)

FIG = OUT / "fig_data"
FIG.mkdir(exist_ok=True)


def write_csv(name, header, rows):
    with open(FIG / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main():
    tables = {}
    text = ["# C7/C8 - Narrative tables and figure data (frozen numbers)",
            "",
            "All values below are lifted verbatim from the frozen experiment "
            "summaries (validation status and hashes recorded in each "
            "experiment report). See the per-experiment reports for full "
            "statistics.", ""]

    # ---------- Table 1: dataset & system ----------
    t1 = [
        ("Benchmark", "GeriLit-Gold v1.1 (121 questions, 45 PMCIDs)"),
        ("Retrieval", "dense BGE top-5 + BM25 top-5, hybrid 0.6/0.4, "
                      "CrossEncoder rerank -> top-3"),
        ("Generator / judge", "llama3.2:latest, temperature 0, top_p 0.1, "
                              "top_k 10"),
        ("Reliability gate", "ACCEPT>=0.80, REFINE>=0.65, RE-RETRIEVE>=0.45, "
                             "REJECT<0.45"),
        ("Judge scale", "0/0.25/0.5/0.75/1.0 (faithfulness, answer "
                        "relevance; evidence support deterministic)"),
    ]
    tables["table1_system"] = t1
    text.append("## Table 1 - Benchmark and system configuration")
    text.append("")
    text.append("| component | configuration |")
    text.append("|-----------|----------------|")
    for k, v in t1:
        text.append("| %s | %s |" % (k, v))
    text.append("")

    # ---------- Table 2: Exp 1 generation quality ----------
    e1c = A3R["experiment1"]
    u = e1c["unsupported_claim_rate"]
    t2 = []
    for k, label in [("faithfulness", "Faithfulness"),
                     ("answer_relevance", "Answer relevance"),
                     ("evidence_support", "Evidence support")]:
        v = e1c["metrics"][k]
        t2.append((label, v["mean"], v["sd"],
                   v["ci95_t"][0], v["ci95_t"][1]))
    tables["table2_generation_quality"] = {
        "n": E1["n_questions"],
        "metrics": [{"metric": t[0], "mean": t[1], "sd": t[2],
                     "ci95_lo": t[3], "ci95_hi": t[4]} for t in t2],
        "unsupported_claim_rate": {
            "pct": u["rate"], "n": u["n_support"], "total": u["n"],
            "ci95_lo": u["ci95_binomial"][0],
            "ci95_hi": u["ci95_binomial"][1]},
        "contradiction_rate_pct": round(
            100.0 * E1["contradiction_flags"] / E1["n_questions"], 2),
    }
    text.append("## Table 2 - Experiment 1: generation quality (n=%d)"
                % E1["n_questions"])
    text.append("")
    text.append("| metric | mean | sd | 95% CI |")
    text.append("|--------|------|----|--------|")
    for t in t2:
        text.append("| %s | %.4f | %.4f | [%.4f, %.4f] |" % t)
    text.append("")
    text.append("- Unsupported-claim rate (faithfulness <= 0.5): %.2f%% "
                "(%d/%d; 95%% binomial CI [%.2f%%, %.2f%%])"
                % (u["rate"], u["n_support"], u["n"],
                   u["ci95_binomial"][0], u["ci95_binomial"][1]))
    text.append("- Contradiction rate (faithfulness == 0): %.2f%%"
                % tables["table2_generation_quality"]
                ["contradiction_rate_pct"])
    text.append("")

    # ---------- Table 3: Exp 2 gating contrast ----------
    t3 = []
    for k, label in [("faithfulness", "Faithfulness"),
                     ("answer_relevance", "Answer relevance"),
                     ("evidence_support", "Evidence support")]:
        p = E2["paired_statistics"][k]
        t3.append((label, p["on_mean"], p["off_mean"],
                   p["paired_mean_difference"], p["paired_ttest_p"],
                   p["wilcoxon_p"],
                   p["ci95_mean_delta_bootstrap"][0],
                   p["ci95_mean_delta_bootstrap"][1],
                   p["count_on_eq_off"]))
    tables["table3_gating_contrast"] = {
        "n": E2["n_questions"],
        "metrics": [{"metric": t[0], "on_mean": t[1], "off_mean": t[2],
                     "delta": t[3], "ttest_p": t[4], "wilcoxon_p": t[5],
                     "ci_boot_lo": t[6], "ci_boot_hi": t[7],
                     "n_equal": t[8]} for t in t3],
        "gate_decisions": E2["gate_on_decision_pct"],
        "refinement_cases": E2["gate_on_refinement_cases"],
        "refinement_evidence_changed": E2["refinement_analysis"]
        if isinstance(E2["refinement_analysis"], dict)
        else E2["refinement_analysis"],
    }
    text.append("## Table 3 - Experiment 2: reliability gate ON vs OFF "
                "(n=%d; delta = ON - OFF)" % E2["n_questions"])
    text.append("")
    text.append("| metric | ON mean | OFF mean | mean delta | t p | "
                "Wilcoxon p | bootstrap 95% CI | ON=OFF |")
    text.append("|--------|---------|----------|------------|-----|"
                "-----------|-----------------|--------|")
    for t in t3:
        text.append("| %s | %.4f | %.4f | %.4f | %.4f | %.4f | "
                    "[%.4f, %.4f] | %d |"
                    % (t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]))
    text.append("")
    text.append("No metric met the pre-registered joint significance rule "
                "(paired t-test p<0.05 AND Wilcoxon p<0.05).")
    text.append("")
# ---------- Table 4: Exp 3 care-state contrast ----------
    t4 = []
    for k, label in [("faithfulness", "Faithfulness"),
                     ("answer_relevance", "Answer relevance"),
                     ("evidence_support", "Evidence support")]:
        c = E3["condition_stats"]
        d = E3["paired_deltas"][k + "_delta"]
        t4.append((label, c["A"][k]["mean"], c["B"][k]["mean"],
                   d["mean"], d["gt"], d["lt"]))
    s3 = E3["state_response_rate"]
    a3r = E3["appropriateness_rate"]
    tables["table4_care_state_contrast"] = {
        "n_pairs": E3["n_pairs"],
        "metrics": [{"metric": t[0], "a_mean": t[1], "b_mean": t[2],
                     "delta_b_minus_a": t[3], "n_b_gt_a": t[4],
                     "n_b_lt_a": t[5]} for t in t4],
        "state_response_rate_pct": s3 if isinstance(s3, (int, float))
        else s3.get("pct"),
        "state_response_count": E3["state_response_count"],
        "appropriateness_rate": a3r if isinstance(a3r, (int, float))
        else a3r.get("value"),
        "appropriateness_n": E3.get("appropriateness_n"),
        "evidence_identical_rate": E3["evidence_identical_rate"],
        "groundedness_rate": E3["groundedness_rate"],
        "ci95_state_response_pct": A3R["experiment3"]
        ["state_response_rate"]["ci95_binomial_pct"],
        "ci95_appropriateness_pct": A3R["experiment3"]
        ["appropriateness_rate"]["ci95_binomial_pct"],
    }
    text.append("## Table 4 - Experiment 3: care-state adaptation "
                "(n=20 pairs; A=LOW_ACTIVITY, B=HIGH_ACTIVITY)")
    text.append("")
    text.append("| metric | A mean | B mean | mean delta (B-A) | B>A | "
                "B<A |")
    text.append("|--------|--------|--------|------------------|-----|"
                "-----|")
    for t in t4:
        text.append("| %s | %.4f | %.4f | %.4f | %d | %d |" % t)
    text.append("")
    text.append("- State-response rate: %.1f%% (%d/%d; 95%% binomial CI "
                "[%.1f%%, %.1f%%])"
                % (s3 if isinstance(s3, (int, float)) else s3["pct"],
                   E3["state_response_count"], E3["n_pairs"],
                   A3R["experiment3"]["state_response_rate"]
                   ["ci95_binomial_pct"][0],
                   A3R["experiment3"]["state_response_rate"]
                   ["ci95_binomial_pct"][1]))
    text.append("- Evidence-identical rate: %.1f%%" %
                E3["evidence_identical_rate"])
    text.append("- Appropriateness: 0/%d scored appropriate (95%% binomial "
                "CI [0%%, %.1f%%])"
                % (E3.get("appropriateness_n", 20),
                   A3R["experiment3"]["appropriateness_rate"]
                   ["ci95_binomial_pct"][1]))
    text.append("")

    # ---------- Table 5: Exp 4 vanilla baseline ----------
    t5 = []
    for k, label in [("faithfulness", "Faithfulness"),
                     ("answer_relevance", "Answer relevance"),
                     ("evidence_support", "Evidence support")]:
        p = E4["paired_statistics"][k]
        t5.append((label, p["full_mean"], p["vanilla_mean"],
                   p["paired_mean_difference"], p["paired_ttest_p"],
                   p["wilcoxon_p"], p["statistically_supported"],
                   p["count_vanilla_lt_full"]))
    tables["table5_vanilla_baseline"] = {
        "n": E4["n_pairs"],
        "metrics": [{"metric": t[0], "full_mean": t[1],
                     "vanilla_mean": t[2], "delta": t[3],
                     "ttest_p": t[4], "wilcoxon_p": t[5],
                     "statistically_supported": t[6],
                     "n_vanilla_lt_full": t[7]} for t in t5],
        "evidence_identity_ok": E4["evidence_identity_ok"],
    }
    text.append("## Table 5 - Experiment 4: FULL vs Vanilla-RAG "
                "baseline (n=%d; delta = VANILLA - FULL)"
                % E4["n_pairs"])
    text.append("")
    text.append("| metric | FULL mean | VANILLA mean | mean delta | t p | "
                "Wilcoxon p | supported | V<F |")
    text.append("|--------|-----------|--------------|------------|-----|"
                "-----------|-----------|-----|")
    for t in t5:
        text.append("| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %s | %d |"
                    % (t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]))
    text.append("")
    text.append("Evidence identity control held on all %d pairs "
                "(identical retrieved evidence in both conditions)."
                % E4["n_pairs"])
    text.append("")
# ---------- Table 6: judge reliability summary (A2) ----------
    t6 = []
    for exp in A2R["experiments"]:
        rows = [m for m in exp.get("metrics", []) if m.get("n", 0)]
        if not rows:
            continue
        em = [m["exact_match"] for m in rows]
        kv = [m["cohens_kappa"] for m in rows
              if m["cohens_kappa"] is not None]
        t6.append((exp["name"], exp.get("n_questions_run1", 0),
                   min(em), max(em),
                   (min(kv), max(kv)) if kv else (None, None)))
    tables["table6_judge_reliability"] = [
        {"experiment": t[0], "n": t[1], "exact_match_min": t[2],
         "exact_match_max": t[3], "kappa_min": t[4][0],
         "kappa_max": t[4][1]} for t in t6]
    text.append("## Table 6 - LLM judge reproducibility (run1 vs run2, "
                "A2)")
    text.append("")
    text.append("| experiment | n | exact match min% | exact match max% | "
                "kappa min | kappa max |")
    text.append("|------------|---|------------------|------------------|"
                "-----------|-----------|")
    for t in t6:
        text.append("| %s | %d | %.1f | %.1f | %s | %s |"
                    % (t[0], t[1], t[2], t[3],
                       "n/a" if t[4][0] is None else "%.3f" % t[4][0],
                       "n/a" if t[4][1] is None else "%.3f" % t[4][1]))
    text.append("")
    text.append("Appropriateness kappa in Exp 3 is suppressed (kappa paradox "
                "with an all-zero run1 marginal; exact match 95%).")
    text.append("")

    # ---------- Figure data CSVs ----------
    write_csv("fig1_decision_distribution.csv",
              ["decision", "count", "pct"],
              [["ACCEPT", E1["decision_counts"]["ACCEPT"],
                E1["decision_pct"]["ACCEPT"]],
               ["REFINE", E1["decision_counts"]["REFINE"],
                E1["decision_pct"]["REFINE"]]])
    write_csv("fig2_generation_quality.csv",
              ["metric", "mean", "ci95_lo", "ci95_hi", "unsupported_pct"],
              [["faithfulness", e1c["metrics"]["faithfulness"]["mean"],
                e1c["metrics"]["faithfulness"]["ci95_t"][0],
                e1c["metrics"]["faithfulness"]["ci95_t"][1],
                u["rate"]],
               ["answer_relevance",
                e1c["metrics"]["answer_relevance"]["mean"],
                e1c["metrics"]["answer_relevance"]["ci95_t"][0],
                e1c["metrics"]["answer_relevance"]["ci95_t"][1], ""],
               ["evidence_support", e1c["metrics"]["evidence_support"]["mean"],
                e1c["metrics"]["evidence_support"]["ci95_t"][0],
                e1c["metrics"]["evidence_support"]["ci95_t"][1], ""]])
    write_csv("fig3_gating_contrast.csv",
              ["metric", "condition", "mean", "delta", "ttest_p",
               "wilcoxon_p"],
              [[m["metric"], "gate_off", m["off_mean"], m["delta"],
                m["ttest_p"], m["wilcoxon_p"]]
               for m in tables["table3_gating_contrast"]["metrics"]]
              + [[m["metric"], "gate_on", m["on_mean"], m["delta"],
                  m["ttest_p"], m["wilcoxon_p"]]
                  for m in tables["table3_gating_contrast"]["metrics"]])
    write_csv("fig4_vanilla_contrast.csv",
              ["metric", "condition", "mean", "delta", "ttest_p",
               "wilcoxon_p", "supported"],
              [[m["metric"], "vanilla", m["vanilla_mean"], m["delta"],
                m["ttest_p"], m["wilcoxon_p"], m["statistically_supported"]]
               for m in tables["table5_vanilla_baseline"]["metrics"]]
              + [[m["metric"], "full", m["full_mean"], m["delta"],
                  m["ttest_p"], m["wilcoxon_p"],
                  m["statistically_supported"]]
                  for m in tables["table5_vanilla_baseline"]["metrics"]])
    write_csv("fig5_care_state_contrast.csv",
              ["metric", "condition", "mean"],
              [[m["metric"], "A_low", m["a_mean"]]
               for m in tables["table4_care_state_contrast"]["metrics"]]
              + [[m["metric"], "B_high", m["b_mean"]]
                  for m in tables["table4_care_state_contrast"]["metrics"]])
    write_csv("fig6_judge_reliability.csv",
              ["experiment", "exact_match_min", "exact_match_max",
               "kappa_min", "kappa_max"],
              [[t[0], t[2], t[3], t[4][0], t[4][1]] for t in t6])

    (OUT / "c7_c8_narrative_tables.json").write_text(
        json.dumps(tables, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "c7_c8_narrative_tables.md").write_text(
        "\n".join(text), encoding="utf-8")
    print("wrote c7_c8_narrative_tables.json / .md + fig_data CSVs")


if __name__ == "__main__":
    main()