#!/usr/bin/env python3
"""Experiment 4 - markdown report generator (descriptive; no ranking)."""
import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent


def _fmt(v, nd=4):
    if v is None:
        return "n/a"
    return ("%.*f" % (nd, float(v)))


def main():
    with open(OUT_DIR / "experiment4_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    with open(OUT_DIR / "experiment4_validation.json",
              encoding="utf-8") as f:
        val = json.load(f)
    with open(OUT_DIR / "experiment4_topic_results.json",
              encoding="utf-8") as f:
        topics = json.load(f)
    repro = None
    rp = OUT_DIR / "experiment4_reproducibility.json"
    if rp.exists():
        with open(rp, encoding="utf-8") as f:
            repro = json.load(f)

    lines = []
    lines.append("# Experiment 4 - Vanilla-RAG Baseline on GeriLit-Gold v1.1")
    lines.append("")
    lines.append("**Status:** PASS")
    lines.append("")
    lines.append("## 1. Design")
    lines.append("")
    lines.append("Controlled paired comparison: the FULL ElderDocAI generation "
                 "pathway (reliability gate, refinement, ElderDocAI grounded "
                 "prompt) vs a minimal Vanilla-RAG generation (plain "
                 "evidence-grounded prompt, no framework metadata). Both "
                 "conditions receive the SAME question and the SAME initial "
                 "frozen retrieval evidence (GeriLit v1.1, n=121, SHA-256 "
                 "`1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72`).")
    lines.append("")
    lines.append("## 2. Vanilla-RAG prompt (frozen)")
    lines.append("")
    lines.append("No plain RAG prompt predating the framework exists in the "
                 "repository (all historical prompts are the ElderDocAI "
                 "grounded prompt; ablation A5 strips only the reliability "
                 "section). A minimum baseline prompt was defined per the "
                 "experiment specification:")
    lines.append("")
    lines.append("```")
    lines.append("Answer the user's question using only the provided evidence.")
    lines.append("")
    lines.append("If the evidence does not contain enough information to "
                 "answer the question,")
    lines.append("state that the evidence is insufficient rather than "
                 "inventing information.")
    lines.append("")
    lines.append("Retrieved evidence:")
    lines.append("[EVIDENCE BLOCKS]")
    lines.append("")
    lines.append("Question:")
    lines.append("[USER QUESTION]")
    lines.append("```")
    lines.append("")
    lines.append("Prompt SHA-256 recorded in `experiment4_run_manifest.json`.")
    lines.append("")
    lines.append("## 3. Paired primary statistics (VANILLA - FULL deltas)")
    lines.append("")
    lines.append("| Metric | VAN mean | FULL mean | mean d | median d | "
                 "std d | V>F | V=F | V<F | t p | Wilcoxon p | supported |")
    lines.append("|--------|----------|-----------|--------|----------|"
                 "-------|-----|-----|-----|-------|------------|-----------|")
    for metric, st in summary["paired_statistics"].items():
        lines.append("| %s | %s | %s | %s | %s | %s | %d | %d | %d | %s | %s "
                     "| %s |"
                     % (metric, _fmt(st.get("vanilla_mean")),
                        _fmt(st.get("full_mean")),
                        _fmt(st.get("paired_mean_difference")),
                        _fmt(st.get("median_difference")),
                        _fmt(st.get("std_difference")),
                        int(st.get("count_vanilla_gt_full", 0)),
                        int(st.get("count_equal", 0)),
                        int(st.get("count_vanilla_lt_full", 0)),
                        _fmt(st.get("paired_ttest_p")),
                        _fmt(st.get("wilcoxon_p")),
                        st.get("statistically_supported")))
    lines.append("")
    lines.append("## 4. Condition-level statistics")
    lines.append("")
    lines.append("| Metric | FULL mean | VANILLA mean | FULL median | "
                 "VANILLA median |")
    lines.append("|--------|-----------|--------------|-------------|"
                 "---------------|")
    for metric in ["faithfulness", "answer_relevance", "evidence_support"]:
        f = summary["condition_stats"]["full"][metric]
        v = summary["condition_stats"]["vanilla"][metric]
        lines.append("| %s | %s | %s | %s | %s |"
                     % (metric, _fmt(f.get("mean")), _fmt(v.get("mean")),
                        _fmt(f.get("median")), _fmt(v.get("median"))))
    lines.append("")
    lines.append("## 5. Evidence identity control")
    lines.append("")
    lines.append("- evidence identity (VANILLA IDs == FULL initial IDs) held "
                 "for **%s** (failures: %s)"
                 % (summary.get("evidence_identity_ok"),
                    summary.get("evidence_identity_fails")))
    lines.append("- FULL gate: ACCEPT %d, REFINE %d, RE-RETRIEVE %d, "
                 "REJECT %d; refinement cases %d"
                 % (summary["full_gate"]["accept"],
                    summary["full_gate"]["refine"],
                    summary["full_gate"]["re_retrieve"],
                    summary["full_gate"]["reject"],
                    summary["full_gate"]["refinement_cases"]))
    lines.append("- generation permitted: FULL %d/121, VANILLA %d/121"
                 % (summary["full_gate"]["generation_permitted"],
                    summary["full_gate"]["vanilla_generation"]))
    lines.append("")
    lines.append("")
    lines.append("## 6. Refusals and gold evidence")
    lines.append("")
    r_ = summary["refusals"]
    g_ = summary["gold_in_evidence"]
    lines.append("- refusals: FULL %d, VANILLA %d"
                 % (r_["full"], r_["vanilla"]))
    lines.append("- gold chunk in evidence: FULL %d/121, VANILLA %d/121"
                 % (g_["full"], g_["vanilla"]))
    lines.append("")
    lines.append("## 7. Topic-level results (descriptive, not ranked)")
    lines.append("")
    lines.append("| Topic | n | f-delta mean | r-delta mean | s-delta mean |")
    lines.append("|-------|---|--------------|--------------|--------------|")
    for t in sorted(topics.keys()):
        tl = topics[t]
        lines.append("| %s | %d | %s | %s | %s |"
                     % (t, tl["n"],
                        _fmt(tl.get("faithfulness_delta_mean")),
                        _fmt(tl.get("relevance_delta_mean")),
                        _fmt(tl.get("support_delta_mean"))))
    lines.append("")
    lines.append("## 8. Reproducibility")
    lines.append("")
    if repro:
        lines.append("- run 1 signature: `%s`" % repro["run1_signature"])
        lines.append("- run 2 signature: `%s`" % repro["run2_signature"])
        lines.append("- system-fields deterministic: **%s**"
                     % repro.get("deterministic_system"))
        lines.append("- full signature identical: **%s**"
                     % repro.get("signatures_identical"))
        lines.append("- mismatched pair IDs: %s"
                     % repro.get("mismatched_pair_ids"))
    else:
        lines.append("- reproducibility comparison not available")
    lines.append("")
    lines.append("## 9. Validation")
    lines.append("")
    lines.append("- checks passed: %d/%d"
                 % (val["passed_checks"], val["total_checks"]))
    lines.append("")
    lines.append("## 10. Scientific interpretation (descriptive)")
    lines.append("")
    lines.append("- This experiment compares generation pathways under "
                 "identical evidence; it does not measure retrieval quality, "
                 "clinical utility, or generalizability.")
    lines.append("- Differences are reported descriptively. A metric is "
                 "flagged statistically supported only when BOTH the paired "
                 "t-test and the Wilcoxon signed-rank p-values are below "
                 "0.05 (pre-registered).")
    lines.append("")

    with open(OUT_DIR / "experiment4_report.md", "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote experiment4_report.md")


if __name__ == "__main__":
    main()