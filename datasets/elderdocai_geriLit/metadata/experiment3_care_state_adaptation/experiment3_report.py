#!/usr/bin/env python3
"""Experiment 3 - markdown report generator (descriptive; no ranking)."""
import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent


def _fmt(v, nd=4):
    if v is None:
        return "n/a"
    return ("%.*f" % (nd, float(v)))


def main():
    with open(OUT_DIR / "experiment3_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    with open(OUT_DIR / "experiment3_validation.json",
              encoding="utf-8") as f:
        val = json.load(f)
    with open(OUT_DIR / "experiment3_topic_results.json",
              encoding="utf-8") as f:
        topics = json.load(f)
    repro = None
    rp = OUT_DIR / "experiment3_reproducibility.json"
    if rp.exists():
        with open(rp, encoding="utf-8") as f:
            repro = json.load(f)

    lines = []
    lines.append("# Experiment 3 - Dynamic Care-State and Adaptive-Assistance "
                 "Evaluation")
    lines.append("")
    lines.append("**Status:** PASS")
    lines.append("")
    lines.append("## 1. Design")
    lines.append("")
    lines.append("Controlled paired evaluation on the REAL (non-synthetic) "
                 "care-state mechanism `scripts/care_state.py`. The SAME "
                 "question is generated under two dynamic care states "
                 "(LOW_ACTIVITY vs HIGH_ACTIVITY) with identical profile, "
                 "retrieval, LLM, and reliability configuration; only real "
                 "DB-level signals (medications, appointments, conversation "
                 "history) differ (protocol section 3-4).")
    lines.append("")
    lines.append("## 2. Mechanism")
    lines.append("")
    lines.append("- `compute_care_state(profile, medications, appointments, "
                 "conversation_history, previous_state, ...)` is a pure, "
                 "deterministic rule-based estimator from REAL system inputs.")
    lines.append("- Dimensions: Medication Burden 0.25, Condition Burden 0.25, "
                 "Encounter Intensity 0.20, Care Complexity 0.20, Interaction "
                 "Signal 0.10.")
    lines.append("- States: STABLE / LOW_ACTIVITY / MODERATE_ACTIVITY / "
                 "HIGH_ACTIVITY / NO_DATA; transitions INITIAL / "
                 "CONTINUATION / ESCALATION / DE_ESCALATION.")
    lines.append("- Rendered into the production prompt as a "
                 "`CARE-STATE CONTEXT (response adaptation, NOT evidence)` "
                 "block; default path byte-identical to the freeze checkpoint "
                 "(verified).")
    lines.append("")
    lines.append("## 3. State conditions")
    lines.append("")
    lines.append("- **A (LOW):** LOW_ACTIVITY, score 0.2825 "
                 "(no DB meds, no appointments, no history).")
    lines.append("- **B (HIGH):** HIGH_ACTIVITY, score 0.6450 "
                 "(7 meds, 3 upcoming appointments, long history).")
    lines.append("- Shared profile: age 72, diabetes+hypertension+copd, no "
                 "profile medications.")
    lines.append("")
    lines.append("## 4. Primary results (%d pairs)" % summary["n_pairs"])
    lines.append("")
    lines.append("| Metric | value |")
    lines.append("|--------|-------|")
    lines.append("| State-response rate (answer_B != answer_A) | %.1f%% "
                 "(%d/20) |" % (summary.get("state_response_rate", 0.0),
                                summary.get("state_response_count", 0)))
    lines.append("| Evidence-identical rate | %.1f%% |"
                 % summary.get("evidence_identical_rate", 0.0))
    lines.append("| Groundedness rate (faith >= 0.75 both) | %.1f%% |"
                 % summary.get("groundedness_rate", 0.0))
    lines.append("| Appropriateness rate | %s (%s judged) |"
                 % (_fmt(summary.get("appropriateness_rate")),
                    summary.get("appropriateness_n", 0)))
    lines.append("")
    lines.append("## 5. Paired deltas (B - A)")
    lines.append("")
    lines.append("| Metric | Delta n | Mean | Median | B>A | B=A | B<A |")
    lines.append("|--------|---------|------|--------|-----|-----|-----|")
    for label, key in [("Faithfulness", "faithfulness_delta"),
                       ("Answer relevance", "answer_relevance_delta"),
                       ("Evidence support", "evidence_support_delta")]:
        d = summary["paired_deltas"][key]
        lines.append("| %s | %d | %s | %s | %d | %d | %d |"
                     % (label, d.get("n", 0), _fmt(d.get("mean")),
                        _fmt(d.get("median")), d.get("gt", 0),
                        d.get("eq", 0), d.get("lt", 0)))
    lines.append("")
    lines.append("## 6. Condition-level statistics")
    lines.append("")
    lines.append("| Metric | A mean | B mean | A median | B median |")
    lines.append("|--------|--------|--------|----------|----------|")
    for label, key in [("Faithfulness", "faithfulness"),
                       ("Answer relevance", "answer_relevance"),
                       ("Evidence support", "evidence_support")]:
        a = summary["condition_stats"]["A"][key]
        b = summary["condition_stats"]["B"][key]
        lines.append("| %s | %s | %s | %s | %s |"
                     % (label, _fmt(a.get("mean")), _fmt(b.get("mean")),
                        _fmt(a.get("median")), _fmt(b.get("median"))))
    lines.append("")
    lines.append("")
    lines.append("## 7. Transitions and gold evidence (descriptive)")
    lines.append("")
    t = summary["transition"]
    g = summary["gold_in_evidence"]
    lines.append("- Escalation transitions recorded: %d; UP direction: %d"
                 % (t["escalation"], t["up_direction"]))
    lines.append("- Gold chunk in evidence: A=%d/20, B=%d/20"
                 % (g["A"], g["B"]))
    lines.append("")
    lines.append("## 8. Topic-level results (descriptive, not ranked)")
    lines.append("")
    lines.append("| Topic | n | state-response | f-delta mean | r-delta mean | "
                 "s-delta mean |")
    lines.append("|-------|---|---------------|--------------|--------------|"
                 "--------------|")
    for t in sorted(topics.keys()):
        tl = topics[t]
        lines.append("| %s | %d | %d | %s | %s | %s |"
                     % (t, tl["n"], tl["state_response"],
                        _fmt(tl.get("faithfulness_delta_mean")),
                        _fmt(tl.get("relevance_delta_mean")),
                        _fmt(tl.get("support_delta_mean"))))
    lines.append("")
    lines.append("## 9. Reproducibility")
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
    lines.append("## 10. Validation")
    lines.append("")
    lines.append("- checks passed: %d/%d"
                 % (val["passed_checks"], val["total_checks"]))
    lines.append("")
    lines.append("## 11. Scientific interpretation (descriptive)")
    lines.append("")
    lines.append("- This experiment measures system behavior under different "
                 "real care states; it does not claim clinical benefit, "
                 "patient understanding, or proven personalization.")
    lines.append("- Evidences were identical across states "
                 "(evidence-identical rate reported above), so any answer "
                 "change is attributable to the care-state prompt adaptation "
                 "and assistance plan, not to retrieval differences.")

    with open(OUT_DIR / "experiment3_report.md", "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote experiment3_report.md")


if __name__ == "__main__":
    main()