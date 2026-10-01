#!/usr/bin/env python3
"""A3 - Journal-grade confidence intervals for frozen headline statistics.

1. Experiment 1 (n=121): 95% CI for mean faithfulness, answer relevance,
   evidence support, and the unsupported-claim rate (binomial Clopper-Pearson
   for the proportion). t-interval and bootstrap percentile CI.
2. Experiment 3 (n=20): exact binomial 95% CIs for the state-response rate
   and appropriateness rate (small-sample proportions).
3. Experiment 4 (n=121): 95% CI for the paired deltas.

Reads only frozen artifacts; writes only under paper_assembly/a3_confidence_intervals/.
"""
import json
import math
import random
import statistics
from pathlib import Path

from scipy import stats as st

OUT = Path(__file__).resolve().parent
META = Path(__file__).resolve().parent.parent.parent

EXP1_JSON = (META / "experiment1_generation_evaluation"
             / "experiment1_per_question_run1.json")
EXP3_JSON = (META / "experiment3_care_state_adaptation"
             / "experiment3_per_question_run1.json")
EXP4_JSON = (META / "experiment4_vanilla_baseline"
             / "experiment4_per_question_run1.json")


def t_interval(mean, sd, n, alpha=0.05):
    if n <= 1 or sd is None:
        return None
    z = 1.959964
    se = sd / math.sqrt(n)
    return (round(mean - z * se, 6), round(mean + z * se, 6))


def bootstrap_ci(values, b=10000, seed=0, alpha=0.05):
    rng = random.Random(seed)
    means = []
    for _ in range(b):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int((alpha / 2) * b)]
    hi = means[int((1 - alpha / 2) * b)]
    return (round(lo, 6), round(hi, 6))


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial outcome-proportion 95% CI (Clopper-Pearson)."""
    if k == 0:
        lo = 0.0
    else:
        lo = 1.0 / (1.0 + (n - k + 1)
                    / (k * st.f.ppf(alpha / 2, 2 * k, 2 * (n - k + 1))))
    if k == n:
        hi = 1.0
    else:
        hi = 1.0 / (1.0 + (n - k)
                    / ((k + 1) * st.f.ppf(1 - alpha / 2, 2 * (k + 1),
                                           2 * (n - k))))
    return (round(lo, 6), round(hi, 6))
def main():
    out = {}

    # ---------------- Experiment 1 ----------------
    with open(EXP1_JSON, "r", encoding="utf-8") as f:
        rows1 = json.load(f)
    exp1 = {"n": len(rows1), "metrics": {}}
    for key in ["faithfulness", "answer_relevance", "evidence_support"]:
        vals = [float(r[key]) for r in rows1 if r.get(key) is not None]
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals)
        exp1["metrics"][key] = {
            "n": len(vals),
            "mean": round(mean, 6),
            "sd": round(sd, 6),
            "ci95_t": t_interval(mean, sd, len(vals)),
            "ci95_bootstrap": bootstrap_ci(vals),
        }
    uns = sum(1 for r in rows1 if r.get("hallucination"))
    lohi = clopper_pearson(uns, len(rows1))
    exp1["unsupported_claim_rate"] = {
        "n_support": uns, "n": len(rows1),
        "rate": round(100.0 * uns / len(rows1), 2),
        "ci95_binomial": (round(100 * lohi[0], 2), round(100 * lohi[1], 2)),
    }
    out["experiment1"] = exp1

    # ---------------- Experiment 3 (small-sample rates) ----------------
    with open(EXP3_JSON, "r", encoding="utf-8") as f:
        rows3 = json.load(f)
    n3 = len(rows3)
    resp = sum(1 for r in rows3 if r["adaptation_judgment"])
    appr = sum(1 for r in rows3 if r["appropriateness_judgment"] == 1)
    lohi_r = clopper_pearson(resp, n3)
    lohi_a = clopper_pearson(appr, n3)
    out["experiment3"] = {
        "n": n3,
        "state_response_rate": {
            "k": resp, "n": n3,
            "rate_pct": round(100.0 * resp / n3, 2),
            "ci95_binomial_pct": (round(100 * lohi_r[0], 2),
                                  round(100 * lohi_r[1], 2)),
        },
        "appropriateness_rate": {
            "k": appr, "n": n3,
            "rate_pct": round(100.0 * appr / n3, 2),
            "ci95_binomial_pct": (round(100 * lohi_a[0], 2),
                                  round(100 * lohi_a[1], 2)),
        },
    }

    # ---------------- Experiment 4 (headline deltas) ----------------
    with open(EXP4_JSON, "r", encoding="utf-8") as f:
        rows4 = json.load(f)
    exp4 = {"n": len(rows4), "metrics": {}}
    for key, dkey in [("faithfulness", "faithfulness_delta"),
                      ("answer_relevance", "answer_relevance_delta"),
                      ("evidence_support", "evidence_support_delta")]:
        deltas = [float(r["paired_differences"][dkey]) for r in rows4
                  if r["paired_differences"].get(dkey) is not None]
        mean = statistics.mean(deltas)
        sd = statistics.stdev(deltas)
        exp4["metrics"][key] = {
            "n": len(deltas),
            "mean_delta": round(mean, 6),
            "sd": round(sd, 6),
            "ci95_t": t_interval(mean, sd, len(deltas)),
            "ci95_bootstrap": bootstrap_ci(deltas),
        }
    out["experiment4"] = exp4

    with open(OUT / "a3_confidence_intervals.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    md = ["# A3 - Confidence Intervals (frozen results)", "",
          "## Experiment 1 (n=121)", "",
          "| metric | mean | sd | 95% CI (t) | 95% CI (bootstrap) |",
          "|--------|------|----|-------------|--------------------|"]
    for k, v in exp1["metrics"].items():
        md.append("| %s | %.4f | %.4f | [%.4f, %.4f] | [%.4f, %.4f] |"
                  % (k, v["mean"], v["sd"], v["ci95_t"][0],
                     v["ci95_t"][1], v["ci95_bootstrap"][0],
                     v["ci95_bootstrap"][1]))
    u = exp1["unsupported_claim_rate"]
    md.append("")
    md.append("Unsupported-claim rate: %.2f%% (95%% binomial CI "
              "[%.2f%%, %.2f%%])" % (u["rate"] or 0.0,
                                      u["ci95_binomial"][0],
                                      u["ci95_binomial"][1]))
    md.append("")
    md.append("## Experiment 3 (n=20, exact binomial 95% CI)")
    md.append("")
    for label in ["state_response_rate", "appropriateness_rate"]:
        v = out["experiment3"][label]
        md.append("- %s: %.0f/%d = %.2f%% (95%% CI [%.2f%%, %.2f%%])"
                  % (label, v["k"], v["n"], v["rate_pct"],
                     v["ci95_binomial_pct"][0], v["ci95_binomial_pct"][1]))
    md.append("")
    md.append("## Experiment 4 (n=121 paired deltas, VANILLA - FULL)")
    md.append("")
    md.append("| metric | mean delta | sd | 95% CI (t) |")
    md.append("|--------|------------|----|------------|")
    for k, v in exp4["metrics"].items():
        md.append("| %s | %.4f | %.4f | [%.4f, %.4f] |"
                  % (k, v["mean_delta"], v["sd"],
                     v["ci95_t"][0], v["ci95_t"][1]))
    (OUT / "a3_confidence_intervals.md").write_text(
        "\n".join(md), encoding="utf-8")
    print("wrote a3_confidence_intervals.json / .md")


if __name__ == "__main__":
    main()