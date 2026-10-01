#!/usr/bin/env python3
"""C7/C8 - Render publication-quality figures from fig_data CSVs.

Reads the frozen figure-data CSVs (written by c7_c8_narrative_tables.py)
and renders 300-dpi PNGs into c_manuscript/figures/. Restrained two-colour
palette, no chartjunk, no titles (captions live in the manuscript text).
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
FIGDATA = OUT / "fig_data"
FIGDIR = OUT / "figures"
FIGDIR.mkdir(exist_ok=True)

C1 = "#1F6FB2"   # FULL / gate ON / condition A
C2 = "#E18727"   # VANILLA / gate OFF / condition B
INK = "#111111"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def read_csv(name):
    with open(FIGDATA / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def grouped_bars(metrics, values_a, values_b, labels, fname, colors,
                 ylabel="mean score", ylim=(0.0, 1.0), annotate=None):
    """values_a/values_b: lists (per metric) of (mean, ci_lo, ci_hi)."""
    import numpy as np
    x = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for vals, col, lab in [(values_a, colors[0], labels[0]),
                           (values_b, colors[1], labels[1])]:
        off = w / 2 if lab == labels[1] else -w / 2
        means = [v[0] for v in vals]
        lo = [v[0] - v[1] for v in vals]
        hi = [v[2] - v[0] for v in vals]
        ax.bar(x + off, means, width=w, color=col, label=lab,
               edgecolor=INK, linewidth=0.6)
        if any(v[1] != 0 or v[2] != 0 for v in vals):
            ax.errorbar(x + off, means, yerr=[lo, hi], fmt="none",
                        ecolor=INK, capsize=3, linewidth=0.9)
    if annotate:
        ax.text(0, -0.14, annotate[0], transform=ax.transAxes,
                fontsize=8, ha="left", va="top", wrap=True)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / fname, bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/%s" % fname)


def fig1():
    rows = read_csv("fig1_decision_distribution.csv")
    labels = [r["decision"] for r in rows]
    counts = [int(r["count"]) for r in rows]
    pcts = [float(r["pct"]) for r in rows]
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.barh(labels, counts, color=[C1, C2], edgecolor=INK,
            linewidth=0.6, height=0.6)
    for i, (c, p) in enumerate(zip(counts, pcts)):
        ax.text(c + 1, i, "%d (%.1f%%)" % (c, p), va="center",
                fontsize=8)
    ax.set_xlim(0, max(counts) * 1.25)
    ax.set_xlabel("number of questions (n=121)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_decision_distribution.png",
                bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/fig1_decision_distribution.png")


def fig2():
    rows = read_csv("fig2_generation_quality.csv")
    metrics = [r["metric"].replace("_", " ") for r in rows]
    vals = [(float(r["mean"]), float(r["ci95_lo"]), float(r["ci95_hi"]))
            for r in rows]
    grouped_bars(metrics, vals, vals, ["score", "score"],
                 "fig2_generation_quality.png", colors=[C1, C1],
                 ylim=(0.0, 1.0))
def fig3():
    rows = read_csv("fig3_gating_contrast.csv")
    metrics = sorted({r["metric"] for r in rows})
    on = [(float(r["mean"]), 0.0, 0.0) for r in rows
          if r["condition"] == "gate_on"]
    off = [(float(r["mean"]), 0.0, 0.0) for r in rows
           if r["condition"] == "gate_off"]
    grouped_bars(metrics, off, on, ["gate OFF", "gate ON"],
                 "fig3_gating_contrast.png", colors=[C2, C1], ylim=(0.0, 1.0),
                 annotate=["Exp 2: no metric met the joint significance "
                           "rule (paired t and Wilcoxon p >= 0.05); "
                           "scores equal for 110-115 of 121 questions."])


def fig4():
    rows = read_csv("fig4_vanilla_contrast.csv")
    metrics = sorted({r["metric"] for r in rows})
    full = [(float(r["mean"]), 0.0, 0.0) for r in rows
            if r["condition"] == "full"]
    van = [(float(r["mean"]), 0.0, 0.0) for r in rows
           if r["condition"] == "vanilla"]
    grouped_bars(metrics, van, full, ["Vanilla-RAG", "FULL framework"],
                 "fig4_vanilla_contrast.png", colors=[C2, C1],
                 ylim=(0.0, 1.0),
                 annotate=["Exp 4: FULL > VANILLA on all three metrics; "
                           "every paired delta statistically supported "
                           "(t and Wilcoxon p < 0.05)."])


def fig5():
    rows = read_csv("fig5_care_state_contrast.csv")
    metrics = sorted({r["metric"] for r in rows})
    a = [(float(r["mean"]), 0.0, 0.0) for r in rows
         if r["condition"] == "A_low"]
    b = [(float(r["mean"]), 0.0, 0.0) for r in rows
         if r["condition"] == "B_high"]
    grouped_bars(metrics, a, b, ["LOW_ACTIVITY", "HIGH_ACTIVITY"],
                 "fig5_care_state_contrast.png", colors=[C1, C2],
                 ylim=(0.0, 1.0),
                 annotate=["Exp 3: 35% state-response (7/20), 0/20 judged "
                           "appropriate; evidence identical across states "
                           "(100%)."])


def fig6():
    rows = read_csv("fig6_judge_reliability.csv")
    exps = [r["experiment"] for r in rows]
    lo = [float(r["exact_match_min"]) for r in rows]
    hi = [float(r["exact_match_max"]) for r in rows]
    import numpy as np
    x = np.arange(len(exps))
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for xi, l, h in zip(x, lo, hi):
        ax.plot([xi, xi], [l, h], color="black", linewidth=1.6,
                marker="o", markersize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(["exp1", "exp2", "exp3", "exp4"])
    ax.set_ylim(80, 102)
    ax.set_ylabel("run1 vs run2 exact match (%)")
    ax.set_xlabel("experiment")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig6_judge_reliability.png",
                bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/fig6_judge_reliability.png")


if __name__ == "__main__":
    print("rendering figures -> figures/")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print("done")