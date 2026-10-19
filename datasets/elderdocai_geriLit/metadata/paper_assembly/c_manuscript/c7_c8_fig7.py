#!/usr/bin/env python3
"""Figure 7 - Experiment 5 generator-model comparison (two-run means).

Renders a publication-quality PNG with matplotlib only (no PyMuPDF).

Panel A: grouped bars for faithfulness / answer relevance / evidence support
         (two-run arithmetic mean of the run-level means, sourced from the
         frozen experiment5_*_summary_run{1,2}.json files).
Panel B: faithfulness-based flag count summed over the two runs.

Notes:
- Phi-3.5 faithfulness / answer-relevance / evidence-support means use n=120
  per run (one failed generation per run, GLG11-001); the flag denominator is
  n=121 per run.
- Flags are evaluator-defined faithfulness <= 0.50 by the Llama-3.2 judge,
  NOT independently verified hallucinations.
- No statistical significance is implied.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
SRC = (OUT / ".." / ".." / "experiment5_generator_models").resolve()
FIGDIR = OUT / "figures"
FIGDIR.mkdir(exist_ok=True)

GEN = [
    ("Phi-3.5", "phi3.5", "#E0A11B"),
    ("Mistral 7B", "mistral_7b", "#5BA3E0"),
    ("Gemma 2 9B", "gemma2_9b", "#7BB3D1"),
    ("Llama 3.1 8B", "llama3.1_8b", "#9AA8B1"),
    ("Qwen 2.5 7B", "qwen2.5_7b", "#C0C0C0"),
]
INK = "#111111"
GREY = "#555555"
METRIC_COLORS = ["#1F6FB2", "#E18727", "#6B7B8C"]  # distinct from model colors

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def summary(tag, run):
    p = SRC / ("experiment5_%s_summary_run%s.json" % (tag, run))
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


rows = []
for disp, tag, col in GEN:
    r1 = summary(tag, "1")
    r2 = summary(tag, "2")
    faith = (r1["faithfulness"]["mean"] + r2["faithfulness"]["mean"]) / 2
    relv = (r1["answer_relevance"]["mean"] + r2["answer_relevance"]["mean"]) / 2
    supp = (r1["evidence_support"]["mean"] + r2["evidence_support"]["mean"]) / 2
    flags = r1["hallucination_flags"] + r2["hallucination_flags"]
    rows.append(dict(disp=disp, col=col, faith=faith, relv=relv, supp=supp,
                     flags=flags))


def render():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                    gridspec_kw={"width_ratios": [1.6, 1.0]})

    # ---- Panel A: grouped bars (3 metrics x 5 generators) ----
    n_gen = len(rows)
    metrics = ["faith", "relv", "supp"]
    metric_labels = ["Faithfulness", "Answer\nrelevance", "Evidence\nsupport"]
    x = np.arange(n_gen)
    w = 0.24
    offsets = [-w, 0.0, w]
    for j, (mk, lab, mc) in enumerate(zip(metrics, metric_labels, METRIC_COLORS)):
        vals = [r[mk] for r in rows]
        bars = axA.bar(x + offsets[j], vals, width=w, color=mc, label=lab,
                       edgecolor=INK, linewidth=0.5, zorder=3)
        for xi, v in zip(x + offsets[j], vals):
            axA.text(xi, v + 0.015, "%.2f" % v, ha="center", va="bottom",
                     fontsize=6.2, color=INK)
    axA.set_xticks(x)
    axA.set_xticklabels([r["disp"] for r in rows], fontsize=8)
    axA.set_ylim(0.0, 1.05)
    axA.set_ylabel("mean score (two runs)")
    axA.legend(frameon=False, fontsize=7.5, loc="upper right")
    axA.tick_params(axis="y", labelsize=8)
    axA.grid(axis="y", color="#E5E5E5", linewidth=0.5, zorder=0)
    axA.set_title("(a) answer quality", fontsize=9, loc="left", pad=8)

    # ---- Panel B: flag counts (sum of two runs) ----
    flag_vals = [r["flags"] for r in rows]
    axB.bar(np.arange(n_gen), flag_vals, width=0.55,
            color=[r["col"] for r in rows], edgecolor=INK, linewidth=0.5,
            zorder=3)
    for xi, v in zip(np.arange(n_gen), flag_vals):
        axB.text(xi, v + 0.6, "%d" % v, ha="center", va="bottom",
                 fontsize=7.5, color=INK)
    axB.set_xticks(np.arange(n_gen))
    axB.set_xticklabels([r["disp"] for r in rows], fontsize=8)
    axB.set_ylim(0, 100)
    axB.set_ylabel("flag count (sum of two runs)")
    axB.tick_params(axis="y", labelsize=8)
    axB.grid(axis="y", color="#E5E5E5", linewidth=0.5, zorder=0)
    axB.set_title("(b) faithfulness \u2264 0.50 flags", fontsize=9, loc="left",
                  pad=8)

    fig.text(0.0, -0.04,
             "Flags are evaluator-defined faithfulness \u2264 0.50 by the "
             "Llama-3.2 judge (not independently verified hallucinations). "
             "Phi-3.5: one failed generation per run (GLG11-001); its quality "
             "means use n=120, flag denominator n=121. No significance implied.",
             fontsize=6.8, color=GREY, va="top")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)
    out = FIGDIR / "fig7_generator_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig7_generator_comparison.png")
    for r in rows:
        print("  %-13s faith=%.4f relv=%.4f supp=%.4f flags=%d"
              % (r["disp"], r["faith"], r["relv"], r["supp"], r["flags"]))


if __name__ == "__main__":
    render()
