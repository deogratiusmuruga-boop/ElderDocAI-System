#!/usr/bin/env python3
"""A2 - LLM judge reliability quantification across the two frozen runs.

Computes per-metric run-to-run agreement (exact match) and Cohen's kappa for
the LLM judge scores in Experiments 1-4, using the two frozen per-question
run files. Distinguishes judge-instrument variability from system
variability.

Reads only frozen artifacts; writes only under paper_assembly/a2_judge_reliability/.
"""
import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent
META = Path(__file__).resolve().parent.parent.parent  # geriLit metadata dir

EXPERIMENTS = [
    {
        "name": "exp1",
        "label": "Exp 1 - generation quality",
        "run1": META / "experiment1_generation_evaluation"
                / "experiment1_per_question_run1.json",
        "run2": META / "experiment1_generation_evaluation"
                / "experiment1_per_question_run2.json",
        "metrics": [
            ("faithfulness", lambda r: r.get("faithfulness")),
            ("answer_relevance", lambda r: r.get("answer_relevance")),
            ("evidence_support", lambda r: r.get("evidence_support")),
        ],
    },
    {
        "name": "exp2",
        "label": "Exp 2 - gating effectiveness",
        "run1": META / "experiment2_gating_effectiveness"
                / "experiment2_per_question_run1.json",
        "run2": META / "experiment2_gating_effectiveness"
                / "experiment2_per_question_run2.json",
        "metrics": [
            ("gate_on_faithfulness",
             lambda r: r["gate_on"]["faithfulness"]),
            ("gate_on_answer_relevance",
             lambda r: r["gate_on"]["answer_relevance"]),
            ("gate_on_evidence_support",
             lambda r: r["gate_on"]["evidence_support"]),
            ("gate_off_faithfulness",
             lambda r: r["gate_off"]["faithfulness"]),
            ("gate_off_answer_relevance",
             lambda r: r["gate_off"]["answer_relevance"]),
            ("gate_off_evidence_support",
             lambda r: r["gate_off"]["evidence_support"]),
        ],
    },
    {
        "name": "exp3",
        "label": "Exp 3 - care-state adaptation",
        "run1": META / "experiment3_care_state_adaptation"
                / "experiment3_per_question_run1.json",
        "run2": META / "experiment3_care_state_adaptation"
                / "experiment3_per_question_run2.json",
        "metrics": [
            ("A_faithfulness", lambda r: r["condition_A"]["faithfulness"]),
            ("A_answer_relevance",
             lambda r: r["condition_A"]["answer_relevance"]),
            ("A_evidence_support",
             lambda r: r["condition_A"]["evidence_support"]),
            ("B_faithfulness", lambda r: r["condition_B"]["faithfulness"]),
            ("B_answer_relevance",
             lambda r: r["condition_B"]["answer_relevance"]),
            ("B_evidence_support",
             lambda r: r["condition_B"]["evidence_support"]),
            ("appropriateness", lambda r: r["appropriateness_judgment"]),
        ],
    },
    {
        "name": "exp4",
        "label": "Exp 4 - vanilla RAG baseline",
        "run1": META / "experiment4_vanilla_baseline"
                / "experiment4_per_question_run1.json",
        "run2": META / "experiment4_vanilla_baseline"
                / "experiment4_per_question_run2.json",
        "metrics": [
            ("full_faithfulness", lambda r: r["full"]["faithfulness"]),
            ("full_answer_relevance",
             lambda r: r["full"]["answer_relevance"]),
            ("full_evidence_support",
             lambda r: r["full"]["evidence_support"]),
            ("vanilla_faithfulness",
             lambda r: r["vanilla"]["faithfulness"]),
            ("vanilla_answer_relevance",
             lambda r: r["vanilla"]["answer_relevance"]),
            ("vanilla_evidence_support",
             lambda r: r["vanilla"]["evidence_support"]),
        ],
    },
]
def cohens_kappa(a, b):
    """Cohen's kappa for two judge score arrays (0.25-grid scale)."""
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
    report = {"experiments": []}
    for exp in EXPERIMENTS:
        if not exp["run1"].exists() or not exp["run2"].exists():
            report["experiments"].append({
                "name": exp["name"], "label": exp["label"],
                "error": "run files missing"})
            continue
        with open(exp["run1"], "r", encoding="utf-8") as f:
            r1 = json.load(f)
        with open(exp["run2"], "r", encoding="utf-8") as f:
            r2 = json.load(f)
        m1 = {}
        for r in r1:
            m1[r.get("question_id", r.get("pair_id"))] = r
        m2 = {}
        for r in r2:
            m2[r.get("question_id", r.get("pair_id"))] = r

        metric_rows = []
        for label, fn in exp["metrics"]:
            pairs = []
            for q in m1:
                if q not in m2:
                    continue
                x = fn(m1[q])
                y = fn(m2[q])
                if x is not None and y is not None:
                    pairs.append((x, y))
            if not pairs:
                metric_rows.append({"metric": label, "n": 0})
                continue
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            match = sum(1 for x, y in pairs if x == y)
            kappa = cohens_kappa(xs, ys)
            metric_rows.append({
                "metric": label,
                "n": len(pairs),
                "exact_match": round(100.0 * match / len(pairs), 2),
                "cohens_kappa": round(kappa, 4) if kappa is not None
                                else None,
                "median_abs_diff": round(
                    statistics.median(abs(x - y) for x, y in pairs), 4),
                "max_abs_diff": round(
                    max(abs(x - y) for x, y in pairs), 4),
            })
        report["experiments"].append({
            "name": exp["name"], "label": exp["label"],
            "n_questions_run1": len(m1),
            "n_questions_run2": len(m2),
            "metrics": metric_rows,
        })

    with open(OUT / "a2_judge_reliability.json", "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    lines = ["# A2 - LLM Judge Reliability (run1 vs run2)", "",
             "Per-metric exact-match agreement and Cohen's kappa across the "
             "two frozen runs per experiment. Judge = llama3.2:latest "
             "(temperature 0, top_p 0.1, top_k 10).", ""]
    for exp in report["experiments"]:
        lines.append("## %s" % exp["label"])
        lines.append("")
        lines.append("| metric | n | exact match % | Cohen's kappa | "
                     "median |A-B| | max |A-B| |")
        lines.append("|--------|---|---------------|---------------|"
                     "------------|-----------|")
        for m in exp.get("metrics", []):
            lines.append("| %s | %d | %.2f | %s | %.4f | %.4f |"
                         % (m["metric"], m.get("n", 0),
                            m.get("exact_match", 0.0),
                            (str(m["cohens_kappa"]) if m.get("cohens_kappa")
                             is not None else "n/a"),
                            m.get("median_abs_diff", 0.0),
                            m.get("max_abs_diff", 0.0)))
        lines.append("")
    (OUT / "a2_judge_reliability.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print("wrote a2_judge_reliability.json / .md")


if __name__ == "__main__":
    main()