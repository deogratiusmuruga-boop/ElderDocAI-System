#!/usr/bin/env python3
"""B4/B5 - v1.1 failure-mode diagnostics and failure-example appendix.

Analyses the FROZEN Experiment 1 per-question data to characterise the
failure modes of the v1.1 final stack:
  B4a - failure taxonomy (contradiction vs partial; topic concentration;
        gate decision of failures; gold-chunk presence)
  B4b - effect of the reliability gate on generation quality (quality by
        decision; correlation of final reliability with faithfulness)
  B5  - a deterministic, non-cherry-picked appendix of failing examples

Reads only frozen artifacts; writes only under
paper_assembly/b4_v11_failure_diagnostics/.
"""
import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent
META = OUT.parent.parent

RUN1 = (META / "experiment1_generation_evaluation"
        / "experiment1_per_question_run1.json")


def spearman_r(xs, ys):
    """Spearman rank correlation (no scipy dependency)."""
    n = len(xs)
    if n < 3:
        return None
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx = my = (n - 1) / 2
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx)
           * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def analyse(rows):
    out = {"n": len(rows)}

    # ---------- B4a: failure taxonomy ----------
    unsup = [r for r in rows if r["hallucination"]]
    contrad = [r for r in rows if r["faithfulness"] == 0.0]
    partial = [r for r in rows if 0.0 < r["faithfulness"] <= 0.5]
    taxonomy = {
        "n_unsupported": len(unsup),
        "n_contradiction": len(contrad),
        "n_partial_low": len(partial),
        "by_topic": [],
        "by_decision": {},
        "gold_chunk_present_in_failures": sum(
            1 for r in unsup if r["gold_chunk_retrieved"]),
        "gold_chunk_absent_in_failures": sum(
            1 for r in unsup if not r["gold_chunk_retrieved"]),
        "mean_reliability_of_failures": round(
            statistics.mean(r["reliability"]["overall_reliability"]
                            for r in unsup), 4),
    }
    for t in sorted({r["topic"] for r in rows}):
        tr = [r for r in rows if r["topic"] == t]
        tf = [r for r in tr if r["hallucination"]]
        taxonomy["by_topic"].append({
            "topic": t, "n": len(tr), "failures": len(tf),
            "failure_rate_pct": round(100.0 * len(tf) / len(tr), 1)})
    for dec in ["ACCEPT", "REFINE", "RE-RETRIEVE", "REJECT"]:
        dr = [r for r in rows if r["decision"] == dec]
        df = [r for r in dr if r["hallucination"]]
        taxonomy["by_decision"][dec] = {
            "n": len(dr), "failures": len(df),
            "failure_rate_pct": round(
                100.0 * len(df) / len(dr), 1) if dr else None}
    out["failure_taxonomy"] = taxonomy

    # ---------- B4b: gate effect on generation quality ----------
    gate = {"by_decision": {}}
    for dec in ["ACCEPT", "REFINE", "RE-RETRIEVE", "REJECT"]:
        dr = [r for r in rows if r["decision"] == dec]
        if not dr:
            gate["by_decision"][dec] = None
            continue
        gate["by_decision"][dec] = {
            "n": len(dr),
            "mean_faithfulness": round(statistics.mean(
                r["faithfulness"] for r in dr), 4),
            "mean_answer_relevance": round(statistics.mean(
                r["answer_relevance"] for r in dr), 4),
            "mean_evidence_support": round(statistics.mean(
                r["evidence_support"] for r in dr), 4),
            "hallucination_rate_pct": round(100.0 * sum(
                1 for r in dr if r["hallucination"]) / len(dr), 1),
            "mean_initial_reliability": round(statistics.mean(
                r["initial_reliability"]["overall_reliability"]
                for r in dr), 4),
            "mean_final_reliability": round(statistics.mean(
                r["reliability"]["overall_reliability"] for r in dr), 4),
        }
    rho = spearman_r(
        [r["reliability"]["overall_reliability"] for r in rows],
        [r["faithfulness"] for r in rows])
    gate["reliability_faithfulness_spearman"] = (
        round(rho, 4) if rho is not None else None)
    n_gold = sum(1 for r in rows if r["gold_chunk_retrieved"])
    gate["gold_chunk_present_n"] = n_gold
    gate["gold_present_failure_rate_pct"] = round(100.0 * sum(
        1 for r in rows if r["gold_chunk_retrieved"] and r["hallucination"])
        / n_gold, 1)
    gate["gold_absent_failure_rate_pct"] = round(100.0 * sum(
        1 for r in rows if not r["gold_chunk_retrieved"]
        and r["hallucination"]) / (len(rows) - n_gold), 1)
    out["gate_quality"] = gate

    return out, unsup
def failure_examples(unsup):
    # ---------- B5: deterministic failure examples ----------
    fails = sorted(unsup, key=lambda r: (r["faithfulness"],
                                          r["answer_relevance"],
                                          r["question_id"]))
    return [{
        "question_id": r["question_id"], "topic": r["topic"],
        "decision": r["decision"], "faithfulness": r["faithfulness"],
        "answer_relevance": r["answer_relevance"],
        "evidence_support": r["evidence_support"],
        "gold_chunk_retrieved": r["gold_chunk_retrieved"],
        "question": r["question"], "answer": r["answer"],
    } for r in fails]


def render(out):
    md = ["# B4/B5 - v1.1 failure-mode diagnostics and failure examples",
          "",
          "Population: %d questions (GeriLit-Gold v1.1, frozen)."
          % out["n"],
          ""]
    ft = out["failure_taxonomy"]
    md.append("## B4a - Failure taxonomy")
    md.append("")
    md.append("- Unsupported (faithfulness <= 0.5): %d (%.2f%%)"
              % (ft["n_unsupported"],
                 100.0 * ft["n_unsupported"] / out["n"]))
    md.append("- Contradiction (faithfulness == 0.0): %d (%.2f%%)"
              % (ft["n_contradiction"],
                 100.0 * ft["n_contradiction"] / out["n"]))
    md.append("- Partial low (0 < faithfulness <= 0.5): %d"
              % ft["n_partial_low"])
    md.append("- Mean final reliability of failing questions: %.4f"
              % ft["mean_reliability_of_failures"])
    md.append("- Gold chunk present in evidence for failures: %d/%d"
              % (ft["gold_chunk_present_in_failures"],
                 ft["n_unsupported"]))
    md.append("")
    md.append("### By gate decision")
    md.append("")
    md.append("| decision | n | failures | failure rate % |")
    md.append("|----------|---|----------|----------------|")
    for dec, v in ft["by_decision"].items():
        md.append("| %s | %d | %d | %s |"
                  % (dec, v["n"], v["failures"], v["failure_rate_pct"]))
    md.append("")
    md.append("### By topic")
    md.append("")
    md.append("| topic | n | failures | failure rate % |")
    md.append("|-------|---|----------|----------------|")
    for t in ft["by_topic"]:
        md.append("| %s | %d | %d | %.1f |"
                  % (t["topic"], t["n"], t["failures"],
                     t["failure_rate_pct"]))
    md.append("")
    g = out["gate_quality"]
    md.append("## B4b - Reliability gate and generation quality")
    md.append("")
    md.append("| decision | n | mean faith | mean relv | mean support | "
              "halluc % | mean init rel | mean final rel |")
    md.append("|----------|---|------------|-----------|--------------|"
              "---------|-----------------|----------------|")
    for dec, v in g["by_decision"].items():
        if v is None:
            continue
        md.append("| %s | %d | %.3f | %.3f | %.3f | %.1f | %.3f | %.3f |"
                  % (dec, v["n"], v["mean_faithfulness"],
                     v["mean_answer_relevance"], v["mean_evidence_support"],
                     v["hallucination_rate_pct"],
                     v["mean_initial_reliability"],
                     v["mean_final_reliability"]))
    md.append("")
    md.append("- Spearman(reliability, faithfulness): %s"
              % g["reliability_faithfulness_spearman"])
    md.append("- Failure rate with gold chunk present: %.1f%% (%d base)"
              % (g["gold_present_failure_rate_pct"],
                 g["gold_chunk_present_n"]))
    md.append("- Failure rate with gold chunk absent: %.1f%%"
              % g["gold_absent_failure_rate_pct"])
    md.append("")
    md.append("## B5 - Failure examples (all %d unsupported rows, "
              "deterministic sort)" % len(out["failure_examples"]))
    md.append("")
    md.append("| question id | topic | decision | faith | relv | support | "
              "gold |")
    md.append("|-------------|-------|----------|-------|------|---------|"
              "------|")
    for e in out["failure_examples"]:
        md.append("| %s | %s | %s | %.2f | %.2f | %.3f | %s |"
                  % (e["question_id"], e["topic"], e["decision"],
                     e["faithfulness"], e["answer_relevance"],
                     e["evidence_support"], e["gold_chunk_retrieved"]))
    md.append("")
    md.append("Illustrative excerpt (worst-faithfulness row):")
    first = out["failure_examples"][0]
    md.append("")
    md.append("**Q:** %s" % first["question"])
    md.append("")
    md.append("**A:** %s" % first["answer"])
    (OUT / "b4_v11_failure_diagnostics.md").write_text(
        "\n".join(md), encoding="utf-8")
    (OUT / "b4_v11_failure_diagnostics.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote b4_v11_failure_diagnostics.json / .md")


if __name__ == "__main__":
    with open(RUN1, "r", encoding="utf-8") as f:
        rows = json.load(f)
    out, unsup = analyse(rows)
    out["failure_examples"] = failure_examples(unsup)
    render(out)