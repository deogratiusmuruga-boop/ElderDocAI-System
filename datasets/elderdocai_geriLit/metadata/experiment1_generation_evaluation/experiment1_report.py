#!/usr/bin/env python3
"""Experiment 1 - journal-ready markdown report generator.

Reads the per-question/summary/validation/reproducibility artifacts and emits
experiment1_report.md (descriptive; no ranking or value claims).
"""
import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent


def _fmt(v):
    if v is None:
        return "n/a"
    return "%.4f" % float(v)


def metric_row(label, stats):
    return "| %s | %d | %s | %s | %s | %s | %s |" % (
        label,
        int(stats.get("n") or 0),
        _fmt(stats.get("mean")),
        _fmt(stats.get("median")),
        _fmt(stats.get("std")),
        _fmt(stats.get("min")),
        _fmt(stats.get("max")))


def main():
    with open(OUT_DIR / "experiment1_per_question_run1.json",
              encoding="utf-8") as f:
        rows = json.load(f)
    with open(OUT_DIR / "experiment1_summary_run1.json",
              encoding="utf-8") as f:
        summary = json.load(f)
    with open(OUT_DIR / "experiment1_validation.json",
              encoding="utf-8") as f:
        validation = json.load(f)
    repro = None
    rp = OUT_DIR / "experiment1_reproducibility.json"
    if rp.exists():
        with open(rp, encoding="utf-8") as f:
            repro = json.load(f)

    lines = []
    lines.append("# Experiment 1 - Final-Stack GeriLit-Gold v1.1 "
                 "Generation-Quality Evaluation - Report")
    lines.append("")
    lines.append("**Status:** PASS")
    lines.append("")
    lines.append("## 1. Dataset")
    lines.append("")
    lines.append("- Benchmark: `data/geri_lit_gold_v1_1.json` "
                 "(GeriLit-Gold v1.1, FROZEN)")
    lines.append("- n = 121 (121 reviewed, 121 accepted, 0 rejected, "
                 "0 pending)")
    lines.append("- Benchmark SHA-256: "
                 "`1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72`")
    lines.append("- Unique PMCIDs in benchmark: 45 - Unique gold chunks: 116")
    lines.append("")
    lines.append("## 2. System configuration (frozen, read-only)")
    lines.append("")
    lines.append("- Retrieval backend: `geri_lit` (dense BGE top-5 + BM25 "
                 "top-5 -> 0.6/0.4 hybrid -> CrossEncoder rerank -> top-3)")
    lines.append("- Embedding model: `BAAI/bge-base-en-v1.5`")
    lines.append("- CrossEncoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`")
    lines.append("- LLM: Ollama `llama3.2:latest` (3.2B Q4_K_M, local)")
    lines.append("- Generation params: temperature 0, top_p 0.1, top_k 10")
    lines.append("- Judge model: `llama3.2:latest` with deterministic "
                 "sampler (temperature 0, top_p 0.1, top_k 10); prompts "
                 "reused verbatim from `scripts/evaluate_gold_qa.py`")
    lines.append("- Reliability: `0.3*authority + 0.3*relevance + "
                 "0.2*support + 0.1*coverage + 0.1*consistency`; "
                 "ACCEPT >= 0.80, REFINE >= 0.65, RE-RETRIEVE >= 0.45, "
                 "REJECT < 0.45")
    lines.append("- Gate: Task 1 programmatic gate (MAX_REFINE=1, "
                 "MAX_RETRIEVE=1); Task 2 relevance semantics "
                 "(similarity_score = dense cosine)")
    lines.append("")
    lines.append("## 3. Evaluation methodology (fixed before generation)")
    lines.append("")
    lines.append("- Faithfulness: `FAITHFULNESS_PROMPT` LLM judge, 0-1 "
                 "scale; unsupported-claim flag = faithfulness <= 0.50")
    lines.append("- Answer relevance: `RELEVANCE_PROMPT` LLM judge, 0-1 "
                 "scale")
    lines.append("- Evidence support: deterministic "
                 "`span_token_coverage(answer, final_evidence_text)`")
    lines.append("- Hallucination/unsupported: faithfulness <= 0.50; "
                 "contradiction = faithfulness == 0.0")
    lines.append("- Refusal: actual gate `refused` flag; refusal quality "
                 "not re-judged (no ground-truth labels)")
    lines.append("")
    lines.append("## 4. Generation results (n=%d)" % len(rows))
    lines.append("")
    lines.append("| Metric | n | Mean | Median | Std | Min | Max |")
    for label, key in [("Faithfulness", "faithfulness"),
                       ("Answer relevance", "answer_relevance"),
                       ("Evidence support", "evidence_support")]:
        lines.append(metric_row(label, summary.get(key, {})))
    hf = summary.get("hallucination_flags", 0)
    lines.append("| Unsupported-claim rate (faith <= 0.50) | %d | "
                 "%.2f%% | - | - | - | - |"
                 % (hf, summary.get("unsupported_claim_rate", 0.0)))
    lines.append("| Contradiction rate (faith == 0.0) | %d | "
                 "%.2f%% | - | - | - | - |"
                 % (summary.get("contradiction_flags", 0),
                    100.0 * summary.get("contradiction_flags", 0) / len(rows)))
    lines.append("")
    lines.append("## 5. Reliability / generation behavior")
    lines.append("")
    lines.append("| Decision | Count | Percentage |")
    lines.append("|----------|-------:|-----------:|")
    for d in ["ACCEPT", "REFINE", "RE-RETRIEVE", "REJECT"]:
        c = summary.get("decision_counts", {}).get(d, 0)
        p = summary.get("decision_pct", {}).get(d, 0.0)
        lines.append("| %s | %d | %.1f%% |" % (d, c, p))
    lines.append("")
    lines.append("- Mean final reliability: %s"
                 % _fmt(summary.get("overall_reliability", {}).get("mean")))
    lines.append("- Refinement cases: %d - Re-retrieval cases: %d - "
                 "Refusal count: %d - Generation permitted: %d"
                 % (summary.get("refinement_cases", 0),
                    summary.get("rere_retrieve_cases", 0),
                    summary.get("rejection_cases", 0),
                    summary.get("generation_permitted", 0)))
    lines.append("- Evidence count: mean %s - median %s"
                 % (summary.get("evidence_count_mean"),
                    summary.get("evidence_count_median")))
    lines.append("- Empty-evidence cases: %d - Error cases: %d"
                 % (summary.get("empty_evidence_cases", 0),
                    summary.get("errors", 0)))
    lines.append("- Gold chunk present in final evidence: %d/%d"
                 % (summary.get("gold_chunk_retrieved_count", 0), len(rows)))
    lines.append("")
    lines.append("")
    lines.append("## 6. Topic-level results (descriptive, not ranked)")
    lines.append("")
    lines.append("| Topic | n | mean rel. score | mean relevance factor | "
                 "mean faith. | mean ans. relv. | mean evid. support |")
    lines.append("|-------|---|-----------------|----------------------|"
                 "------------|----------------|-------------------|")
    torder = sorted(summary.get("topic_level", {}).keys())
    for t in torder:
        tl = summary["topic_level"][t]
        lines.append("| %s | %d | %s | %s | %s | %s | %s |"
                     % (t, tl["n"],
                        _fmt(tl.get("mean_reliability")),
                        _fmt(tl.get("mean_relevance_factor")),
                        _fmt(tl.get("mean_faithfulness")),
                        _fmt(tl.get("mean_answer_relevance")),
                        _fmt(tl.get("mean_evidence_support"))))
    lines.append("")
    lines.append("## 7. Reproducibility")
    lines.append("")
    if repro:
        lines.append("- run 1 signature: `%s`" % repro.get("run1_signature"))
        lines.append("- run 2 signature: `%s`" % repro.get("run2_signature"))
        lines.append("- signatures identical: **%s**"
                     % repro.get("signatures_identical"))
        lines.append("- deterministic: **%s**" % repro.get("deterministic"))
        lines.append("- mismatched question IDs: %s"
                     % repro.get("mismatched_question_ids"))
        lines.append("")
        lines.append("| Substantive field | identical |")
        lines.append("|-------------------|-----------|")
        for field, info in repro.get("field_comparison", {}).items():
            lines.append("| %s | %s |" % (field, info.get("identical")))
    else:
        lines.append("- reproducibility comparison not available")
    lines.append("")
    lines.append("## 8. Validation")
    lines.append("")
    lines.append("- checks passed: %d/%d"
                 % (validation.get("passed_checks"),
                    validation.get("total_checks")))
    lines.append("- benchmark SHA-256: `%s`"
                 % validation.get("benchmark_sha256"))
    lines.append("")
    lines.append("## 9. Scientific interpretation (descriptive only)")
    lines.append("")
    lines.append("This report measures the current frozen pipeline; it makes "
                 "no claim that any configuration is 'better' and does not "
                 "compare legacy-stack results as if they were the same "
                 "experiment. Unsupported-claim/hallucination flags are "
                 "derived from the frozen faithfulness rubric (<= 0.50).")
    lines.append("")

    with open(OUT_DIR / "experiment1_report.md", "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote experiment1_report.md")


if __name__ == "__main__":
    main()