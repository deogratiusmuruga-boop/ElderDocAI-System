#!/usr/bin/env python3
"""Build experiment5_summary_table.csv (one row per generator x run).

Reads the frozen Experiment 5 per-question + summary + manifest files for the
five generators plus the newly-run llama3.2:latest reference condition, and
emits the requested columns. Read-only with respect to the frozen artifacts.
"""
import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent

GEN = [
    ("llama3.2", "llama3.2_latest"),
    ("phi3.5", "phi3.5"),
    ("mistral:7b", "mistral_7b"),
    ("gemma2:9b", "gemma2_9b"),
    ("llama3.1:8b", "llama3.1_8b"),
    ("qwen2.5:7b", "qwen2.5_7b"),
]


def load(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def word_count(text):
    return len((text or "").split())


rows = []
for disp, tag in GEN:
    for run in ("1", "2"):
        pq = load("experiment5_%s_per_question_run%s.json" % (tag, run))
        summ = load("experiment5_%s_summary_run%s.json" % (tag, run))
        man = load("experiment5_%s_run_manifest_run%s.json" % (tag, run))

        faith_vals = [r["faithfulness"] for r in pq
                      if r.get("faithfulness") is not None]
        relv_vals = [r["answer_relevance"] for r in pq
                     if r.get("answer_relevance") is not None]
        supp_vals = [r["evidence_support"] for r in pq
                     if r.get("evidence_support") is not None]

        n = len(pq)
        flags = [r for r in pq if r.get("hallucination")]
        zero_faith = [r for r in pq if r.get("faithfulness") == 0]
        refusals = [r for r in pq
                    if "find that information" in (r.get("answer") or "").lower()]
        ans_lens = [word_count(r.get("answer")) for r in pq if r.get("generated")]
        pec = [r["prompt_eval_count_max"] for r in pq
               if r.get("prompt_eval_count_max") is not None]

        # flag rate split by gold-chunk retrieval
        with_gold = [r for r in pq if r.get("gold_chunk_retrieved")]
        without_gold = [r for r in pq if not r.get("gold_chunk_retrieved")]
        flag_with = sum(1 for r in with_gold if r.get("hallucination"))
        flag_without = sum(1 for r in without_gold if r.get("hallucination"))

        rows.append({
            "generator": disp,
            "run": run,
            "n": n,
            "faithfulness_mean": round(statistics.mean(faith_vals), 4)
            if faith_vals else None,
            "relevance_mean": round(statistics.mean(relv_vals), 4)
            if relv_vals else None,
            "evidence_support_mean": round(statistics.mean(supp_vals), 4)
            if supp_vals else None,
            "flagged_count": len(flags),
            "flagged_rate": round(len(flags) / n, 4),
            "zero_faithfulness_count": len(zero_faith),
            "refusal_count": len(refusals),
            "mean_answer_words": round(statistics.mean(ans_lens), 2)
            if ans_lens else None,
            "prompt_eval_count_max": max(pec) if pec else None,
            "runtime_seconds": man.get("runtime_seconds"),
            "errors": summ.get("errors", 0),
            "flagged_with_gold": flag_with,
            "flagged_with_gold_rate": round(flag_with / len(with_gold), 4)
            if with_gold else None,
            "flagged_without_gold": flag_without,
            "flagged_without_gold_rate": round(flag_without / len(without_gold), 4)
            if without_gold else None,
        })

cols = [
    "generator", "run", "n",
    "faithfulness_mean", "relevance_mean", "evidence_support_mean",
    "flagged_count", "flagged_rate",
    "zero_faithfulness_count", "refusal_count", "mean_answer_words",
    "prompt_eval_count_max", "runtime_seconds", "errors",
    "flagged_with_gold", "flagged_with_gold_rate",
    "flagged_without_gold", "flagged_without_gold_rate",
]

out = HERE / "experiment5_summary_table.csv"
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

print("wrote", out.name)
for r in rows:
    print("  %-12s run%s n=%3d faith=%.4f relv=%.4f supp=%.4f "
          "flags=%d/%.1f%% zero=%d refuse=%d words=%.1f pec_max=%s err=%d "
          "flag_with=%.1f%% flag_without=%.1f%%"
          % (r["generator"], r["run"], r["n"], r["faithfulness_mean"],
             r["relevance_mean"], r["evidence_support_mean"],
             r["flagged_count"], 100 * r["flagged_rate"],
             r["zero_faithfulness_count"], r["refusal_count"],
             r["mean_answer_words"], r["prompt_eval_count_max"], r["errors"],
             100 * r["flagged_with_gold_rate"],
             100 * r["flagged_without_gold_rate"]))
