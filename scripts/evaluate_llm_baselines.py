"""
ElderDocAI - External LLM Generation Baselines.

Controlled experiment that keeps EVERYTHING identical to the existing
RQ2/ablation study EXCEPT the generation model:
  * Same 16 gold QA questions.
  * Same hybrid retrieval (dense + BM25 + CrossEncoder reranking, top 3).
  * Same patient profile / adaptive assistance plan.
  * Same reliability config + decision.
  * Same grounded-prompt templates and SAME judge model + rubric.
  * Same sampling options (temperature 0 / top_p 0.1 / top_k 10).

Two prompt contexts are compared for every generator model:
  FULL  : complete ElderDocAI grounded prompt (dynamic care state +
          adaptive assistance plan + reliability gate block), byte-identical
          to ablation condition A0.
  PLAIN : static-patient profile + plain RAG grounded prompt (no dynamic
          state, no adaptive plan), byte-identical to ablation condition A1.

Reference generator (llama3.2:latest) is NOT re-run: its FULL and PLAIN
records are REUSED from the completed ablation run (conditions A0 / A1)
after a deterministic SHA-256 prompt-equivalence check, so the ONLY
experimental difference between any two conditions is the generation model.

Analysis-only: reads production modules and ablation outputs read-only,
writes ONLY under data/evaluation_results/llm_baselines/. Does NOT modify
production code, datasets, configuration, thresholds, or existing RQ result
files. The longitudinal records are SYNTHETIC (Synthea-derived): this is an
architectural / internal-consistency comparison and implies no clinical
benefit.
"""

import os
import re
import json
import time
import statistics
import sys
import argparse
from collections import Counter

import ollama

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.hybrid_retriever import hybrid_search
from scripts.rag_chat import prepare_evidence
from scripts.reliability_evaluation import evaluate_reliability
from scripts.adaptive_decision_controller import make_reliability_decision
from scripts.build_grounded_prompt import build_grounded_prompt
from scripts.evaluate_gold_qa import (
    FAITHFULNESS_PROMPT,
    RELEVANCE_PROMPT,
    judge,
    normalize_for_span,
    span_token_coverage,
)
from scripts.evaluate_ablation import (
    build_patient_context,
    compute_grounding,
    sha256_text,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_QA_FILE = os.path.join(BASE_DIR, "data", "gold_qa_evaluation.json")
DEFAULT_ABLATION_JSON = os.path.join(
    BASE_DIR, "data", "evaluation_results", "ablation", "ablation_results.json"
)
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "data", "evaluation_results", "llm_baselines")
ABLATION_JSON = DEFAULT_ABLATION_JSON
OUT_DIR = DEFAULT_OUT_DIR
FIG_DIR = os.path.join(OUT_DIR, "figures")

RESULTS_JSON = os.path.join(OUT_DIR, "llm_baselines_results.json")
REPORT_MD = os.path.join(OUT_DIR, "llm_baselines_report.md")
CONSOLE_LOG = os.path.join(OUT_DIR, "llm_baselines_console_output.txt")

JUDGE_MODEL = "llama3.2:latest"
REFERENCE_MODEL = "llama3.2:latest"

# Generation models under test (distinct local families).
BASELINE_MODELS = [
    "mistral:latest",
    "qwen3:latest",
    "gemma2:9b",
    "phi3:mini",
]

# FULL == ablation A0 (dynamic care state + adaptive plan + gate).
# PLAIN == ablation A1 (static profile + plain RAG + gate).
CONTEXTS = ["FULL", "PLAIN"]

# Sampling parity with RQ2 / RQ5 / ablation.
GEN_OPTIONS = {"temperature": 0, "top_p": 0.1, "top_k": 10}


def cround(value, ndigits=4):
    if value is None:
        return None
    return round(float(value), ndigits)


# ---------------------------------------------------------------------------
# Generation (per-model). Mirrors evaluate_ablation._run_generation exactly,
# except the model name is a parameter.
# ---------------------------------------------------------------------------
def _run_generation(prompt, model, retry=2):
    """Call the given generation model with the exact RQ2 sampling settings."""
    last_error = None
    for _ in range(retry + 1):
        try:
            response = ollama.chat(
                model=model,
                options=dict(GEN_OPTIONS),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are CareBuddy, an evidence-grounded "
                            "elderly-care assistant. Answer ONLY using the "
                            "provided context and retrieved knowledge. "
                            "Do not guess. Do not diagnose."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            answer = response["message"]["content"].strip()
            answer = strip_think_blocks(answer)
            if "\nSources:" in answer:
                answer = answer.split("\nSources:", 1)[0].strip()
            elif answer.startswith("Sources:"):
                answer = answer[len("Sources:"):].strip()
            if answer.startswith("Answer:"):
                answer = answer[len("Answer:"):].strip()
            return answer, None
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2)
    return None, last_error


def make_condition(model, context):
    """Condition spec: same retrieval/rerank/gate as the ablation study."""
    if context == "FULL":
        return {
            "model": model,
            "context": context,
            "retrieval": "hybrid_full",
            "reranking": True,
            "reliability_gate": True,
            "adaptive_assistance": "adaptive",
            "patient_attached": True,
            "reference_condition": "A0",
        }
    if context == "PLAIN":
        return {
            "model": model,
            "context": context,
            "retrieval": "hybrid_full",
            "reranking": True,
            "reliability_gate": True,
            "adaptive_assistance": "none",
            "patient_attached": False,
            "reference_condition": "A1",
        }
    raise ValueError(f"unknown context: {context}")
def evaluate_baseline_question(cond, q_item, patient_ctx, run_llm=True,
                               expected_prompt_sha256=None):
    """
    Evaluate one gold question under one (model, context) condition.

    Mirrors evaluate_ablation.evaluate_condition_question with the generation
    model parametrized. When run_llm=False the deterministic prompt-side
    record is returned (no generation, no judge calls).
    """
    record = {
        "condition": None,
        "model": cond["model"],
        "context": cond["context"],
        "id": q_item.get("id"),
        "question": q_item.get("question"),
        "topic": q_item.get("topic"),
        "category": q_item.get("category"),
        "expected_source_document": q_item.get("source_document"),
        "expected_chunk_ids": q_item.get("chunk_ids", []),
        "supporting_span": q_item.get("supporting_span"),
        "retrieved_sources": [],
        "retrieved_chunk_ids": [],
        "retrieved_evidence_texts": [],
        "generated_answer": None,
        "reliability": None,
        "adaptive_decision": None,
        "grounding": None,
        "answer_relevance": None,
        "faithfulness": None,
        "prompt_sha256": None,
        "prompt_reference_matches": None,
        "assistance_plan_strategy": None,
        "timings": {},
        "error": None,
    }

    query = q_item.get("question", "")
    t0 = time.perf_counter()
    try:
        t_r = time.perf_counter()
        chunks = hybrid_search(query)
        record["timings"]["retrieval_seconds"] = round(time.perf_counter() - t_r, 3)

        if not chunks:
            record["generated_answer"] = (
                "I couldn't find that information in the knowledge base."
            )
            record["timings"]["total_seconds"] = round(time.perf_counter() - t0, 3)
            record["grounding"] = compute_grounding(q_item, [])
            return record

        evidence_items = prepare_evidence(chunks)
        record["retrieved_sources"] = [
            e.get("source_document") for e in evidence_items
        ]
        record["retrieved_chunk_ids"] = [
            e.get("chunk_id") for e in evidence_items
        ]
        record["retrieved_evidence_texts"] = [
            e.get("text", "") for e in evidence_items
        ]
        record["grounding"] = compute_grounding(q_item, evidence_items)

        if cond["patient_attached"]:
            user_profile = patient_ctx["patient_profile"]
        else:
            user_profile = patient_ctx["static_profile"]

        if cond["adaptive_assistance"] == "adaptive":
            assistance_plan = patient_ctx["assistance_plan"]
        else:
            assistance_plan = None
        record["assistance_plan_strategy"] = (
            assistance_plan.get("assistance_strategy")
            if assistance_plan else None
        )

        reliability = evaluate_reliability(query=query, evidence_items=evidence_items)
        decision = make_reliability_decision(reliability)
        record["reliability"] = {
            k: cround(float(v)) for k, v in reliability.items()
        }
        record["adaptive_decision"] = decision.get("decision")

        prompt = build_grounded_prompt(
            query=query,
            evidence_items=evidence_items,
            reliability=reliability,
            decision=decision,
            user_profile=user_profile,
            conversation_context="",
            response_language="en",
            assistance_plan=assistance_plan,
        )
        record["prompt_sha256"] = sha256_text(prompt)
        if expected_prompt_sha256 is not None:
            record["prompt_reference_matches"] = (
                record["prompt_sha256"] == expected_prompt_sha256
            )
        else:
            record["prompt_reference_matches"] = None

        if not run_llm:
            record["prompt_built_without_llm"] = True
            return record

        t_g = time.perf_counter()
        answer, gen_error = _run_generation(prompt, cond["model"])
        record["generated_answer"] = answer
        record["timings"]["generation_seconds"] = round(
            time.perf_counter() - t_g, 3
        )
        if gen_error and answer is None:
            record["error"] = f"ollama generation failed: {gen_error}"

        t_j = time.perf_counter()
        if answer is not None:
            relevance_score, relevance_reason = judge(
                query, answer, evidence_items, RELEVANCE_PROMPT,
                "You evaluate RAG answer relevance only.",
            )
            faithfulness_score, faithfulness_reason = judge(
                query, answer, evidence_items, FAITHFULNESS_PROMPT,
                "You evaluate RAG faithfulness only.",
            )
            record["answer_relevance"] = {
                "score": relevance_score, "reason": relevance_reason
            }
            record["faithfulness"] = {
                "score": faithfulness_score, "reason": faithfulness_reason
            }
        record["timings"]["judge_seconds"] = round(time.perf_counter() - t_j, 3)
        record["timings"]["total_seconds"] = round(time.perf_counter() - t0, 3)
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["timings"]["total_seconds"] = round(time.perf_counter() - t0, 3)

    return record


def summarize_seconds(records, key):
    vals = [
        r.get("timings", {}).get(key)
        for r in records
        if r.get("timings", {}).get(key) is not None
    ]
    if not vals:
        return None, None
    return cround(statistics.mean(vals), 4), cround(statistics.median(vals), 4)


def score_list(records, key):
    out = []
    for r in records:
        val = r.get(key)
        if isinstance(val, dict):
            val = val.get("score")
        if val is not None:
            try:
                out.append(float(val))
            except (TypeError, ValueError):
                continue
    return out


def summarize_condition(cond, records, elapsed, reference_used=False):
    n = len(records)
    n_generated = sum(1 for r in records if r.get("generated_answer"))
    n_errors = sum(1 for r in records if r.get("error"))

    src_ok = sum(
        1 for r in records
        if (r.get("grounding") or {}).get("source_retrieval_correct")
    )
    chunk_ok = sum(
        1 for r in records
        if (r.get("grounding") or {}).get("chunk_recall_any_expected")
    )
    span_ok = sum(
        1 for r in records
        if (r.get("grounding") or {}).get("supporting_span_supported")
    )

    recalls = [
        (r.get("grounding") or {}).get("recall_at_3")
        for r in records
        if (r.get("grounding") or {}).get("recall_at_3") is not None
    ]
    mrrs = [
        (r.get("grounding") or {}).get("mrr_source")
        for r in records
        if (r.get("grounding") or {}).get("mrr_source") is not None
    ]

    faith = score_list(records, "faithfulness")
    relv = score_list(records, "answer_relevance")
    rels = [
        float(r["reliability"]["overall_reliability"])
        for r in records
        if r.get("reliability")
        and r["reliability"].get("overall_reliability") is not None
    ]
    decisions = Counter(r.get("adaptive_decision") for r in records)

    gen_stat = summarize_seconds(records, "generation_seconds")
    judge_stat = summarize_seconds(records, "judge_seconds")

    return {
        "condition_name": condition_name(cond),
        "model": cond["model"],
        "context": cond["context"],
        "reference_condition": cond.get("reference_condition"),
        "reference_used": bool(reference_used),
        "n_questions": n,
        "n_generated_answers": n_generated,
        "successful_generation_rate_pct": pct(n_generated, n),
        "n_errors": n_errors,
        "runtime_seconds": elapsed,
        "generation_seconds_mean": gen_stat[0],
        "generation_seconds_median": gen_stat[1],
        "judge_seconds_mean": judge_stat[0],
        "judge_seconds_median": judge_stat[1],
        "retrieval": {
            "source_accuracy_pct": pct(src_ok, n),
            "chunk_recall_any_expected_pct": pct(chunk_ok, n),
            "supporting_span_supported_pct": pct(span_ok, n),
            "recall_at_3_mean": cround(statistics.mean(recalls), 4) if recalls else None,
            "mrr_source_mean": cround(statistics.mean(mrrs), 4) if mrrs else None,
            "recall_at_3_per_q": recalls,
            "mrr_source_per_q": mrrs,
        },
        "answer": {
            "faithfulness_mean": cround(statistics.mean(faith), 4) if faith else None,
            "faithfulness_median": cround(statistics.median(faith), 4) if faith else None,
            "faithfulness_stdev": cround(statistics.stdev(faith), 4) if len(faith) > 1 else None,
            "faithfulness_per_q": [cround(v, 4) for v in faith],
            "answer_relevance_mean": cround(statistics.mean(relv), 4) if relv else None,
            "answer_relevance_median": cround(statistics.median(relv), 4) if relv else None,
            "answer_relevance_stdev": cround(statistics.stdev(relv), 4) if len(relv) > 1 else None,
            "answer_relevance_per_q": [cround(v, 4) for v in relv],
            "judge_parse_failures": (n_generated - len(faith)) + (n_generated - len(relv)),
        },
        "reliability": {
            "mean": cround(statistics.mean(rels), 4) if rels else None,
            "min": cround(min(rels), 4) if rels else None,
            "max": cround(max(rels), 4) if rels else None,
            "median": cround(statistics.median(rels), 4) if rels else None,
            "decision_distribution": jsonable(decisions),
        },
    }


def run_condition(cond, gold_questions, patient_ctx, cond_log,
                  expected_hashes_by_qid=None):
    """Run 16 gold questions under one (model, context) condition."""
    records = []
    t0 = time.perf_counter()
    for i, q in enumerate(gold_questions, start=1):
        line = (
            f"[{i}/{len(gold_questions)}] "
            f"{condition_name(cond)} :: {q.get('id')} :: {q.get('question')}"
        )
        print(line, flush=True)
        cond_log.append(line)
        expected = None
        if expected_hashes_by_qid is not None:
            expected = expected_hashes_by_qid.get(q.get("id"))
        rec = evaluate_baseline_question(
            cond, q, patient_ctx, run_llm=True,
            expected_prompt_sha256=expected,
        )
        rec["condition"] = condition_name(cond)
        records.append(rec)
        if rec.get("error"):
            print(f"  ERROR: {rec['error']}", flush=True)
            cond_log.append(f"  ERROR: {rec['error']}")
        else:
            g = rec.get("grounding") or {}
            detail = (
                f"  src={g.get('source_retrieval_correct')} "
                f"prompt_match={rec.get('prompt_reference_matches')} "
                f"rel={(rec.get('reliability') or {}).get('overall_reliability')} "
                f"dec={rec.get('adaptive_decision')} "
                f"faith={(rec.get('faithfulness') or {}).get('score')} "
                f"relv={(rec.get('answer_relevance') or {}).get('score')} "
                f"gen_s={(rec.get('timings') or {}).get('generation_seconds')}"
            )
            print(detail + " | " + str(rec.get("generated_answer"))[:60], flush=True)
            cond_log.append(detail)
    elapsed = round(time.perf_counter() - t0, 2)
    summary = summarize_condition(cond, records, elapsed)
    return records, summary


# ---------------------------------------------------------------------------
# Reference conditions (llama3.2) - REUSED from the ablation run.
# ---------------------------------------------------------------------------
def reference_records(ref_by_id, context):
    """Relabel ablation A0/A1 records as llama3.2 FULL/PLAIN (no LLM calls)."""
    cond = make_condition(REFERENCE_MODEL, context)
    ref_cid = cond["reference_condition"]
    recs = []
    for qid in sorted({k for k in ref_by_id[ref_cid]}):
        r = ref_by_id[ref_cid][qid]
        new = dict(r)
        new["condition"] = condition_name(cond)
        new["model"] = cond["model"]
        new["context"] = cond["context"]
        new["reference_condition"] = ref_cid
        new["reference_used"] = True
        new["prompt_reference_matches"] = True
        recs.append(new)
    return recs


def build_reference(cond_log):
    """Construct llama3.2 FULL and PLAIN records (no LLM calls)."""
    with open(ABLATION_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    pq = data["per_question_results"]
    ref_by_id = {c: {r["id"]: r for r in pq[c]} for c in ("A0", "A1")}
    conds = {}
    recs = {}
    for context in CONTEXTS:
        cond = make_condition(REFERENCE_MODEL, context)
        conds[context] = cond
        recs[context] = reference_records(ref_by_id, context)
        note = (
            f"{condition_name(cond)} : {len(recs[context])} records REUSED from "
            f"ablation condition {cond['reference_condition']} "
            "(no generation/judge calls)."
        )
        print(note, flush=True)
        cond_log.append(note)
    return conds, recs, ref_by_id


def summarize_reference(cond, records):
    return summarize_condition(cond, records, 0.0, reference_used=True)


# ---------------------------------------------------------------------------
# Statistics (paired deltas + Wilcoxon, identical to the ablation evaluator).
# ---------------------------------------------------------------------------
def paired_deltas(base_records, ablated_records, key):
    """Per-question paired deltas (ablated - base) for a numeric metric."""
    by_id = {}
    for r in ablated_records:
        val = r.get(key)
        if isinstance(val, dict):
            val = val.get("score")
        if val is not None:
            try:
                by_id[r["id"]] = float(val)
            except (TypeError, ValueError):
                pass
    deltas = []
    for b in base_records:
        val = b.get(key)
        if isinstance(val, dict):
            val = val.get("score")
        if val is None:
            continue
        try:
            b_val = float(val)
        except (TypeError, ValueError):
            continue
        if b["id"] in by_id:
            deltas.append((b["id"], cround(by_id[b["id"]] - b_val, 4)))
    return deltas


def run_statistics(records_by_cond, base="llama3.2:latest|FULL",
                   metrics=("faithfulness", "answer_relevance")):
    """
    Paired Wilcoxon signed-rank (two-sided) for every condition vs the
    reference FULL condition, plus within-model FULL vs PLAIN comparisons.
    n <= 16: results are descriptive; not proof of significance.
    """
    from scipy.stats import wilcoxon

    out = {}
    base_records = records_by_cond[base]

    # 1) each condition vs reference FULL
    for cname, recs in sorted(records_by_cond.items()):
        if cname == base:
            continue
        out[cname] = {"vs_reference_FULL": {}}
        ref_key = f"vs_reference_FULL"
        for metric in metrics:
            deltas = paired_deltas(base_records, recs, metric)
            d_vals = [d for _, d in deltas]
            entry = _wilcoxon_entry(deltas, d_vals, metric)
            out[cname][ref_key][metric] = entry

    # 2) within-model context effect: FULL vs PLAIN (plain - full)
    full_key = f"{'llama3.2:latest'}|FULL"
    for model in [REFERENCE_MODEL] + BASELINE_MODELS:
        full_name = f"{model}|FULL"
        plain_name = f"{model}|PLAIN"
        if full_name not in records_by_cond or plain_name not in records_by_cond:
            continue
        out[f"context_effect|{model}"] = {"vs_own_FULL": {}}
        for metric in metrics:
            deltas = paired_deltas(records_by_cond[full_name], records_by_cond[plain_name], metric)
            d_vals = [d for _, d in deltas]
            entry = _wilcoxon_entry(deltas, d_vals, metric)
            out[f"context_effect|{model}"]["vs_own_FULL"][metric] = entry
    return out


def _wilcoxon_entry(deltas, d_vals, metric):
    entry = {
        "n_paired": len(d_vals),
        "deltas_per_q": [{"id": qid, "delta": d} for qid, d in deltas],
        "delta_mean": cround(statistics.mean(d_vals), 4) if d_vals else None,
        "delta_median": cround(statistics.median(d_vals), 4) if d_vals else None,
        "delta_stdev": cround(statistics.stdev(d_vals), 4) if len(d_vals) > 1 else None,
    }
    if not d_vals:
        entry["test"] = "not_applicable"
        entry["note"] = "no paired values (judge parse failures or missing rows)"
    elif sum(1 for v in d_vals if v != 0) == 0:
        entry["test"] = "wilcoxon_signed_rank"
        entry["statistic"] = 0.0
        entry["p_value"] = None
        entry["all_deltas_zero"] = True
        entry["note"] = "identical per-question values; test not meaningful"
    elif len(d_vals) < 6:
        entry["test"] = "not_run"
        entry["note"] = f"n={len(d_vals)} too small for a meaningful paired test"
    else:
        try:
            from scipy.stats import wilcoxon
            stat, p = wilcoxon(d_vals, zero_method="wilcox", alternative="two-sided")
            mean_d = statistics.mean(d_vals)
            sd_d = statistics.stdev(d_vals) if len(d_vals) > 1 else 0.0
            dz = mean_d / sd_d if sd_d else (0.0 if mean_d == 0 else float("inf"))
            entry["test"] = "wilcoxon_signed_rank"
            entry["statistic"] = cround(float(stat), 4)
            entry["p_value"] = cround(float(p), 6)
            entry["effect_size_cohens_dz"] = cround(dz, 4)
            entry["note"] = "descriptive only; small n; dz undefined/inflated when sd=0"
        except Exception as exc:  # noqa: BLE001
            entry["test"] = "error"
            entry["note"] = str(exc)
    return entry
def condition_name(cond):
    return f"{cond['model']}|{cond['context']}"


def jsonable(x):
    if isinstance(x, Counter):
        return {str(k): jsonable(v) for k, v in sorted(x.items())}
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def pct(n, d):
    return round(100.0 * n / d, 2) if d else 0.0


def strip_think_blocks(text):
    """Remove model-emitted reasoning blocks (e.g. qwen3 <|think|> tags).

    The RQ2 judge rubric is defined for the final answer only, so any
    thinking trails are stripped before scoring. Harmless when absent.
    """
    return re.sub(r"<\|?think\|?>.*?</\|?think\|?>", " ", text,
                  flags=re.DOTALL).strip()

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save_fig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figure {name}")
    return name


def make_figures(summaries, all_records, n_questions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures = []
    order = [f"llama3.2:latest|{c}" for c in CONTEXTS]
    for m in BASELINE_MODELS:
        order += [f"{m}|{c}" for c in CONTEXTS]
    short = [o.replace(":latest", "") for o in order]

    # Figure 1: faithfulness and relevance by condition
    faith = [summaries[c]["answer"]["faithfulness_mean"] or 0.0 for c in order]
    relv = [summaries[c]["answer"]["answer_relevance_mean"] or 0.0 for c in order]
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(order))
    w = 0.38
    ax.bar(x - w / 2, faith, w, label="Faithfulness", color="#4caf50")
    ax.bar(x + w / 2, relv, w, label="Answer relevance", color="#ff5722")
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Mean judge score (0-1)")
    ax.set_title(f"External Baseline LLMs - Answer Quality ({n_questions} gold questions)")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    figures.append(save_fig(fig, "fig1_baseline_quality.png"))

    # Figure 2: FULL - PLAIN context delta per model (faith/relevance)
    fig, ax = plt.subplots(figsize=(10, 5))
    models = ["llama3.2:latest"] + BASELINE_MODELS
    d_f = []
    d_r = []
    for m in models:
        fn = f"{m}|FULL"
        pn = f"{m}|PLAIN"
        s_f, s_p = summaries[fn], summaries[pn]
        d_f.append((s_p["answer"]["faithfulness_mean"] or 0.0)
                   - (s_f["answer"]["faithfulness_mean"] or 0.0))
        d_r.append((s_p["answer"]["answer_relevance_mean"] or 0.0)
                   - (s_f["answer"]["answer_relevance_mean"] or 0.0))
    x = np.arange(len(models))
    ax.bar(x - 0.2, d_f, 0.4, label="Faithfulness (PLAIN - FULL)", color="#2196f3")
    ax.bar(x + 0.2, d_r, 0.4, label="Relevance (PLAIN - FULL)", color="#ff9800")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(":latest", "") for m in models],
                       rotation=20, ha="right")
    ax.set_ylabel("Delta (0-1)")
    ax.set_title("Context Effect: PLAIN vs FULL prompt, per generation model")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    figures.append(save_fig(fig, "fig2_context_effect.png"))

    # Figure 3: generation latency per model
    gen_mean = [summaries[c]["generation_seconds_mean"] or 0.0 for c in order]
    fig, ax = plt.subplots(figsize=(12, 5))
    x3 = np.arange(len(order))
    ax.bar(x3, gen_mean, color="#9c27b0")
    for i, v in enumerate(gen_mean):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("Mean generation seconds / question")
    ax.set_xticks(x3)
    ax.set_xticklabels(short, rotation=40, ha="right", fontsize=9)
    ax.set_title("Generation Latency per (model, context)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    figures.append(save_fig(fig, "fig3_generation_latency.png"))

    return figures
# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def fmt_table(headers, rows):
    sep = "|" + "|".join(["--:" for _ in headers]) + "|"
    lines = ["|" + "|".join(headers) + "|", sep]
    for row in rows:
        lines.append("|" + "|".join(str(v) for v in row) + "|")
    return "\n".join(lines)


def write_report(summaries, conditions, stats, figures, metadata, cond_log, n_questions):
    rows = []
    A = rows.append

    A("# ElderDocAI - External LLM Generation Baselines Report")
    A("")
    A(f"> Synthetic longitudinal records (Synthea-derived) + {n_questions} gold QA questions.")
    A("> Every condition uses the SAME retrieval (hybrid top-3), same reliability "
      "config, same grounded-prompt templates, and the SAME judge model/rubric "
      "(llama3.2). ONLY the generation model differs between the reference and "
      "the baseline conditions.")
    A("")
    A("## 1. Design")
    A("")
    A(fmt_table(
        ["Condition", "Generation model", "Context", "Reference ablation cond."],
        [["|".join([c["model"], c["context"]]), c["model"], c["context"],
          c.get("reference_condition")] for c in conditions],
    ))
    A("")
    A("- FULL = complete ElderDocAI grounded prompt (dynamic care state + "
      "adaptive assistance plan + reliability gate); byte-identical to ablation A0.")
    A("- PLAIN = static profile + plain RAG grounded prompt; byte-identical to "
      "ablation A1.")
    A(f"- Reference generator (`{REFERENCE_MODEL}`) is not re-run: FULL/PLAIN "
      "records are reused from ablation conditions A0/A1 (verified prompt-"
      "identical at build time via SHA-256). Baseline models are run fresh.")
    A("")
    A("## 2. Per-condition results")
    A("")
    order = ["llama3.2:latest|" + c for c in CONTEXTS]
    for m in BASELINE_MODELS:
        order += [m + "|" + c for c in CONTEXTS]
    A(fmt_table(
        ["Condition", "Faithfulness", "Relevance", "Reliability", "Gen s (mean)",
         "Judge s (mean)", "Errors", "Judge parse failures"],
        [
            [
                c,
                summaries[c]["answer"].get("faithfulness_mean"),
                summaries[c]["answer"].get("answer_relevance_mean"),
                summaries[c]["reliability"].get("mean"),
                summaries[c].get("generation_seconds_mean"),
                summaries[c].get("judge_seconds_mean"),
                summaries[c]["n_errors"],
                summaries[c]["answer"].get("judge_parse_failures"),
            ]
            for c in order
        ],
    ))
    A("")
    A("## 3. Delta vs reference FULL (llama3.2 FULL)")
    A("")
    base = "llama3.2:latest|FULL"
    A(fmt_table(
        ["Condition", "Faithfulness delta", "Relevance delta"],
        [
            [
                c,
                (round(summaries[c]["answer"]["faithfulness_mean"]
                       - summaries[base]["answer"]["faithfulness_mean"], 4)
                 if summaries[c]["answer"].get("faithfulness_mean") is not None
                 else None),
                (round(summaries[c]["answer"]["answer_relevance_mean"]
                       - summaries[base]["answer"]["answer_relevance_mean"], 4)
                 if summaries[c]["answer"].get("answer_relevance_mean") is not None
                 else None),
            ]
            for c in order if c != base
        ],
    ))
    A("")
    A("## 4. Reliability / grounding note")
    A("")
    A("Because all conditions use the same hybrid retrieval (top-3, reranked) and "
      "the same reliability configuration, the retrieved evidence, grounding "
      "metrics, and reliability/decision scores are identical across every "
      "condition (any difference would indicate a data-integrity problem). "
      "Reliability decision distribution: "
      + json.dumps(list({str(summaries[c]["reliability"]["decision_distribution"])
                         for c in order}))
      + ".")
    A("")
    A("## 5. Statistical analysis (paired Wilcoxon vs reference FULL)")
    A("")
    for group, inner in sorted(stats.items()):
        if "vs_reference_FULL" in inner:
            A(f"### {group} vs reference FULL")
            A("")
            A(fmt_table(
                ["Metric", "n", "Delta mean", "Delta median", "Test", "p-value",
                 "dz", "Note"],
                [
                    [metric, e["n_paired"], e["delta_mean"], e["delta_median"],
                     e.get("test"), e.get("p_value"),
                     e.get("effect_size_cohens_dz"), e.get("note", "")]
                    for metric, e in inner["vs_reference_FULL"].items()
                ],
            ))
            A("")
        else:
            A(f"## {group} (context effect: PLAIN minus FULL)")
            A("")
            A(fmt_table(
                ["Metric", "n paired", "Delta mean", "Delta median", "Test",
                 "p-value", "dz", "Note"],
                [
                    [metric, e["n_paired"], e["delta_mean"], e["delta_median"],
                     e.get("test"), e.get("p_value"),
                     e.get("effect_size_cohens_dz"), e.get("note", "")]
                    for metric, e in inner["vs_own_FULL"].items()
                ],
            ))
            A("")
    A("## 6. Figures")
    A("")
    for f in figures:
        A(f"- `{f}`")
    A("")
    A("## 7. Limitations")
    A("")
    A("- All generation models are small local models (<=9B); no large external "
      "API model is included.")
    A("- The judge is llama3.2 (same as RQ2/ablation) for cross-condition "
      "comparability; it may be biased toward the reference generator's style.")
    A(f"- {n_questions} curated in-scope gold questions; ceiling effects apply.")
    A("- The longitudinal records are SYNTHETIC (Synthea-derived); no clinical "
      "validity is claimed.")
    return "\n".join(rows)
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="ElderDocAI external LLM baselines (Gold-QA).",
    )
    parser.add_argument(
        "--qa-file",
        default=DEFAULT_QA_FILE,
        help="Path to the Gold-QA dataset JSON "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--ablation-json",
        default=DEFAULT_ABLATION_JSON,
        help="Ablation results JSON referenced for the llama3.2 FULL/PLAIN "
             "reference records (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUT_DIR,
        help="Directory for LLM baseline outputs "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional: evaluate only the first N gold questions "
             "(0 = all; use a small N for a smoke test).",
    )
    args = parser.parse_args(argv)

    # Thread selected paths through module-level globals so build_reference
    # and the result writers honour the choice without touching analysis logic.
    global ABLATION_JSON, OUT_DIR, FIG_DIR, RESULTS_JSON, REPORT_MD, CONSOLE_LOG
    ABLATION_JSON = os.path.abspath(args.ablation_json)
    OUT_DIR = os.path.abspath(args.output_dir)
    FIG_DIR = os.path.join(OUT_DIR, "figures")
    RESULTS_JSON = os.path.join(OUT_DIR, "llm_baselines_results.json")
    REPORT_MD = os.path.join(OUT_DIR, "llm_baselines_report.md")
    CONSOLE_LOG = os.path.join(OUT_DIR, "llm_baselines_console_output.txt")

    os.makedirs(OUT_DIR, exist_ok=True)
    cond_log = []
    t0 = time.perf_counter()

    print("=" * 70, flush=True)
    print("ELDERDOCAI EXTERNAL LLM BASELINES", flush=True)
    print("=" * 70, flush=True)
    cond_log.append("ELDERDOCAI EXTERNAL LLM BASELINES")

    with open(args.qa_file, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
    gold_questions = gold_data["gold_questions"]
    if args.limit and args.limit > 0:
        gold_questions = gold_questions[: args.limit]

    patient_ctx = build_patient_context()
    cond_log.append(
        "patient_ctx: strategy=" + str(patient_ctx["assistance_plan_strategy"])
    )

    # ---- Reference conditions (reused from ablation A0/A1) ---------------
    print("\n### REFERENCE CONDITIONS (reused from ablation)", flush=True)
    ref_conds, ref_recs, ref_by_id = build_reference(cond_log)
    records_by_cond = {}
    summaries = {}
    for context in CONTEXTS:
        cond = ref_conds[context]
        cname = condition_name(cond)
        records_by_cond[cname] = ref_recs[context]
        summaries[cname] = summarize_reference(cond, ref_recs[context])

    # ---- Baseline conditions (fresh generation + judges) ----------------
    for model in BASELINE_MODELS:
        for context in CONTEXTS:
            cond = make_condition(model, context)
            cname = condition_name(cond)
            print(f"\n### CONDITION {cname}", flush=True)
            cond_log.append(f"### CONDITION {cname}")
            ref_hashes = {
                qid: r["prompt_sha256"]
                for qid, r in ref_by_id[cond["reference_condition"]].items()
            }
            recs, summary = run_condition(
                cond, gold_questions, patient_ctx, cond_log,
                expected_hashes_by_qid=ref_hashes,
            )
            records_by_cond[cname] = recs
            summaries[cname] = summary
            cond_log.append("")

    # ---- Prompt-equivalence validation across all baseline conditions ----
    print("\n== Prompt-equivalence validation ==", flush=True)
    mismatch_flags = []
    for cname, recs in records_by_cond.items():
        if recs and recs[0].get("reference_used"):
            continue
        bad = [r["id"] for r in recs if r.get("prompt_reference_matches") is False]
        if bad:
            mismatch_flags.append({"condition": cname, "questions": bad})
    if mismatch_flags:
        raise RuntimeError(
            "Baseline prompt-equivalence check FAILED: conditions/prompts differ "
            "from the reference. Details: " + json.dumps(mismatch_flags)
        )
    prompt_equivalence = {}
    for cname, recs in records_by_cond.items():
        prompt_equivalence[cname] = {
            r["id"]: {
                "prompt_sha256": r.get("prompt_sha256"),
                "reference_matches": r.get("prompt_reference_matches"),
            }
            for r in recs
        }
    print(f"  {len(gold_questions)}/{len(gold_questions)} prompt hashes match the reference "
          "for every baseline condition.", flush=True)
    cond_log.append(
        f"Baseline prompt-equivalence: {len(gold_questions)}/{len(gold_questions)} "
        "match the reference."
    )

    # ---- Statistics ----
    print("\n== Paired statistics ==", flush=True)
    stats = run_statistics(records_by_cond)

    # ---- Figures ----
    figures = make_figures(summaries, records_by_cond, n_questions=len(gold_questions))

    # ---- Assemble result ----
    conditions = [make_condition(REFERENCE_MODEL, c) for c in CONTEXTS]
    for m in BASELINE_MODELS:
        for c in CONTEXTS:
            conditions.append(make_condition(m, c))
    result = {
        "metadata": {
            "title": "ElderDocAI External LLM Generation Baselines",
            "n_gold_questions_actual": len(gold_questions),
            "reference_model": REFERENCE_MODEL,
            "judge_model": JUDGE_MODEL,
            "baseline_models": BASELINE_MODELS,
            "contexts": CONTEXTS,
            "sampling": GEN_OPTIONS,
            "reliability_config": "config/reliability_config.json",
            "disclaimer": "SYNTHETIC data; architectural comparison only; "
                          "no clinical benefit claimed.",
        },
        "condition_definitions": jsonable(conditions),
        "per_condition_aggregates": jsonable(summaries),
        "per_question_results": jsonable(records_by_cond),
        "prompt_equivalence_per_question": jsonable(prompt_equivalence),
        "prompt_equivalence_all_passed": not mismatch_flags,
        "paired_statistics": jsonable(stats),
        "figures": figures,
        "runtime_seconds": round(time.perf_counter() - t0, 2),
        "errors_by_condition": {
            cname: [r.get("id") for r in recs if r.get("error")]
            for cname, recs in records_by_cond.items()
        },
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  wrote {RESULTS_JSON}")

    report = write_report(
        summaries, conditions, stats, figures, result["metadata"], cond_log,
        n_questions=len(gold_questions),
    )
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  wrote {REPORT_MD}")

    with open(CONSOLE_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(cond_log))
    print(f"  wrote {CONSOLE_LOG}")

    print("\n" + "=" * 70, flush=True)
    print("MAIN TABLE", flush=True)
    for cname in sorted(summaries):
        s = summaries[cname]
        print(
            f"{cname:-<30} acc={s['retrieval']['source_accuracy_pct']}% "
            f"faith={s['answer']['faithfulness_mean']} "
            f"relv={s['answer']['answer_relevance_mean']} "
            f"rel={s['reliability']['mean']} "
            f"gen_s={s['generation_seconds_mean']} "
            f"errors={s['n_errors']}",
            flush=True,
        )
    print("=" * 70, flush=True)
    print("LLM BASELINES COMPLETE", flush=True)


if __name__ == "__main__":
    main()