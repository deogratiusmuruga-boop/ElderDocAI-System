#!/usr/bin/env python3
"""Experiment 2 - Reliability-Gating Effectiveness (paired ablation).

Condition A (Gate ON):  exact current production gate flow, replaying the
  frozen `_run_gated_generation` algorithm (REFINE/RE-RETRIEVE/REJECT with
  MAX_REFINE=1 / MAX_RETRIEVE=1) on a single shared initial retrieval.
Condition B (Gate OFF): same initial evidence, same grounded prompt template,
  same system prompt, same generation options; reliability is computed and
  recorded but NOT enforced (no refine/re-retrieve/reject).

Both conditions share the IDENTICAL initial retrieval evidence so the paired
comparison isolates reliability-gated evidence handling and refinement.

Evaluation methodology is byte-for-byte that of Experiment 1:
  FAITHFULNESS_PROMPT / RELEVANCE_PROMPT judges (llama3.2, deterministic
  sampler) and deterministic span_token_coverage(...) evidence support.

All frozen artifacts are read-only. Writes ONLY under this directory.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[4]   # ElderDocAI-System
OUT_DIR = Path(__file__).resolve().parent

os.environ["ELDERDOCAI_RETRIEVAL_BACKEND"] = "geri_lit"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

GERE_DIR = BASE_DIR / "datasets" / "elderdocai_geriLit"
if str(GERE_DIR) not in sys.path:
    sys.path.insert(0, str(GERE_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import ollama  # noqa: E402

from scripts.rag_chat import (  # noqa: E402
    LLM_MODEL,
    GENERATION_SYSTEM_PROMPT,
    REJECTION_RESPONSE,
    EMPTY_EVIDENCE_RESPONSE,
    MAX_REFINE_ATTEMPTS,
    MAX_RETRIEVE_ATTEMPTS,
    prepare_evidence,
    _refine_evidence,   # noqa: PLC2701 (frozen production helper, read-only)
)
from scripts.retrieval_router import retrieve as retrieve_evidence  # noqa: E402
from scripts.reliability_evaluation import evaluate_reliability  # noqa: E402
from scripts.adaptive_decision_controller import (  # noqa: E402
    make_reliability_decision,
)
from scripts.build_grounded_prompt import build_grounded_prompt  # noqa: E402
from scripts.evaluate_gold_qa import (  # noqa: E402
    FAITHFULNESS_PROMPT,
    RELEVANCE_PROMPT,
    build_evidence_text,
    span_token_coverage,
)

JUDGE_MODEL = "llama3.2:latest"
JUDGE_SYSTEM_FAITHFULNESS = "You evaluate RAG faithfulness only."
JUDGE_SYSTEM_RELEVANCE = "You evaluate RAG answer relevance only."
JUDGE_OPTIONS = {"temperature": 0, "top_p": 0.1, "top_k": 10}

BENCHMARK_FILE = BASE_DIR / "data" / "geri_lit_gold_v1_1.json"
EXPECTED_BENCHMARK_SHA256 = (
    "1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72"
)

FROZEN_FILES = {
    "benchmark_v1_0": BASE_DIR / "data" / "geri_lit_gold.json",
    "benchmark_v1_1": BENCHMARK_FILE,
    "chunks": GERE_DIR / "chunks" / "chunks.jsonl",
    "embeddings": GERE_DIR / "index" / "embeddings.npy",
    "faiss": GERE_DIR / "index" / "faiss_index.bin",
    "bm25": GERE_DIR / "index" / "bm25.pkl",
    "row_mapping": GERE_DIR / "index" / "row_mapping.json",
    "reliability_config": BASE_DIR / "config" / "reliability_config.json",
}
FROZEN_EVAL_FILES = {
    "task10d_results": GERE_DIR / "metadata" / "task10d_retrieval_results.json",
    "task10d_summary": GERE_DIR / "metadata" / "task10d_summary.json",
    "task10e_analysis": GERE_DIR / "metadata" / "task10e_failure_analysis.json",
    "task3_summary": GERE_DIR
    / "metadata"
    / "task3_reliability_evaluation"
    / "task3_summary.json",
    "exp1_summary": GERE_DIR
    / "metadata"
    / "experiment1_generation_evaluation"
    / "experiment1_summary_run1.json",
    "exp1_manifest": GERE_DIR
    / "metadata"
    / "experiment1_generation_evaluation"
    / "experiment1_run_manifest.json",
}

GENERATION_RETRY_LIMIT = 2
JUDGE_RETRY_LIMIT = 2
TOPIC_CODES = ["C01", "C02", "C03", "C04", "C05",
               "C06", "C07", "C08", "C09", "C10"]


def sha256(path):
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_frozen():
    snap = {}
    for name, path in FROZEN_FILES.items():
        snap[name] = sha256(path)
    for name, path in FROZEN_EVAL_FILES.items():
        snap[name] = sha256(path)
    return snap
def judge(question, answer, evidence_items, prompt_template, system_content):
    """LLM judge, methodology byte-identical to Experiment 1.

    - Prompt templates reused verbatim from scripts/evaluate_gold_qa.py.
    - Deterministic sampler options (temperature 0, top_p 0.1, top_k 10).
    Returns (score or None, reason or error-message).
    """
    text = build_evidence_text(evidence_items) if evidence_items else (
        "(no evidence retrieved)")
    prompt = prompt_template.format(
        question=question, answer=answer, evidence=text)
    last_error = None
    for attempt in range(JUDGE_RETRY_LIMIT + 1):
        try:
            response = ollama.chat(
                model=JUDGE_MODEL,
                format="json",
                options=JUDGE_OPTIONS,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
            )
            parsed = json.loads(response["message"]["content"])
            return float(parsed["score"]), str(parsed.get("reason", ""))
        except Exception as exc:  # noqa: BLE001
            last_error = "%s: %s" % (type(exc).__name__, exc)
            time.sleep(0.5)
    return None, "judge-error: %s" % last_error


def load_benchmark():
    """Load and validate the frozen GeriLit-Gold v1.1 benchmark."""
    actual = sha256(BENCHMARK_FILE)
    if actual != EXPECTED_BENCHMARK_SHA256:
        raise SystemExit(
            "BENCHMARK HASH MISMATCH: expected %s actual %s"
            % (EXPECTED_BENCHMARK_SHA256, actual))
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]
    if len(records) != 121:
        raise SystemExit("Unexpected record count: %d" % len(records))
    ids = [r["final_benchmark_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise SystemExit("Duplicate final_benchmark_id found.")
    if any(r.get("review_status") != "ACCEPTED" for r in records):
        raise SystemExit("Not all records are ACCEPTED.")
    return records


def _postprocess_answer(answer):
    """Replicate production answer post-processing (rag_chat.generate_answer)."""
    answer = str(answer or "").strip()
    if "\nSources:" in answer:
        answer = answer.split("\nSources:", 1)[0].strip()
    elif answer.startswith("Sources:"):
        answer = ""
    if answer.startswith("Answer:"):
        answer = answer[len("Answer:"):].strip()
    return answer


def _llm_generate(query, evidence_items, reliability, decision):
    """Grounded generation with the frozen production prompt/options.

    Byte-identical construction to scripts/rag_chat.py::_run_gated_generation:
    build_grounded_prompt(query, evidence_items, reliability, decision,
    user_profile=None, conversation_context="", response_language="en",
    assistance_plan={}) + GENERATION_SYSTEM_PROMPT + ollama.chat options.
    """
    prompt = build_grounded_prompt(
        query=query,
        evidence_items=evidence_items,
        reliability=reliability,
        decision=decision,
        user_profile=None,
        conversation_context="",
        response_language="en",
        assistance_plan={},
    )
    response = ollama.chat(
        model=LLM_MODEL,
        options={"temperature": 0, "top_p": 0.1, "top_k": 10},
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return _postprocess_answer(response["message"]["content"])
def _replay_gate_on(query, initial_evidence, initial_rel, initial_dec):
    """Gate ON - replay the frozen production gate algorithm exactly.

    Mirrors scripts/rag_chat.py::_run_gated_generation (budgets
    MAX_REFINE=1 / MAX_RETRIEVE=1) but begins from the shared initial
    evidence (production retrieves first, then runs this loop). RE-RETRIEVE
    calls retrieve_evidence() which is deterministic and returns identical
    chunks. Returns the same dict shape as production.
    """
    evidence_items = list(initial_evidence)
    reliability = dict(initial_rel)
    decision = dict(initial_dec)
    retrieval_attempts = 1          # shared initial retrieval already performed
    refinement_attempts = 0
    refused = False
    why_refused = ""
    guard = 0
    max_guard = 8

    while guard < max_guard:
        guard += 1
        label = decision["decision"]

        if label == "ACCEPT":
            break

        if label == "REJECT":
            refused = True
            why_refused = "REJECT"
            break

        if label == "REFINE" and refinement_attempts < MAX_REFINE_ATTEMPTS:
            refinement_attempts += 1
            refined = _refine_evidence(evidence_items, query)
            if not refined:
                refused = True
                why_refused = "REFINE_EMPTY"
                evidence_items = []
            else:
                evidence_items = refined
            reliability = evaluate_reliability(
                query=query, evidence_items=evidence_items)
            decision = make_reliability_decision(reliability)
            continue

        if (label == "RE-RETRIEVE"
                and retrieval_attempts <= MAX_RETRIEVE_ATTEMPTS):
            retrieval_attempts += 1
            re_chunks = retrieve_evidence(query)
            if not re_chunks:
                refused = True
                why_refused = "RERETRIEVE_EMPTY"
                break
            evidence_items = prepare_evidence(re_chunks)
            reliability = evaluate_reliability(
                query=query, evidence_items=evidence_items)
            decision = make_reliability_decision(reliability)
            continue

        # REFINE with refinement budget exhausted: generate from the
        # once-refined evidence, provided it remains (production behavior).
        if label == "REFINE" and evidence_items:
            break

        refused = True
        why_refused = label or "UNKNOWN"
        break

    if refused:
        answer_text = (
            EMPTY_EVIDENCE_RESPONSE if not evidence_items
            else REJECTION_RESPONSE)
        return {
            "answer": answer_text,
            "evidence_items": evidence_items,
            "reliability": reliability,
            "decision": decision,
            "retrieval_attempts": retrieval_attempts,
            "refinement_attempts": refinement_attempts,
            "refused": True,
            "message": why_refused,
        }

    answer_text = _llm_generate(query, evidence_items, reliability, decision)
    return {
        "answer": answer_text,
        "evidence_items": evidence_items,
        "reliability": reliability,
        "decision": decision,
        "retrieval_attempts": retrieval_attempts,
        "refinement_attempts": refinement_attempts,
        "refused": False,
        "message": "generated",
    }


def _run_gate_off(query, initial_evidence, initial_rel, initial_dec):
    """Gate OFF - bypass reliability enforcement.

    Generates directly from the SHARED initial evidence with the same
    grounded-prompt template, system prompt, and generation options. No
    refinement, no re-retrieval, no rejection, no evidence change. Empty
    evidence returns the same production empty-evidence response (this is
    the retrieval-empty path, not a gate decision).
    """
    if not initial_evidence:
        return {
            "answer": EMPTY_EVIDENCE_RESPONSE,
            "evidence_items": [],
            "reliability": initial_rel,
            "decision": initial_dec,
            "retrieval_attempts": 1,
            "refinement_attempts": 0,
            "refused": True,
            "message": "EMPTY_EVIDENCE",
        }
    answer_text = _llm_generate(
        query, initial_evidence, initial_rel, initial_dec)
    return {
        "answer": answer_text,
        "evidence_items": list(initial_evidence),
        "reliability": initial_rel,
        "decision": initial_dec,
        "retrieval_attempts": 1,
        "refinement_attempts": 0,
        "refused": False,
        "message": "generated",
    }
def _retry_call(fn, retries=GENERATION_RETRY_LIMIT, sleep=1.0):
    """Call fn() with a fixed retry policy (protocol section 3)."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            return fn(), None
        except Exception as exc:  # noqa: BLE001
            last_error = "%s: %s" % (type(exc).__name__, exc)
            traceback.print_exc()
            time.sleep(sleep)
    return None, last_error


def process_question(record):
    """Build one complete paired Gate ON / Gate OFF record.

    Shared retrieval: ONE retrieve_evidence call per question supplies the
    initial evidence to BOTH conditions (protocol section 3).
    """
    qid = record["final_benchmark_id"]
    question = record["question"]
    gold_chunks = record.get("gold_relevant_chunk_ids") or []

    row = {
        "question_id": qid,
        "topic": record.get("topic"),
        "question": question,
        "gold_pmcid": record.get("pmcid"),
        "gold_chunk_ids": gold_chunks,
        "initial_evidence_ids": [],
        "initial_reliability": None,
        "initial_decision": None,
        "gate_on": {
            "evidence_ids": [], "reliability": None, "decision": None,
            "refinement_attempts": 0, "retrieval_attempts": 1,
            "refused": False, "generated": False, "answer": "",
            "faithfulness": None, "faithfulness_reason": "",
            "answer_relevance": None, "answer_relevance_reason": "",
            "evidence_support": None, "hallucination": None,
            "gold_chunk_retrieved": False,
            "latency_seconds": None, "error": None,
        },
        "gate_off": {
            "evidence_ids": [], "reliability": None, "decision": None,
            "refinement_attempts": 0, "retrieval_attempts": 1,
            "refused": False, "generated": False, "answer": "",
            "faithfulness": None, "faithfulness_reason": "",
            "answer_relevance": None, "answer_relevance_reason": "",
            "evidence_support": None, "hallucination": None,
            "gold_chunk_retrieved": False,
            "latency_seconds": None, "error": None,
        },
        "paired_differences": {
            "faithfulness_delta": None,
            "answer_relevance_delta": None,
            "evidence_support_delta": None,
        },
    }

    # ---- shared initial retrieval + reliability (deterministic) ----
    try:
        initial_chunks = retrieve_evidence(question)
        initial_evidence = prepare_evidence(initial_chunks)
    except Exception as exc:  # noqa: BLE001
        row["gate_on"]["error"] = "retrieval: %s" % type(exc).__name__
        row["gate_off"]["error"] = "retrieval: %s" % type(exc).__name__
        return row

    row["initial_evidence_ids"] = [
        e.get("chunk_id") for e in initial_evidence]

    try:
        initial_rel = evaluate_reliability(
            query=question, evidence_items=initial_evidence)
        initial_dec = make_reliability_decision(initial_rel)
    except Exception as exc:  # noqa: BLE001
        row["gate_on"]["error"] = "reliability: %s" % type(exc).__name__
        row["gate_off"]["error"] = "reliability: %s" % type(exc).__name__
        return row

    row["initial_reliability"] = {
        k: round(float(v), 6) for k, v in initial_rel.items()}
    row["initial_decision"] = initial_dec["decision"]

    # ---- Condition A: Gate ON (replay production gate on shared evidence) ----
    t0 = time.perf_counter()
    on, on_err = _retry_call(
        lambda: _replay_gate_on(
            question, initial_evidence, initial_rel, initial_dec))
    row["gate_on"]["latency_seconds"] = round(time.perf_counter() - t0, 3)
    if on is None:
        row["gate_on"]["error"] = "gate-on failed: %s" % on_err
        row["gate_off"]["error"] = "gate-on failed: %s" % on_err
        return row

    gon = row["gate_on"]
    gon["answer"] = str(on["answer"]).strip()
    gon["generated"] = bool(on["answer"]) and not on["refused"]
    gon["refused"] = bool(on["refused"])
    gon["refinement_attempts"] = int(on["refinement_attempts"])
    gon["retrieval_attempts"] = int(on["retrieval_attempts"])
    r_on = on.get("reliability") or {}
    gon["reliability"] = {k: round(float(v), 6)
                          for k, v in r_on.items()}
    gon["decision"] = (on.get("decision") or {}).get("decision")
    gon["evidence_ids"] = [e.get("chunk_id")
                           for e in on.get("evidence_items") or []]
    gon["gold_chunk_retrieved"] = bool(
        set(gon["evidence_ids"]) & set(gold_chunks))
# ---- Condition B: Gate OFF (direct generation on shared evidence) ----
    t0 = time.perf_counter()
    off, off_err = _retry_call(
        lambda: _run_gate_off(
            question, initial_evidence, initial_rel, initial_dec))
    row["gate_off"]["latency_seconds"] = round(time.perf_counter() - t0, 3)
    if off is None:
        row["gate_off"]["error"] = "gate-off failed: %s" % off_err
        return row

    goff = row["gate_off"]
    goff["answer"] = str(off["answer"]).strip()
    goff["generated"] = bool(off["answer"]) and not off["refused"]
    goff["refused"] = bool(off["refused"])
    r_off = off.get("reliability") or {}
    goff["reliability"] = {k: round(float(v), 6)
                           for k, v in r_off.items()}
    goff["decision"] = (off.get("decision") or {}).get("decision")
    goff["evidence_ids"] = [e.get("chunk_id")
                            for e in off.get("evidence_items") or []]
    goff["gold_chunk_retrieved"] = bool(
        set(goff["evidence_ids"]) & set(gold_chunks))

    # ---- judges (identical methodology for both conditions) ----
    if gon["generated"] and gon["answer"]:
        score, reason = judge(
            question, gon["answer"], on["evidence_items"],
            FAITHFULNESS_PROMPT, JUDGE_SYSTEM_FAITHFULNESS)
        gon["faithfulness"] = score
        gon["faithfulness_reason"] = reason[:400]
        score, reason = judge(
            question, gon["answer"], on["evidence_items"],
            RELEVANCE_PROMPT, JUDGE_SYSTEM_RELEVANCE)
        gon["answer_relevance"] = score
        gon["answer_relevance_reason"] = reason[:400]
        gon["evidence_support"] = span_token_coverage(
            gon["answer"], "\n".join(e.get("text") or ""
                                     for e in on["evidence_items"]))
        gon["hallucination"] = bool(
            gon["faithfulness"] is not None
            and gon["faithfulness"] <= 0.50)

    if goff["generated"] and goff["answer"]:
        score, reason = judge(
            question, goff["answer"], off["evidence_items"],
            FAITHFULNESS_PROMPT, JUDGE_SYSTEM_FAITHFULNESS)
        goff["faithfulness"] = score
        goff["faithfulness_reason"] = reason[:400]
        score, reason = judge(
            question, goff["answer"], off["evidence_items"],
            RELEVANCE_PROMPT, JUDGE_SYSTEM_RELEVANCE)
        goff["answer_relevance"] = score
        goff["answer_relevance_reason"] = reason[:400]
        goff["evidence_support"] = span_token_coverage(
            goff["answer"], "\n".join(e.get("text") or ""
                                      for e in off["evidence_items"]))
        goff["hallucination"] = bool(
            goff["faithfulness"] is not None
            and goff["faithfulness"] <= 0.50)

    # ---- paired differences ----
    for metric, key in [
        ("faithfulness_delta", "faithfulness"),
        ("answer_relevance_delta", "answer_relevance"),
        ("evidence_support_delta", "evidence_support"),
    ]:
        a = gon.get(key)
        b = goff.get(key)
        if a is not None and b is not None:
            row["paired_differences"][metric] = round(float(a) - float(b), 6)

    return row
# ============================================================
# Statistics (pre-registered in experiment2_protocol.md section 4)
# ============================================================

import random  # noqa: E402

try:
    from scipy import stats as scipy_stats  # noqa: E402
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False


def _paired_stats(on_vals, off_vals):
    """Paired statistics for one metric (ON=Gate ON, OFF=Gate OFF).

    Only pairs where both values exist are used. Pre-registered tests:
    paired t-test (ttest_rel), Wilcoxon signed-rank, bootstrap 95% CI
    of the mean delta (B=10000, seed=0), Cohen's d_z effect size.
    """
    pairs = [(float(a), float(b))
             for a, b in zip(on_vals, off_vals)
             if a is not None and b is not None]
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    on = [p[0] for p in pairs]
    off = [p[1] for p in pairs]
    deltas = [a - b for a, b in pairs]
    md = statistics.mean(deltas)
    sdd = statistics.stdev(deltas) if n > 1 else 0.0

    out = {
        "n": n,
        "on_mean": round(statistics.mean(on), 6),
        "off_mean": round(statistics.mean(off), 6),
        "paired_mean_difference": round(md, 6),
        "median_difference": round(statistics.median(deltas), 6),
        "std_difference": round(sdd, 6),
        "count_on_gt_off": sum(1 for d in deltas if d > 1e-9),
        "count_on_eq_off": sum(1 for d in deltas if abs(d) <= 1e-9),
        "count_on_lt_off": sum(1 for d in deltas if d < -1e-9),
    }

    if HAVE_SCIPY:
        # Paired t-test (two-sided, H0: mean delta = 0)
        t_stat, t_p = scipy_stats.ttest_rel(on, off)
        out["paired_ttest_t"] = round(float(t_stat), 6)
        out["paired_ttest_p"] = round(float(t_p), 6)
        # 95% CI of the mean delta (t distribution with n-1 df)
        se = sdd / (n ** 0.5)
        ci = scipy_stats.t.interval(0.95, df=n - 1, loc=md, scale=se)
        out["ci95_mean_delta_t"] = [round(ci[0], 6), round(ci[1], 6)]
        # Wilcoxon signed-rank (two-sided, zero_method='wilcox')
        try:
            w_stat, w_p = scipy_stats.wilcoxon(
                on, off, zero_method="wilcox", alternative="two-sided")
            out["wilcoxon_statistic"] = round(float(w_stat), 6)
            out["wilcoxon_p"] = round(float(w_p), 6)
        except Exception as exc:  # noqa: BLE001
            out["wilcoxon_error"] = str(exc)
    else:
        out["stats_error"] = "scipy unavailable"

    # Bootstrap 95% percentile CI of mean delta (deterministic, seed=0)
    rng = random.Random(0)
    boot_means = []
    for _ in range(10000):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boot_means.append(statistics.mean(sample))
    boot_means.sort()
    lo = boot_means[int(0.025 * len(boot_means))]
    hi = boot_means[int(0.975 * len(boot_means))]
    out["ci95_mean_delta_bootstrap"] = [round(lo, 6), round(hi, 6)]

    # Effect size (paired Cohen's d_z)
    out["cohens_dz"] = round(md / sdd, 6) if sdd > 0 else None
    return out
def build_summary(rows):
    """Aggregate per-question paired results (protocol sections 4-6)."""
    summary = {"n_questions": len(rows)}
    primary = {
        "faithfulness": ([r["gate_on"]["faithfulness"] for r in rows],
                         [r["gate_off"]["faithfulness"] for r in rows]),
        "answer_relevance": (
            [r["gate_on"]["answer_relevance"] for r in rows],
            [r["gate_off"]["answer_relevance"] for r in rows]),
        "evidence_support": (
            [r["gate_on"]["evidence_support"] for r in rows],
            [r["gate_off"]["evidence_support"] for r in rows]),
    }
    summary["paired_statistics"] = {}
    for metric, (on_vals, off_vals) in primary.items():
        summary["paired_statistics"][metric] = _paired_stats(
            on_vals, off_vals)

    # gate behavior (Gate ON)
    dec_on = {}
    for r in rows:
        d = r["gate_on"]["decision"] or "MISSING"
        dec_on[d] = dec_on.get(d, 0) + 1
    summary["gate_on_decision_counts"] = dec_on
    summary["gate_on_decision_pct"] = {
        d: round(100.0 * c / len(rows), 2)
        for d, c in sorted(dec_on.items())}
    summary["gate_on_refinement_cases"] = sum(
        1 for r in rows if r["gate_on"]["refinement_attempts"] > 0)
    summary["gate_on_rere_retrieve_cases"] = sum(
        1 for r in rows if r["gate_on"]["retrieval_attempts"] > 1)
    summary["gate_on_rejection_cases"] = sum(
        1 for r in rows if r["gate_on"]["refused"])
    summary["gate_on_generation_permitted"] = sum(
        1 for r in rows if r["gate_on"]["generated"])
    summary["gate_off_generation_permitted"] = sum(
        1 for r in rows if r["gate_off"]["generated"])
    summary["gate_on_empty_evidence"] = sum(
        1 for r in rows if not r["initial_evidence_ids"])
    summary["errors"] = sum(
        1 for r in rows
        if r["gate_on"]["error"] or r["gate_off"]["error"])
# refinement analysis (descriptive; no answer-quality claim)
    refine_rows = [r for r in rows
                   if r["gate_on"]["refinement_attempts"] > 0]
    summary["refinement_analysis"] = {
        "n": len(refine_rows),
        "changed_evidence": sum(
            1 for r in refine_rows
            if r["gate_on"]["evidence_ids"]
            != r["initial_evidence_ids"]),
        "mean_initial_reliability": round(
            statistics.mean(
                float(r["initial_reliability"]["overall_reliability"])
                for r in refine_rows), 6),
        "mean_final_reliability": round(
            statistics.mean(
                float(r["gate_on"]["reliability"]["overall_reliability"])
                for r in refine_rows), 6),
        "note": (
            "No pre-refinement answer is generated by the architecture; "
            "answer-quality change cannot be causally isolated here."),
    }

    # gold evidence analysis (descriptive)
    g_on = sum(1 for r in rows if r["gate_on"]["gold_chunk_retrieved"])
    g_off = sum(1 for r in rows if r["gate_off"]["gold_chunk_retrieved"])
    summary["gold_evidence"] = {
        "gate_on_present": g_on,
        "gate_off_present": g_off,
        "paired_difference_on_minus_off": g_on - g_off,
    }

    # latency
    lat_on = [r["gate_on"]["latency_seconds"] for r in rows
              if r["gate_on"]["latency_seconds"] is not None]
    lat_off = [r["gate_off"]["latency_seconds"] for r in rows
               if r["gate_off"]["latency_seconds"] is not None]
    summary["latency_on_mean"] = round(
        statistics.mean(lat_on), 4) if lat_on else None
    summary["latency_off_mean"] = round(
        statistics.mean(lat_off), 4) if lat_off else None
# topic-level (descriptive)
    topics = {}
    for t in TOPIC_CODES:
        sub = [r for r in rows if r.get("topic") == t]
        if not sub:
            continue
        f_on = [r["gate_on"]["faithfulness"] for r in sub
                if r["gate_on"]["faithfulness"] is not None]
        f_off = [r["gate_off"]["faithfulness"] for r in sub
                 if r["gate_off"]["faithfulness"] is not None]
        rel_on = [r["gate_on"]["answer_relevance"] for r in sub
                  if r["gate_on"]["answer_relevance"] is not None]
        rel_off = [r["gate_off"]["answer_relevance"] for r in sub
                   if r["gate_off"]["answer_relevance"] is not None]
        s_on = [r["gate_on"]["evidence_support"] for r in sub
                if r["gate_on"]["evidence_support"] is not None]
        s_off = [r["gate_off"]["evidence_support"] for r in sub
                 if r["gate_off"]["evidence_support"] is not None]
        topics[t] = {
            "n": len(sub),
            "faithfulness_on_mean": round(statistics.mean(f_on), 6)
                if f_on else None,
            "faithfulness_off_mean": round(statistics.mean(f_off), 6)
                if f_off else None,
            "faithfulness_delta": round(
                statistics.mean(f_on) - statistics.mean(f_off), 6)
                if f_on and f_off else None,
            "relevance_on_mean": round(statistics.mean(rel_on), 6)
                if rel_on else None,
            "relevance_off_mean": round(statistics.mean(rel_off), 6)
                if rel_off else None,
            "relevance_delta": round(
                statistics.mean(rel_on) - statistics.mean(rel_off), 6)
                if rel_on and rel_off else None,
            "evidence_support_on_mean": round(statistics.mean(s_on), 6)
                if s_on else None,
            "evidence_support_off_mean": round(statistics.mean(s_off), 6)
                if s_off else None,
            "evidence_support_delta": round(
                statistics.mean(s_on) - statistics.mean(s_off), 6)
                if s_on and s_off else None,
        }
    summary["topic_level"] = topics
    return summary
def _short(text):
    return " ".join(str(text).strip().split())[:4000]


def _signature(rows):
    """Deterministic signature over substantive paired fields."""
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda x: x["question_id"]):
        payload = {
            "qid": r["question_id"],
            "initial_evidence_ids": r["initial_evidence_ids"],
            "initial_decision": r["initial_decision"],
            "on": {
                "answer": _short(r["gate_on"]["answer"]),
                "decision": r["gate_on"]["decision"],
                "reliability": r["gate_on"]["reliability"],
                "evidence_ids": r["gate_on"]["evidence_ids"],
                "refinement_attempts": r["gate_on"]["refinement_attempts"],
                "retrieval_attempts": r["gate_on"]["retrieval_attempts"],
                "refused": r["gate_on"]["refused"],
                "generated": r["gate_on"]["generated"],
                "faithfulness": r["gate_on"]["faithfulness"],
                "answer_relevance": r["gate_on"]["answer_relevance"],
                "evidence_support": r["gate_on"]["evidence_support"],
            },
            "off": {
                "answer": _short(r["gate_off"]["answer"]),
                "evidence_ids": r["gate_off"]["evidence_ids"],
                "refused": r["gate_off"]["refused"],
                "generated": r["gate_off"]["generated"],
                "faithfulness": r["gate_off"]["faithfulness"],
                "answer_relevance": r["gate_off"]["answer_relevance"],
                "evidence_support": r["gate_off"]["evidence_support"],
            },
            "deltas": r["paired_differences"],
        }
        h.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return h.hexdigest()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True, help="1 or 2")
    ap.add_argument("--limit", type=int, default=None,
                    help="optional diagnostic limit")
    args = ap.parse_args()
    run_id = args.run_id
    t_all = time.perf_counter()

    snap_before = snapshot_frozen()
    records = load_benchmark()
    print("Benchmark OK: %d records, hash verified" % len(records))
    target = records if args.limit is None else records[: args.limit]
    print("Evaluating %d paired questions (run %s)..." % (len(target), run_id))

    rows = []
    for i, rec in enumerate(target, 1):
        t0 = time.perf_counter()
        row = process_question(rec)
        rows.append(row)
        gon = row["gate_on"]
        goff = row["gate_off"]
        print(
            "[%s %d/%d] %s | init=%s | ON: %s ref=%s | "
            "fON=%s fOFF=%s sON=%s sOFF=%s (%4.1fs)"
            % (
                run_id, i, len(target), row["question_id"],
                row.get("initial_decision"),
                gon.get("decision"), gon.get("refinement_attempts"),
                (gon.get("faithfulness") if gon.get("faithfulness")
                 is not None else "n/a"),
                (goff.get("faithfulness") if goff.get("faithfulness")
                 is not None else "n/a"),
                (gon.get("evidence_support") if gon.get("evidence_support")
                 is not None else "n/a"),
                (goff.get("evidence_support") if goff.get("evidence_support")
                 is not None else "n/a"),
                time.perf_counter() - t0,
            ),
            flush=True,
        )
        with open(OUT_DIR / ("experiment2_per_question_run%s.json" % run_id),
                  "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    if args.limit is not None:
        print("Diagnostic limit run; skipping summary writeback.")
        return

    summary = build_summary(rows)
    signature = _signature(rows)
    snap_after = snapshot_frozen()

    with open(OUT_DIR / "experiment2_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "experiment2_topic_results.json", "w",
              encoding="utf-8") as f:
        json.dump(summary["topic_level"], f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "experiment2_frozen_hashes.json", "w",
              encoding="utf-8") as f:
        json.dump({
            "run_%s" % run_id: {"before": snap_before, "after": snap_after},
        }, f, indent=2, ensure_ascii=False)
    manifest = {
        "run_id": run_id,
        "benchmark": "data/geri_lit_gold_v1_1.json",
        "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
        "n": len(rows),
        "signature": signature,
        "retrieval_backend": "geri_lit",
        "llm_model": LLM_MODEL,
        "judge_model": JUDGE_MODEL,
        "judge_options": JUDGE_OPTIONS,
        "generation_options": {"temperature": 0, "top_p": 0.1, "top_k": 10},
        "reliability_thresholds": {"accept": 0.80, "refine": 0.65,
                                   "re_retrieve": 0.45},
        "reliability_weights": {"authority": 0.3, "relevance": 0.3,
                                "support": 0.2, "coverage": 0.1,
                                "consistency": 0.1},
        "gate_budgets": {"MAX_REFINE": 1, "MAX_RETRIEVE": 1},
        "runtime_seconds": round(time.perf_counter() - t_all, 2),
        "frozen_unchanged": snap_before == snap_after,
    }
    with open(OUT_DIR / "experiment2_run_manifest.json", "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n=== RUN %s SUMMARY ===" % run_id)
    print("signature:", signature)
    for metric, st in summary["paired_statistics"].items():
        print("  %-18s ON=%s OFF=%s delta=%s" % (
            metric, st.get("on_mean"), st.get("off_mean"),
            st.get("paired_mean_difference")))
    print("gate decisions:", summary["gate_on_decision_counts"])
    print("frozen unchanged:", manifest["frozen_unchanged"])


if __name__ == "__main__":
    main()