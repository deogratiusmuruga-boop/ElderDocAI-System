#!/usr/bin/env python3
"""Experiment 3 - Dynamic Care-State and Adaptive-Assistance Evaluation.

Paired within-subject comparison of the SAME question under two REAL
(non-synthetic) dynamic care states (STABLE vs HIGH_ACTIVITY) computed by
scripts/care_state.py and rendered through the production prompt.

Controls (protocol section 3): question, retrieval, LLM, reliability config,
prompt template, and user profile are identical across the two conditions.
Only the real care-state signals differ (DB medications, appointments,
conversation history).

Evaluation reuses Experiment 1/2 methodology: FAITHFULNESS_PROMPT /
RELEVANCE_PROMPT judges (deterministic sampler) + span_token_coverage, plus a
frozen appropriateness judge comparing answer_B vs answer_A.

All frozen artifacts read-only. Writes ONLY under this directory.
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

BASE_DIR = Path(__file__).resolve().parents[4]
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
    generate_answer,
)
from scripts.care_state import compute_care_state  # noqa: E402
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
    "exp2_summary": GERE_DIR
    / "metadata"
    / "experiment2_gating_effectiveness"
    / "experiment2_summary.json",
}

GENERATION_RETRY_LIMIT = 2
JUDGE_RETRY_LIMIT = 2
TOPIC_CODES = ["C01", "C02", "C03", "C04", "C05",
               "C06", "C07", "C08", "C09", "C10"]
QUESTIONS_PER_TOPIC = 2
NOW = "2026-09-22"  # fixed reference date for determinism

# Real signals for the two paired conditions (identical profile).
PROFILE_STABLE = {
    "age": 72,
    "chronic_conditions": ["diabetes", "hypertension", "copd"],
    "medications": [],
    "location": "Seoul",
    "preferred_language": "en",
    "speech_speed": "normal",
}
MEDS_HIGH = [
    {"medicine_name": "metformin", "dosage": "500mg", "time": "08:00",
     "frequency": "daily"},
    {"medicine_name": "lisinopril", "dosage": "20mg", "time": "08:00",
     "frequency": "daily"},
    {"medicine_name": "glimepiride", "dosage": "1mg", "time": "08:00",
     "frequency": "daily"},
    {"medicine_name": "atorvastatin", "dosage": "40mg", "time": "21:00",
     "frequency": "daily"},
    {"medicine_name": "aspirin", "dosage": "100mg", "time": "09:00",
     "frequency": "daily"},
    {"medicine_name": "gabapentin", "dosage": "300mg", "time": "21:00",
     "frequency": "daily"},
    {"medicine_name": "prednisone", "dosage": "5mg", "time": "08:00",
     "frequency": "daily"},
]
APPOINTMENTS_HIGH = [
    {"title": "Primary care visit", "appointment_date": "2026-09-28",
     "location": "Clinic"},
    {"title": "Lab follow-up", "appointment_date": "2026-10-02",
     "location": "Lab"},
    {"title": "Medication review", "appointment_date": "2026-10-07",
     "location": "Pharmacy"},
]
HISTORY_HIGH = [
    {"role": "user", "content": "I have been feeling unsteady"},
    {"role": "assistant", "content": "Please be careful and stay seated"},
    {"role": "user", "content": "Should I call someone?"},
    {"role": "assistant", "content": "You may want to inform your doctor"},
]
APPROPRIATENESS_PROMPT = """You evaluate whether an assistance answer
appropriately reflects a HIGHER dynamic care state relative to a LOWER state,
using only the two answers provided. You do NOT evaluate factual correctness,
medical quality, or faithfulness.

SCENARIO
- state_A is the LOWER care-need state (e.g. STABLE).
- state_B is the HIGHER care-need state (e.g. HIGH_ACTIVITY).
- answer_A was generated under state_A.
- answer_B was generated under state_B.
- The question, evidence, and model settings were identical.

TASK
Determine whether answer_B appropriately reflects the higher care need
compared with answer_A, e.g. by being more attentive to safety, more
supportive/monitoring-oriented, more proactive about follow-up, or more
cautious - and only when that difference is consistent with the described
state difference.

RULE
Return 1 only if answer_B shows an appropriate higher-attention response
relative to answer_A and the difference is NOT arbitrary. Return 0 if the
answers are essentially the same, or if answer_B differs in an arbitrary or
inappropriate way.

QUESTION
{question}

STATE_A:
{state_A}

STATE_B:
{state_B}

ANSWER_A:
{answer_A}

ANSWER_B:
{answer_B}

Return ONLY JSON:
{{"appropriate": 0, "reason": "short explanation"}}
"""


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


def load_benchmark_and_select():
    """Load frozen v1.1 benchmark and select the experiment question set.

    Deterministic selection protocol (protocol section 5): first
    QUESTIONS_PER_TOPIC records per topic by final_benchmark_id order.
    The benchmark is not modified. Returns the 20 selected records and the
    full topic count map.
    """
    actual = sha256(BENCHMARK_FILE)
    if actual != EXPECTED_BENCHMARK_SHA256:
        raise SystemExit(
            "BENCHMARK HASH MISMATCH: expected %s actual %s"
            % (EXPECTED_BENCHMARK_SHA256, actual))
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = sorted(data["records"], key=lambda r: r["final_benchmark_id"])
    selected = []
    seen = {t: 0 for t in TOPIC_CODES}
    for r in records:
        topic = r.get("topic")
        if topic in TOPIC_CODES and seen[topic] < QUESTIONS_PER_TOPIC:
            selected.append(r)
            seen[topic] += 1
    return selected, seen


def judge(question, answer, evidence_items, prompt_template, system_content):
    """LLM judge - methodology identical to Experiments 1-2."""
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


def appropriateness_judge(question, state_a, state_b, answer_a, answer_b):
    """Frozen appropriateness judge for the paired answers."""
    prompt = APPROPRIATENESS_PROMPT.format(
        question=question, state_A=state_a, state_B=state_b,
        answer_A=answer_a, answer_B=answer_b)
    last_error = None
    for attempt in range(JUDGE_RETRY_LIMIT + 1):
        try:
            response = ollama.chat(
                model=JUDGE_MODEL,
                format="json",
                options=JUDGE_OPTIONS,
                messages=[
                    {"role": "system",
                     "content": "You evaluate assistance-state appropriateness."},
                    {"role": "user", "content": prompt},
                ],
            )
            parsed = json.loads(response["message"]["content"])
            return int(parsed.get("appropriate", 0) or 0), str(
                parsed.get("reason", ""))[:300]
        except Exception as exc:  # noqa: BLE001
            last_error = "%s: %s" % (type(exc).__name__, exc)
            time.sleep(0.5)
    return None, "appropriateness-error: %s" % last_error
def _retry_call(fn, retries=GENERATION_RETRY_LIMIT, sleep=1.0):
    last_error = None
    for attempt in range(retries + 1):
        try:
            return fn(), None
        except Exception as exc:  # noqa: BLE001
            last_error = "%s: %s" % (type(exc).__name__, exc)
            traceback.print_exc()
            time.sleep(sleep)
    return None, last_error


def _condition_record(question, care_record, gold_chunks):
    """Run one condition through the production generation path."""
    out = {
        "adaptive_context": care_record,
        "state": care_record["care_state"]["state"],
        "overall_score": care_record["care_state"]["overall_score"],
        "evidence_ids": [], "reliability": None, "decision": None,
        "refinement_attempts": 0, "retrieval_attempts": 1,
        "refused": False, "generated": False, "answer": "",
        "gold_chunk_retrieved": False, "faithfulness": None,
        "faithfulness_reason": "", "answer_relevance": None,
        "answer_relevance_reason": "", "evidence_support": None,
        "latency_seconds": None, "error": None,
    }
    t0 = time.perf_counter()
    res, err = _retry_call(lambda: generate_answer(
        question, user_profile=PROFILE_STABLE,
        return_evaluation=True,
        adaptive_context=dict(care_record)))
    out["latency_seconds"] = round(time.perf_counter() - t0, 3)
    if res is None:
        out["error"] = "generation-failed: %s" % err
        return out
    out["answer"] = str(res.get("answer", "")).strip()
    out["generated"] = bool(out["answer"])
    out["refused"] = bool(res.get("refused", False))
    out["refinement_attempts"] = int(res.get("refinement_attempts", 0))
    out["retrieval_attempts"] = int(res.get("retrieval_attempts", 0))
    rel = res.get("reliability") or {}
    out["reliability"] = {k: round(float(v), 6) for k, v in rel.items()}
    out["decision"] = (res.get("decision") or {}).get("decision")
    evidence = res.get("evidence_items") or []
    out["evidence_ids"] = [e.get("chunk_id") for e in evidence]
    out["gold_chunk_retrieved"] = bool(
        set(out["evidence_ids"]) & set(gold_chunks or []))
    if out["generated"] and out["answer"]:
        score, reason = judge(question, out["answer"], evidence,
                              FAITHFULNESS_PROMPT, JUDGE_SYSTEM_FAITHFULNESS)
        out["faithfulness"] = score
        out["faithfulness_reason"] = reason[:400]
        score, reason = judge(question, out["answer"], evidence,
                              RELEVANCE_PROMPT, JUDGE_SYSTEM_RELEVANCE)
        out["answer_relevance"] = score
        out["answer_relevance_reason"] = reason[:400]
        out["evidence_support"] = span_token_coverage(
            out["answer"], "\n".join(e.get("text") or "" for e in evidence))
    return out


def process_pair(record):
    """One paired observation: STABLE (A) vs HIGH_ACTIVITY (B)."""
    question = record["question"]
    gold = record.get("gold_relevant_chunk_ids") or []
    care_a = compute_care_state(
        profile=PROFILE_STABLE, medications=[],
        appointments=[], conversation_history=[],
        now=NOW, patient_id=record["final_benchmark_id"])
    care_b = compute_care_state(
        profile=PROFILE_STABLE, medications=MEDS_HIGH,
        appointments=APPOINTMENTS_HIGH, conversation_history=HISTORY_HIGH,
        previous_state=care_a["care_state"]["state"],
        previous_score=care_a["care_state"]["overall_score"],
        now=NOW, patient_id=record["final_benchmark_id"])

    row = {
        "pair_id": record["final_benchmark_id"],
        "question_id": record["final_benchmark_id"],
        "topic": record.get("topic"),
        "question": question,
        "gold_pmcid": record.get("pmcid"),
        "gold_chunk_ids": gold,
        "state_A": care_a["care_state"]["state"],
        "state_B": care_b["care_state"]["state"],
        "state_difference": "%s -> %s" % (
            care_a["care_state"]["state"], care_b["care_state"]["state"]),
        "score_A": care_a["care_state"]["overall_score"],
        "score_B": care_b["care_state"]["overall_score"],
        "transition_type": care_b["transition"]["type"],
        "transition_direction": care_b["transition"]["direction"],
        "condition_A": _condition_record(question, care_a, gold),
        "condition_B": _condition_record(question, care_b, gold),
        "adaptation_judgment": None,
        "appropriateness_judgment": None,
        "paired_differences": {
            "faithfulness_delta": None,
            "answer_relevance_delta": None,
            "evidence_support_delta": None,
        },
    }

    a = row["condition_A"]
    b = row["condition_B"]
    # adaptation judgment: whether answers differ (system behavior)
    row["adaptation_judgment"] = bool(a.get("answer") != b.get("answer"))
    for metric, key in [
        ("faithfulness_delta", "faithfulness"),
        ("answer_relevance_delta", "answer_relevance"),
        ("evidence_support_delta", "evidence_support"),
    ]:
        x, y = a.get(key), b.get(key)
        if x is not None and y is not None:
            row["paired_differences"][metric] = round(float(y) - float(x), 6)

    if (a.get("generated") and a.get("answer")
            and b.get("generated") and b.get("answer")):
        appr, reason = appropriateness_judge(
            question, row["state_A"], row["state_B"],
            a["answer"], b["answer"])
        row["appropriateness_judgment"] = appr
        row["appropriateness_reason"] = reason
    return row
def _stats_on(rows, cond, key):
    vals = []
    for r in rows:
        v = r[cond].get(key)
        if v is None:
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if not vals:
        return {"n": 0, "mean": None, "median": None, "std": None,
                "min": None, "max": None}
    return {"n": len(vals),
            "mean": round(statistics.mean(vals), 6),
            "median": round(statistics.median(vals), 6),
            "std": round(statistics.stdev(vals), 6) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 6), "max": round(max(vals), 6)}


def _delta_stats(rows, key):
    vals = [r["paired_differences"][key] for r in rows
            if r["paired_differences"].get(key) is not None]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "gt": 0, "eq": 0,
                "lt": 0}
    return {"n": len(vals),
            "mean": round(statistics.mean(vals), 6),
            "median": round(statistics.median(vals), 6),
            "gt": sum(1 for v in vals if v > 1e-9),
            "eq": sum(1 for v in vals if abs(v) <= 1e-9),
            "lt": sum(1 for v in vals if v < -1e-9)}


def build_summary(rows):
    summary = {"n_pairs": len(rows)}
    summary["condition_stats"] = {
        "A": {k: _stats_on(rows, "condition_A", k)
              for k in ["faithfulness", "answer_relevance",
                        "evidence_support"]},
        "B": {k: _stats_on(rows, "condition_B", k)
              for k in ["faithfulness", "answer_relevance",
                        "evidence_support"]},
    }
    summary["paired_deltas"] = {
        k: _delta_stats(rows, k) for k in
        ["faithfulness_delta", "answer_relevance_delta",
         "evidence_support_delta"]}
    summary["state_response_rate"] = round(
        100.0 * sum(1 for r in rows if r["adaptation_judgment"])
        / len(rows), 2)
    summary["state_response_count"] = sum(
        1 for r in rows if r["adaptation_judgment"])
    summary["evidence_identical_rate"] = round(
        100.0 * sum(1 for r in rows
                    if r["condition_A"]["evidence_ids"]
                    == r["condition_B"]["evidence_ids"]) / len(rows), 2)
    grounded_both = sum(
        1 for r in rows
        if (r["condition_A"]["faithfulness"] is not None
            and r["condition_A"]["faithfulness"] >= 0.75
            and r["condition_B"]["faithfulness"] is not None
            and r["condition_B"]["faithfulness"] >= 0.75))
    summary["groundedness_rate"] = round(
        100.0 * grounded_both / len(rows), 2)
    appr = [r["appropriateness_judgment"] for r in rows
            if r["appropriateness_judgment"] is not None]
    summary["appropriateness_rate"] = (
        round(100.0 * sum(appr) / len(appr), 2) if appr else None)
    summary["appropriateness_n"] = len(appr)
    summary["gold_in_evidence"] = {
        "A": sum(1 for r in rows if r["condition_A"]["gold_chunk_retrieved"]),
        "B": sum(1 for r in rows if r["condition_B"]["gold_chunk_retrieved"]),
    }
    summary["transition"] = {
        "escalation": sum(1 for r in rows
                          if r["transition_type"] == "ESCALATION"),
        "up_direction": sum(1 for r in rows
                            if r["transition_direction"] == "UP"),
    }
    summary["errors"] = sum(
        1 for r in rows
        if r["condition_A"]["error"] or r["condition_B"]["error"])
    topics = {}
    for t in sorted({r["topic"] for r in rows}):
        sub = [r for r in rows if r["topic"] == t]
        topics[t] = {
            "n": len(sub),
            "state_response": sum(1 for r in sub
                                  if r["adaptation_judgment"]),
            "faithfulness_delta_mean": _delta_stats(
                sub, "faithfulness_delta")["mean"],
            "relevance_delta_mean": _delta_stats(
                sub, "answer_relevance_delta")["mean"],
            "support_delta_mean": _delta_stats(
                sub, "evidence_support_delta")["mean"],
        }
    summary["topic_level"] = topics
    return summary
def _signature(rows):
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda x: x["pair_id"]):
        payload = {
            "pair_id": r["pair_id"],
            "state_A": r["state_A"], "state_B": r["state_B"],
            "score_A": r["score_A"], "score_B": r["score_B"],
            "A": {k: r["condition_A"][k] for k in
                  ["evidence_ids", "reliability", "decision",
                   "refinement_attempts", "refused", "generated", "answer",
                   "faithfulness", "answer_relevance", "evidence_support"]},
            "B": {k: r["condition_B"][k] for k in
                  ["evidence_ids", "reliability", "decision",
                   "refinement_attempts", "refused", "generated", "answer",
                   "faithfulness", "answer_relevance", "evidence_support"]},
            "adaptation": r["adaptation_judgment"],
            "appropriateness": r["appropriateness_judgment"],
        }
        h.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run_id = args.run_id
    t_all = time.perf_counter()

    snap_before = snapshot_frozen()
    records, seen = load_benchmark_and_select()
    print("Selected %d questions; topic counts: %s"
          % (len(records), seen))
    target = records if args.limit is None else records[:args.limit]
    print("Running paired conditions for %d questions (run %s)..."
          % (len(target), run_id))

    rows = []
    for i, rec in enumerate(target, 1):
        t0 = time.perf_counter()
        row = process_pair(rec)
        rows.append(row)
        a, b = row["condition_A"], row["condition_B"]
        print("[%s %d/%d] %s | %s->%s | adapted=%s appr=%s "
              "fA=%s fB=%s (%.0fs)"
              % (run_id, i, len(target), row["pair_id"],
                 row["state_A"], row["state_B"],
                 row["adaptation_judgment"],
                 row["appropriateness_judgment"],
                 a.get("faithfulness"), b.get("faithfulness"),
                 time.perf_counter() - t0), flush=True)
        with open(OUT_DIR / ("experiment3_per_question_run%s.json" % run_id),
                  "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    if args.limit is not None:
        print("Diagnostic run; skipped summary writeback.")
        return

    summary = build_summary(rows)
    signature = _signature(rows)
    snap_after = snapshot_frozen()
    with open(OUT_DIR / "experiment3_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "experiment3_topic_results.json", "w",
              encoding="utf-8") as f:
        json.dump(summary["topic_level"], f, indent=2, ensure_ascii=False)
    hash_path = OUT_DIR / "experiment3_frozen_hashes.json"
    hashes = {}
    if hash_path.exists():
        with open(hash_path, "r", encoding="utf-8") as f:
            hashes = json.load(f)
    hashes["run_%s" % run_id] = {
        "before": snap_before, "after": snap_after,
        "note": "frozen artifacts are read-only; before == after."}
    with open(hash_path, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, ensure_ascii=False)
    manifest = {
        "run_id": run_id,
        "n": len(rows),
        "signature": signature,
        "question_set": "first 2 per topic, frozen v1.1 (read-only)",
        "retrieval_backend": "geri_lit",
        "llm_model": LLM_MODEL,
        "judge_model": JUDGE_MODEL,
        "judge_options": JUDGE_OPTIONS,
        "now": NOW,
        "states": {"A": "STABLE", "B": "HIGH_ACTIVITY"},
        "runtime_seconds": round(time.perf_counter() - t_all, 2),
        "frozen_unchanged": snap_before == snap_after,
    }
    with open(OUT_DIR / "experiment3_run_manifest.json", "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=== RUN %s SUMMARY ===" % run_id)
    print("signature:", signature)
    print("state_response_rate:", summary["state_response_rate"])
    print("evidence_identical_rate:", summary["evidence_identical_rate"])
    print("groundedness_rate:", summary["groundedness_rate"])
    print("appropriateness_rate:", summary["appropriateness_rate"])
    print("frozen_unchanged:", manifest["frozen_unchanged"])


if __name__ == "__main__":
    main()