#!/usr/bin/env python3
"""Experiment 1 - Final-Stack GeriLit-Gold v1.1 Generation-Quality Evaluation.

Evaluation-only. Uses the CURRENT production stack unchanged:
  - retrieval backend: geri_lit (via ELDERDOCAI_RETRIEVAL_BACKEND env)
  - reliability formula / thresholds: frozen config (read-only)
  - Task 1 programmatic gate (ACCEPT/REFINE/RE-RETRIEVE/REJECT)
  - Task 2 relevance semantics (similarity_score = dense cosine)
  - generation: llama3.2:latest, temperature 0 / top_p 0.1 / top_k 10

Methodology (frozen in experiment1_protocol.md):
  - faithfulness   : FAITHFULNESS_PROMPT (scripts/evaluate_gold_qa.py) +
                     llama3.2 judge + deterministic sampling options
  - answer relevance: RELEVANCE_PROMPT  (scripts/evaluate_gold_qa.py) +
                     llama3.2 judge + deterministic sampling options
  - evidence support: span_token_coverage(answer, evidence_text) [reused]
  - unsupported-claim flag: faithfulness <= 0.50

Writes ONLY under this directory. Never modifies production code, benchmarks,
indexes, or frozen research artifacts.
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

BASE_DIR = Path(__file__).resolve().parents[4]  # ElderDocAI-System
OUT_DIR = Path(__file__).resolve().parent

# ---- Force the frozen GeriLit retrieval backend before any import ----
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
    generate_answer,
    LLM_MODEL,
    REJECTION_RESPONSE,
    EMPTY_EVIDENCE_RESPONSE,
)
from scripts.retrieval_router import retrieve as retrieve_evidence  # noqa: E402
from scripts.reliability_evaluation import evaluate_reliability  # noqa: E402
from scripts.adaptive_decision_controller import (  # noqa: E402
    make_reliability_decision,
)
from scripts.rag_chat import prepare_evidence  # noqa: E402
from scripts.evaluate_gold_qa import (  # noqa: E402
    FAITHFULNESS_PROMPT,
    RELEVANCE_PROMPT,
    build_evidence_text,
    span_token_coverage,
)

JUDGE_MODEL = "llama3.2:latest"
JUDGE_SYSTEM_FAITHFULNESS = "You evaluate RAG faithfulness only."
JUDGE_SYSTEM_RELEVANCE = "You evaluate RAG answer relevance only."
# Deterministic judge sampling - identical options to production generation.
JUDGE_OPTIONS = {"temperature": 0, "top_p": 0.1, "top_k": 10}

BENCHMARK_FILE = BASE_DIR / "data" / "geri_lit_gold_v1_1.json"
EXPECTED_BENCHMARK_SHA256 = (
    "1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72"
)

# Frozen artifacts to hash (pre- and post-run).
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

# Task 3 + Task 10 artifacts to verify (unchanged).
FROZEN_EVAL_FILES = {
    "task10d_results": GERE_DIR / "metadata" / "task10d_retrieval_results.json",
    "task10d_summary": GERE_DIR / "metadata" / "task10d_summary.json",
    "task10e_analysis": GERE_DIR / "metadata" / "task10e_failure_analysis.json",
    "task3_summary": GERE_DIR
    / "metadata"
    / "task3_reliability_evaluation"
    / "task3_summary.json",
}

GENERATION_RETRY_LIMIT = 2
JUDGE_RETRY_LIMIT = 2

TOPIC_CODES = ["C01", "C02", "C03", "C04", "C05",
               "C06", "C07", "C08", "C09", "C10"]


def sha256(path):
    """Return SHA-256 hex digest of a file (or None if missing)."""
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_frozen():
    """Return {name: sha256} for all frozen artifacts."""
    snap = {}
    for name, path in FROZEN_FILES.items():
        snap[name] = sha256(path)
    for name, path in FROZEN_EVAL_FILES.items():
        snap[name] = sha256(path)
    return snap
def judge(question, answer, evidence_items, prompt_template, system_content):
    """LLM judge with the frozen prompt templates and deterministic options.

    Prompt templates are reused verbatim from scripts/evaluate_gold_qa.py.
    Deterministic sampler options are applied explicitly (see protocol 4.1).
    Returns (score or None, reason-string or error-message).
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
def process_question(record):
    """Run one benchmark question through the current production stack.

    Returns a complete per-question record (protocol section 5).
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
        "retrieved_evidence_ids": [],
        "retrieved_evidence_texts": [],
        "retrieval_backend": "geri_lit",
        "evidence_count": 0,
        "initial_reliability": None,
        "initial_decision": None,
        "reliability": None,
        "decision": None,
        "retrieval_attempts": 0,
        "refinement_attempts": 0,
        "refused": False,
        "generated": False,
        "answer": "",
        "latency_seconds": None,
        "faithfulness": None,
        "faithfulness_reason": "",
        "answer_relevance": None,
        "answer_relevance_reason": "",
        "evidence_support": None,
        "hallucination": None,
        "gold_chunk_retrieved": False,
        "error": None,
    }

    # ---- initial (pre-gate) retrieval + reliability (deterministic) ----
    initial_evidence = []
    try:
        initial_evidence = prepare_evidence(
            retrieve_evidence(question))
    except Exception as exc:  # noqa: BLE001
        row["error"] = "initial-retrieval: %s" % type(exc).__name__
    try:
        rel0 = evaluate_reliability(question, initial_evidence)
        dec0 = make_reliability_decision(rel0)
        row["initial_reliability"] = {
            k: round(float(v), 6) for k, v in rel0.items()}
        row["initial_decision"] = dec0["decision"]
    except Exception as exc:  # noqa: BLE001
        row["error"] = "initial-reliability: %s" % type(exc).__name__

    # ---- production gated generation ----
    gen_error = None
    t0 = time.perf_counter()
    result = None
    for attempt in range(GENERATION_RETRY_LIMIT + 1):
        try:
            result = generate_answer(question, return_evaluation=True)
            break
        except Exception as exc:  # noqa: BLE001
            gen_error = "%s: %s" % (type(exc).__name__, exc)
            traceback.print_exc()
            time.sleep(1.0)
    row["latency_seconds"] = round(time.perf_counter() - t0, 3)
    if result is None:
        row["error"] = "generation-failed: %s" % gen_error
        return row

    row["answer"] = str(result.get("answer", "")).strip()
    row["generated"] = bool(result.get("answer"))
    row["refused"] = bool(result.get("refused", False))
    row["retrieval_attempts"] = int(result.get("retrieval_attempts", 0))
    row["refinement_attempts"] = int(result.get("refinement_attempts", 0))
    rel = result.get("reliability") or {}
    row["reliability"] = {k: round(float(v), 6) for k, v in rel.items()}
    dec = result.get("decision") or {}
    row["decision"] = dec.get("decision")
    evidence = result.get("evidence_items") or []
    row["retrieved_evidence_ids"] = [e.get("chunk_id") for e in evidence]
    row["retrieved_evidence_texts"] = [
        e.get("text") or "" for e in evidence]
    row["evidence_count"] = len(evidence)
    row["gold_chunk_retrieved"] = bool(
        set(row["retrieved_evidence_ids"]) & set(gold_chunks))

    # ---- judge: faithfulness ----
    if row["generated"] and row["answer"]:
        score, reason = judge(
            question, row["answer"], evidence,
            FAITHFULNESS_PROMPT, JUDGE_SYSTEM_FAITHFULNESS)
        row["faithfulness"] = score
        row["faithfulness_reason"] = reason[:400]
        # ---- judge: answer relevance ----
        score, reason = judge(
            question, row["answer"], evidence,
            RELEVANCE_PROMPT, JUDGE_SYSTEM_RELEVANCE)
        row["answer_relevance"] = score
        row["answer_relevance_reason"] = reason[:400]
        # ---- deterministic evidence support ----
        evidence_text = "\n".join(row["retrieved_evidence_texts"])
        row["evidence_support"] = span_token_coverage(
            row["answer"], evidence_text)
        row["hallucination"] = bool(
            row["faithfulness"] is not None
            and row["faithfulness"] <= 0.50)
    return row
DESC = {
    "overall_reliability": "reliability",
    "authority": "reliability",
    "relevance": "reliability",
    "support": "reliability",
    "coverage": "reliability",
    "consistency": "reliability",
    "faithfulness": "generation",
    "answer_relevance": "generation",
    "evidence_support": "generation",
    "latency_seconds": "latency",
}


def _stats(rows, key):
    """Compute {n, mean, median, std, min, max} for a numeric field.

    Supports flat row fields and fields nested under the final ``reliability``
    dict (authority/relevance/support/coverage/consistency/overall_reliability).
    """
    vals = []
    for r in rows:
        v = r.get(key)
        if v is None:
            rel = r.get("reliability") or {}
            v = rel.get(key)
        if v is None:
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if not vals:
        return {"n": 0, "mean": None, "median": None, "std": None,
                "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 6),
        "median": round(statistics.median(vals), 6),
        "std": round(statistics.stdev(vals), 6) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 6),
        "max": round(max(vals), 6),
    }


def build_summary(rows):
    """Aggregate the required summary statistics from per-question rows."""
    metrics = [
        "overall_reliability", "authority", "relevance", "support",
        "coverage", "consistency", "faithfulness", "answer_relevance",
        "evidence_support", "latency_seconds",
    ]
    summary = {"n_questions": len(rows)}
    for m in metrics:
        summary[m] = _stats(rows, m)

    # decision distribution (final decision)
    dec_counts = {}
    for r in rows:
        d = r.get("decision") or "MISSING"
        dec_counts[d] = dec_counts.get(d, 0) + 1
    summary["decision_counts"] = dec_counts
    summary["decision_pct"] = {
        d: round(100.0 * c / len(rows), 2)
        for d, c in sorted(dec_counts.items())}

    # initial->final decision transitions
    transitions = {}
    for r in rows:
        key = "%s->%s" % (r.get("initial_decision") or "None",
                          r.get("decision") or "None")
        transitions[key] = transitions.get(key, 0) + 1
    summary["decision_transitions"] = transitions

    # gate behavior
    summary["empty_evidence_cases"] = sum(
        1 for r in rows if r.get("evidence_count", 0) == 0)
    summary["refinement_cases"] = sum(
        1 for r in rows if r.get("refinement_attempts", 0) > 0)
    summary["rere_retrieve_cases"] = sum(
        1 for r in rows if r.get("retrieval_attempts", 0) > 1)
    summary["rejection_cases"] = sum(
        1 for r in rows if r.get("refused", False))
    summary["generation_permitted"] = sum(
        1 for r in rows if r.get("generated", False))
    summary["generation_blocked"] = sum(
        1 for r in rows if not r.get("generated", False))
    summary["errors"] = sum(1 for r in rows if r.get("error"))
    summary["hallucination_flags"] = sum(
        1 for r in rows if r.get("hallucination"))
    summary["contradiction_flags"] = sum(
        1 for r in rows
        if r.get("faithfulness") is not None and r["faithfulness"] == 0.0)

    # evidence stats
    ec = [r.get("evidence_count", 0) for r in rows]
    summary["evidence_count_mean"] = round(
        statistics.mean(ec), 4) if ec else None
    summary["evidence_count_median"] = round(
        statistics.median(ec), 4) if ec else None

    # topic-level
    topics = {}
    for t in TOPIC_CODES:
        sub = [r for r in rows if r.get("topic") == t]
        if not sub:
            continue
        topics[t] = {
            "n": len(sub),
            "mean_reliability": _stats(sub, "overall_reliability")["mean"],
            "mean_relevance_factor": _stats(sub, "relevance")["mean"],
            "mean_faithfulness": _stats(sub, "faithfulness")["mean"],
            "mean_answer_relevance": _stats(sub, "answer_relevance")["mean"],
            "mean_evidence_support": _stats(sub, "evidence_support")["mean"],
            "accepted": sum(1 for r in sub if r.get("decision") == "ACCEPT"),
            "refined": sum(1 for r in sub if r.get("decision") == "REFINE"),
            "re_retrieved": sum(1 for r in sub
                                if r.get("decision") == "RE-RETRIEVE"),
            "rejected": sum(1 for r in sub if r.get("decision") == "REJECT"),
            "hallucination": sum(1 for r in sub if r.get("hallucination")),
        }
    summary["topic_level"] = topics

    # rates
    n_gen = max(1, summary["generation_permitted"])
    n_all = max(1, len(rows))
    summary["unsupported_claim_rate"] = round(
        100.0 * summary["hallucination_flags"] / n_all, 2)
    summary["unsupported_claim_rate_generated"] = round(
        100.0 * summary["hallucination_flags"] / n_gen, 2)
    summary["gold_chunk_retrieved_count"] = sum(
        1 for r in rows if r.get("gold_chunk_retrieved"))
    return summary
def _short_answer(answer):
    """Produce a concise stable answer fingerprint (for reproducibility)."""
    return " ".join(str(answer).strip().split())[:4000]


def _signature(rows):
    """Deterministic signature over substantive per-question fields."""
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda x: x["question_id"]):
        payload = json.dumps(
            {
                "question_id": r["question_id"],
                "retrieved_evidence_ids": r["retrieved_evidence_ids"],
                "reliability": r["reliability"],
                "decision": r["decision"],
                "refinement_attempts": r["refinement_attempts"],
                "retrieval_attempts": r["retrieval_attempts"],
                "refused": r["refused"],
                "generated": r["generated"],
                "answer": _short_answer(r["answer"]),
                "faithfulness": r["faithfulness"],
                "answer_relevance": r["answer_relevance"],
                "evidence_support": r["evidence_support"],
                "hallucination": r["hallucination"],
            },
            sort_keys=True,
        )
        h.update(payload.encode("utf-8"))
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True, help="1 or 2")
    ap.add_argument("--limit", type=int, default=None,
                    help="optional diagnostic limit (default: all 121)")
    args = ap.parse_args()
    run_id = args.run_id
    t_all = time.perf_counter()

    snap_before = snapshot_frozen()

    records = load_benchmark()
    print("Benchmark OK: %d records, hash verified" % len(records))

    target = records if args.limit is None else records[: args.limit]
    print("Evaluating %d questions (run %s)..." % (len(target), run_id))

    rows = []
    for i, rec in enumerate(target, 1):
        t0 = time.perf_counter()
        row = process_question(rec)
        rows.append(row)
        print(
            "[%s %d/%d] %s | init=%s final=%s gen=%s "
            "faith=%s relv=%s support=%s (%4.1fs)"
            % (
                run_id, i, len(target), row["question_id"],
                row.get("initial_decision"), row.get("decision"),
                row.get("generated"), row.get("faithfulness"),
                row.get("answer_relevance"), row.get("evidence_support"),
                time.perf_counter() - t0,
            ),
            flush=True,
        )
        # incremental save so partial results survive crashes
        with open(OUT_DIR / ("experiment1_per_question_run%s.json" % run_id),
                  "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    if args.limit is not None:
        print("Diagnostic limit run; skipping summary/hash writeback.")
        return

    summary = build_summary(rows)
    signature = _signature(rows)
    snap_after = snapshot_frozen()

    with open(OUT_DIR / ("experiment1_summary_run%s.json" % run_id),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(OUT_DIR / "experiment1_frozen_hashes.json", "w",
              encoding="utf-8") as f:
        json.dump(
            {"run_%s" % run_id: {"before": snap_before, "after": snap_after}},
            f, indent=2, ensure_ascii=False)

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
        "generation": {"temperature": 0, "top_p": 0.1, "top_k": 10},
        "reliability_thresholds": {"accept": 0.80, "refine": 0.65,
                                   "re_retrieve": 0.45},
        "reliability_weights": {"authority": 0.3, "relevance": 0.3,
                                "support": 0.2, "coverage": 0.1,
                                "consistency": 0.1},
        "runtime_seconds": round(time.perf_counter() - t_all, 2),
        "frozen_unchanged": snap_before == snap_after,
    }
    with open(OUT_DIR / "experiment1_run_manifest.json", "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n=== RUN %s SUMMARY ===" % run_id)
    print("signature:", signature)
    print("decision_counts:", summary["decision_counts"])
    print("mean reliability:", summary["overall_reliability"]["mean"])
    print("mean faithfulness:", summary["faithfulness"]["mean"])
    print("mean answer_relevance:", summary["answer_relevance"]["mean"])
    print("mean evidence_support:", summary["evidence_support"]["mean"])
    print("hallucination flags:", summary["hallucination_flags"])
    print("frozen unchanged:", manifest["frozen_unchanged"])
    print("runtime (s):", manifest["runtime_seconds"])


if __name__ == "__main__":
    main()