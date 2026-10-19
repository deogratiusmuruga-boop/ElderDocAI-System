#!/usr/bin/env python3
"""Experiment 5 - Generator-Model Evaluation on GeriLit-Gold v1.1 (generator swap only).

Evaluation-only. Uses the CURRENT production stack unchanged:
  - retrieval backend: geri_lit (via ELDERDOCAI_RETRIEVAL_BACKEND env)
  - reliability formula / thresholds: frozen config (read-only)
  - Task 1 programmatic gate (ACCEPT/REFINE/RE-RETRIEVE/REJECT)
  - Task 2 relevance semantics (similarity_score = dense cosine)
  - generation: model given by --generator (required), temperature 0 / top_p 0.1 / top_k 10

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
import tempfile
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

import scripts.rag_chat as _rag_chat_module  # noqa: E402
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

# ---------------- Experiment 5 additions (generator swap only) ----------------
EXPECTED_OUT_DIR_NAME = "experiment5_generator_models"
GENERATOR = None  # set from --generator in main()
_GEN_CALLS = []  # non-judge ollama.chat calls made for the current question
_ORIGINAL_CHAT = ollama.chat


def _resp_field(resp, name):
    """Read a field from an Ollama response (attribute or dict style)."""
    value = getattr(resp, name, None)
    if value is None:
        try:
            value = resp.get(name)
        except Exception:  # noqa: BLE001
            value = None
    return value


def _recording_chat(*args, **kwargs):
    """Call the real ollama.chat unchanged; record token counts of generator calls."""
    response = _ORIGINAL_CHAT(*args, **kwargs)
    if kwargs.get("format") != "json":  # judge calls use format="json"
        _GEN_CALLS.append({
            "model": kwargs.get("model"),
            "prompt_eval_count": _resp_field(response, "prompt_eval_count"),
            "eval_count": _resp_field(response, "eval_count"),
            "total_duration_ns": _resp_field(response, "total_duration"),
        })
    return response


def _check_out_dir():
    """Abort unless outputs go to the Experiment 5 folder (never Experiments 1-4)."""
    if OUT_DIR.name != EXPECTED_OUT_DIR_NAME:
        raise SystemExit("OUT_DIR is not the Experiment 5 folder: %s" % OUT_DIR)
    bad = ("experiment1", "experiment2", "experiment3", "experiment4")
    if any(p.lower().startswith(bad) for p in OUT_DIR.parts):
        raise SystemExit("OUT_DIR points into an Experiment 1-4 folder: %s" % OUT_DIR)


def _install_generator(model):
    """Point the production generation call at `model` without editing rag_chat.py."""
    ollama.show(model)  # fail fast if the model is not installed
    _rag_chat_module.LLM_MODEL = model
    if _rag_chat_module.LLM_MODEL != model:
        raise SystemExit("Generator override failed")
    ollama.chat = _recording_chat


def _model_info(model):
    """Best-effort record of model identity plus client and server versions."""
    info = {"model_tag": model}
    try:
        listing = ollama.list()
        models = getattr(listing, "models", None) or listing.get("models", [])
        for m in models:
            name = getattr(m, "model", None) or getattr(m, "name", None)
            if name in (model, model + ":latest"):
                info["listed_as"] = name
                info["digest"] = getattr(m, "digest", None)
                info["size_bytes"] = getattr(m, "size", None)
                d = getattr(m, "details", None)
                if d is not None:
                    info["details"] = {
                        k: getattr(d, k, None)
                        for k in ("format", "family", "parameter_size",
                                  "quantization_level")}
    except Exception as exc:  # noqa: BLE001
        info["list_error"] = "%s: %s" % (type(exc).__name__, exc)
    try:
        import importlib.metadata as _md
        info["ollama_client_version"] = _md.version("ollama")
    except Exception:  # noqa: BLE001
        info["ollama_client_version"] = None
    try:
        import urllib.request
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        if not host.startswith("http"):
            host = "http://" + host
        with urllib.request.urlopen(
                host.rstrip("/") + "/api/version", timeout=5) as r:
            info["ollama_server_version"] = json.loads(
                r.read().decode("utf-8")).get("version")
    except Exception as exc:  # noqa: BLE001
        info["ollama_server_version"] = "unavailable: %s" % type(exc).__name__
    return info

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
    """Return {name: sha256} for all required frozen artifacts.

    Aborts immediately if any artifact is missing (None) or if an
    exception is raised while opening/reading/hashing it."""
    snap = {}
    for name, path in [*FROZEN_FILES.items(), *FROZEN_EVAL_FILES.items()]:
        try:
            h = sha256(path)
        except Exception as exc:
            raise SystemExit(
                "FROZEN ARTIFACT READ/HASH FAILED: %s -> %s   (%s)"
                % (name, path, exc)) from exc
        if h is None:
            raise SystemExit(
                "FROZEN ARTIFACT MISSING FROM DISK: %s -> %s" % (name, path))
        snap[name] = h
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
    del _GEN_CALLS[:]
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
    row["generator_model"] = GENERATOR
    row["gen_calls"] = list(_GEN_CALLS)
    _pec = [c["prompt_eval_count"] for c in _GEN_CALLS
            if c.get("prompt_eval_count") is not None]
    row["prompt_eval_count_max"] = max(_pec) if _pec else None
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
    ap.add_argument("--generator", required=True,
                    help="Ollama tag of the generator model (Experiment 5)")
    ap.add_argument("--run-id", required=True, help="1 or 2")
    ap.add_argument("--limit", type=int, default=None,
                    help="optional diagnostic limit (default: all 121)")
    args = ap.parse_args()
    run_id = args.run_id
    t_all = time.perf_counter()

    # ---- Experiment 5 setup: safety checks, generator injection, file names ----
    global GENERATOR
    GENERATOR = args.generator
    _check_out_dir()
    tag = GENERATOR.replace(":", "_").replace("/", "_")
    diag = args.limit is not None
    if diag:
        pq_name = "experiment5_%s_diag_limit%d_run%s.json" % (
            tag, args.limit, run_id)
    else:
        pq_name = "experiment5_%s_per_question_run%s.json" % (tag, run_id)
    names = {
        "per_question": pq_name,
        "summary": "experiment5_%s_summary_run%s.json" % (tag, run_id),
        "hashes": (
            "experiment5_%s_diag_frozen_hashes_limit%d_run%s.json"
            % (tag, args.limit, run_id)
            if diag else
            "experiment5_%s_frozen_hashes_run%s.json" % (tag, run_id)
        ),
        "manifest": "experiment5_%s_run_manifest_run%s.json" % (tag, run_id),
    }
    for fname in names.values():
        if (OUT_DIR / fname).exists():
            raise SystemExit("Refusing to overwrite: %s" % fname)
    model_info = _model_info(GENERATOR)
    _install_generator(GENERATOR)
    print("Generator: %s | Ollama server: %s" % (
        GENERATOR, model_info.get("ollama_server_version")), flush=True)

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
        with open(OUT_DIR / names["per_question"],
                  "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    _no_calls = sum(
        1 for r in rows if r.get("generated") and not r.get("gen_calls"))
    if _no_calls:
        print("WARNING: %d generated answers have no recorded generator call "
              "(token-count recording not effective)" % _no_calls, flush=True)
    if args.limit is not None:
        snap_after = snapshot_frozen()
        if snap_before != snap_after:
            changed = sorted(
                k for k in (set(snap_before) | set(snap_after))
                if snap_before.get(k) != snap_after.get(k)
            )
            raise SystemExit(
                "FROZEN ARTIFACT(S) CHANGED DURING DIAGNOSTIC RUN: %s"
                % (", ".join(changed) if changed else "unknown"))
        # Write completed hash report atomically via temp file.
        # The .exists() guard below prevents accidental overwrites by
        # ordinary re-runs and is guaranteed by Fix 4 (no output file
        # exists at process start).  It does NOT protect against a
        # concurrent process creating the destination between the check
        # and os.replace; this script assumes single-process operation.
        hashes_path = OUT_DIR / names["hashes"]
        if hashes_path.exists():
            raise SystemExit(
                "Refusing to overwrite existing diagnostic hash file: %s"
                % hashes_path)
        # Use tempfile to create a unique temporary file in OUT_DIR.
        # A stale temporary file from an interrupted run is never truncated.
        fd, tmp_path = tempfile.mkstemp(
            dir=str(OUT_DIR), suffix=hashes_path.suffix)
        os.close(fd)
        tmp_path = Path(tmp_path)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"run_%s_diag" % run_id: {"before": snap_before,
                                               "after": snap_after}},
                    f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, hashes_path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise
        print("Diagnostic limit run; integrity verified, hashes saved.")
        return

    summary = build_summary(rows)
    summary["generator_model"] = GENERATOR
    summary["not_found_answers_heuristic"] = sum(
        1 for r in rows
        if "find that information" in (r.get("answer") or "").lower())
    _pec = [r["prompt_eval_count_max"] for r in rows
            if r.get("prompt_eval_count_max") is not None]
    summary["prompt_eval_count"] = {
        "n": len(_pec),
        "max": max(_pec) if _pec else None,
        "mean": round(statistics.mean(_pec), 1) if _pec else None,
    }
    signature = _signature(rows)
    snap_after = snapshot_frozen()

    if snap_before != snap_after:
        changed = sorted(
            k for k in (set(snap_before) | set(snap_after))
            if snap_before.get(k) != snap_after.get(k)
        )
        raise SystemExit(
            "FROZEN ARTIFACT(S) CHANGED DURING RUN: %s"
            % (", ".join(changed) if changed else "unknown"))

    with open(OUT_DIR / names["summary"],
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(OUT_DIR / names["hashes"], "w",
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
        "llm_model": GENERATOR,
        "experiment": 5,
        "model_info": model_info,
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
    with open(OUT_DIR / names["manifest"], "w",
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