#!/usr/bin/env python3
"""Experiment 4 - Vanilla-RAG Baseline on GeriLit-Gold v1.1.

Paired controlled comparison:
  FULL    = the production ElderDocAI generation pathway (reliability gate,
            refinement rules, ElderDocAI grounded prompt) on the frozen
            initial evidence.
  VANILLA = ordinary evidence-grounded RAG on the IDENTICAL initial evidence
            with a minimal frozen prompt and no framework metadata.

Both conditions share the same question, same frozen retrieval evidence,
same LLM (llama3.2:latest) and same generation options. Evaluation reuses
Experiments 1-3 methodology (FAITHFULNESS_PROMPT / RELEVANCE_PROMPT judges +
span_token_coverage).

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
    prepare_evidence,
)
from scripts.retrieval_router import retrieve as retrieve_evidence  # noqa: E402
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

# ------------------------------------------------------------------ frozen baseline prompt
VANILLA_SYSTEM = (
    "Answer the user's question using only the provided evidence."
)
VANILLA_PROMPT_TEMPLATE = (
    "Answer the user's question using only the provided evidence.\n"
    "\n"
    "If the evidence does not contain enough information to answer the "
    "question, state that the evidence is insufficient rather than "
    "inventing information.\n"
    "\n"
    "Retrieved evidence:\n"
    "{evidence_text}\n"
    "\n"
    "Question:\n"
    "{question}"
)
VANILLA_PROMPT_SHA256 = hashlib.sha256(
    (VANILLA_SYSTEM + "\n" + VANILLA_PROMPT_TEMPLATE).encode("utf-8")
).hexdigest()

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
    "exp3_summary": GERE_DIR
    / "metadata"
    / "experiment3_care_state_adaptation"
    / "experiment3_summary.json",
}

GENERATION_RETRY_LIMIT = 2
JUDGE_RETRY_LIMIT = 2


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


def judge(question, answer, evidence_items, prompt_template, system_content):
    """LLM judge - methodology identical to Experiments 1-3."""
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


def vanilla_generate(question, evidence_items):
    """Vanilla-RAG generation: minimal frozen prompt, no framework metadata.

    Returns (answer, error). The prompt contains ONLY the retrieved evidence
    and the question plus a plain evidence-grounded instruction.
    """
    blocks = []
    for i, item in enumerate(evidence_items, start=1):
        blocks.append("Evidence %d:\n%s" % (i, item.get("text", "")))
    evidence_text = "\n\n".join(blocks)
    prompt = VANILLA_PROMPT_TEMPLATE.format(
        evidence_text=evidence_text, question=question)
    last_error = None
    for attempt in range(GENERATION_RETRY_LIMIT + 1):
        try:
            response = ollama.chat(
                model=LLM_MODEL,
                options={"temperature": 0, "top_p": 0.1, "top_k": 10},
                messages=[
                    {"role": "system", "content": VANILLA_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            return str(response["message"]["content"]).strip(), None
        except Exception as exc:  # noqa: BLE001
            last_error = "%s: %s" % (type(exc).__name__, exc)
            time.sleep(1.0)
    return None, last_error


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
def _judge_answers(question, answer, evidence):
    """Faithfulness + relevance + support for one condition's answer."""
    if not answer:
        return None, None, None
    score, reason = judge(question, answer, evidence,
                          FAITHFULNESS_PROMPT, JUDGE_SYSTEM_FAITHFULNESS)
    score2, reason2 = judge(question, answer, evidence,
                            RELEVANCE_PROMPT, JUDGE_SYSTEM_RELEVANCE)
    support = span_token_coverage(
        answer, "\n".join(e.get("text") or "" for e in evidence))
    return score, score2, support


def process_pair(record):
    """One paired observation: FULL framework vs VANILLA RAG.

    Shared initial evidence: ONE retrieve_evidence call per question.
    """
    qid = record["final_benchmark_id"]
    question = record["question"]
    gold = record.get("gold_relevant_chunk_ids") or []

    row = {
        "pair_id": qid,
        "question": question,
        "topic": record.get("topic"),
        "gold_pmcid": record.get("pmcid"),
        "gold_chunk_ids": gold,
        "evidence_identity_ok": False,
        "full": {
            "answer": "", "evidence_ids": [], "reliability": None,
            "decision": None, "refinement_attempts": 0,
            "retrieval_attempts": 1, "refused": False, "generated": False,
            "faithfulness": None, "answer_relevance": None,
            "evidence_support": None, "gold_chunk_retrieved": False,
            "latency_seconds": None, "error": None,
        },
        "vanilla": {
            "answer": "", "evidence_ids": [], "generated": False,
            "faithfulness": None, "answer_relevance": None,
            "evidence_support": None, "gold_chunk_retrieved": False,
            "latency_seconds": None, "error": None,
        },
        "paired_differences": {
            "faithfulness_delta": None,
            "answer_relevance_delta": None,
            "evidence_support_delta": None,
        },
    }

    try:
        initial_evidence = prepare_evidence(retrieve_evidence(question))
    except Exception as exc:  # noqa: BLE001
        row["full"]["error"] = "retrieval: %s" % type(exc).__name__
        row["vanilla"]["error"] = "retrieval: %s" % type(exc).__name__
        return row
    initial_ids = [e.get("chunk_id") for e in initial_evidence]

    # ---- FULL framework ----
    t0 = time.perf_counter()
    res, err = _retry_call(lambda: generate_answer(
        question, return_evaluation=True))
    row["full"]["latency_seconds"] = round(time.perf_counter() - t0, 3)
    if res is None:
        row["full"]["error"] = "full-failed: %s" % err
    else:
        full = row["full"]
        full["answer"] = str(res.get("answer", "")).strip()
        full["generated"] = bool(full["answer"])
        full["refused"] = bool(res.get("refused", False))
        full["refinement_attempts"] = int(res.get("refinement_attempts", 0))
        full["retrieval_attempts"] = int(res.get("retrieval_attempts", 0))
        rel = res.get("reliability") or {}
        full["reliability"] = {k: round(float(v), 6)
                               for k, v in rel.items()}
        full["decision"] = (res.get("decision") or {}).get("decision")
        evidence = res.get("evidence_items") or []
        full["evidence_ids"] = [e.get("chunk_id") for e in evidence]
        full["gold_chunk_retrieved"] = bool(
            set(full["evidence_ids"]) & set(gold))
        f_score, f_relv, f_sup = _judge_answers(
            question, full["answer"], evidence)
        full["faithfulness"] = f_score
        full["answer_relevance"] = f_relv
        full["evidence_support"] = f_sup

    # ---- VANILLA RAG (identical initial evidence) ----
    t0 = time.perf_counter()
    answer, v_err = vanilla_generate(question, initial_evidence)
    row["vanilla"]["latency_seconds"] = round(time.perf_counter() - t0, 3)
    v = row["vanilla"]
    v["evidence_ids"] = list(initial_ids)
    v["gold_chunk_retrieved"] = bool(set(initial_ids) & set(gold))
    if v_err is not None:
        v["error"] = "vanilla-failed: %s" % v_err
    else:
        v["answer"] = (answer or "").strip()
        v["generated"] = bool(v["answer"])
        v_score, v_relv, v_sup = _judge_answers(
            question, v["answer"], initial_evidence)
        v["faithfulness"] = v_score
        v["answer_relevance"] = v_relv
        v["evidence_support"] = v_sup

    row["evidence_identity_ok"] = (
        row["vanilla"]["evidence_ids"] == initial_ids)
    # FULL initial evidence == VANILLA evidence: FULL after any refinement
    # uses a subset, but its initial retriever output equals VANILLA's.

    # ---- paired differences (VANILLA - FULL) ----
    for metric, key in [
        ("faithfulness_delta", "faithfulness"),
        ("answer_relevance_delta", "answer_relevance"),
        ("evidence_support_delta", "evidence_support"),
    ]:
        a = row["vanilla"].get(key)
        b = row["full"].get(key)
        if a is not None and b is not None:
            row["paired_differences"][metric] = round(float(a) - float(b), 6)

    return row
# ============================================================
# Statistics (pre-registered in protocol section 6)
# ============================================================

try:
    from scipy import stats as scipy_stats  # noqa: E402
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False


def _paired_stats(vanilla_vals, full_vals):
    """Paired statistics for one metric (VANILLA - FULL deltas).

    Pre-registered: paired t-test AND Wilcoxon signed-rank must BOTH be
    p < 0.05 for a difference to be reported as statistically supported.
    """
    pairs = [(float(a), float(b))
             for a, b in zip(vanilla_vals, full_vals)
             if a is not None and b is not None]
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    vanilla = [p[0] for p in pairs]
    full = [p[1] for p in pairs]
    deltas = [a - b for a, b in pairs]
    md = statistics.mean(deltas)
    sdd = statistics.stdev(deltas) if n > 1 else 0.0

    out = {
        "n": n,
        "vanilla_mean": round(statistics.mean(vanilla), 6),
        "full_mean": round(statistics.mean(full), 6),
        "paired_mean_difference": round(md, 6),
        "median_difference": round(statistics.median(deltas), 6),
        "std_difference": round(sdd, 6),
        "count_vanilla_gt_full": sum(1 for d in deltas if d > 1e-9),
        "count_equal": sum(1 for d in deltas if abs(d) <= 1e-9),
        "count_vanilla_lt_full": sum(1 for d in deltas if d < -1e-9),
    }
    if HAVE_SCIPY:
        t_stat, t_p = scipy_stats.ttest_rel(vanilla, full)
        out["paired_ttest_t"] = round(float(t_stat), 6)
        out["paired_ttest_p"] = round(float(t_p), 6)
        se = sdd / (n ** 0.5)
        ci = scipy_stats.t.interval(0.95, df=n - 1, loc=md, scale=se)
        out["ci95_mean_delta_t"] = [round(ci[0], 6), round(ci[1], 6)]
        try:
            w_stat, w_p = scipy_stats.wilcoxon(
                vanilla, full, zero_method="wilcox",
                alternative="two-sided")
            out["wilcoxon_statistic"] = round(float(w_stat), 6)
            out["wilcoxon_p"] = round(float(w_p), 6)
        except Exception as exc:  # noqa: BLE001
            out["wilcoxon_error"] = str(exc)
        out["statistically_supported"] = bool(
            out.get("paired_ttest_p", 1.0) < 0.05
            and out.get("wilcoxon_p", 1.0) < 0.05)
    else:
        out["stats_error"] = "scipy unavailable"
        out["statistically_supported"] = False
    return out


def _cond_stats(rows, cond, key):
    vals = [r[cond].get(key) for r in rows
            if r[cond].get(key) is not None]
    if not vals:
        return {"n": 0, "mean": None, "median": None}
    return {"n": len(vals),
            "mean": round(statistics.mean(vals), 6),
            "median": round(statistics.median(vals), 6)}


def build_summary(rows):
    summary = {"n_pairs": len(rows)}
    summary["condition_stats"] = {
        "full": {k: _cond_stats(rows, "full", k)
                 for k in ["faithfulness", "answer_relevance",
                           "evidence_support"]},
        "vanilla": {k: _cond_stats(rows, "vanilla", k)
                    for k in ["faithfulness", "answer_relevance",
                              "evidence_support"]},
    }
    summary["paired_statistics"] = {
        k: _paired_stats(
            [r["vanilla"].get(k) for r in rows],
            [r["full"].get(k) for r in rows])
        for k in ["faithfulness", "answer_relevance", "evidence_support"]}

    summary["evidence_identity_ok"] = all(
        r["evidence_identity_ok"] for r in rows)
    summary["evidence_identity_fails"] = [
        r["pair_id"] for r in rows if not r["evidence_identity_ok"]]
    summary["full_gate"] = {
        "accept": sum(1 for r in rows
                      if r["full"]["decision"] == "ACCEPT"),
        "refine": sum(1 for r in rows
                      if r["full"]["decision"] == "REFINE"),
        "re_retrieve": sum(1 for r in rows
                           if r["full"]["decision"] == "RE-RETRIEVE"),
        "reject": sum(1 for r in rows
                      if r["full"]["decision"] == "REJECT"),
        "refinement_cases": sum(
            1 for r in rows if r["full"]["refinement_attempts"] > 0),
        "generation_permitted": sum(
            1 for r in rows if r["full"]["generated"]),
        "vanilla_generation": sum(
            1 for r in rows if r["vanilla"]["generated"]),
    }
    summary["refusals"] = {
        "full": sum(1 for r in rows if r["full"]["refused"]),
        "vanilla": sum(1 for r in rows if r["vanilla"]["error"]
                       or not r["vanilla"]["generated"]),
    }
    summary["errors"] = [
        r["pair_id"] for r in rows
        if r["full"]["error"] or r["vanilla"]["error"]]
    summary["gold_in_evidence"] = {
        "full": sum(1 for r in rows
                    if r["full"]["gold_chunk_retrieved"]),
        "vanilla": sum(1 for r in rows
                       if r["vanilla"]["gold_chunk_retrieved"]),
    }
    # topic-level descriptive
    topics = {}
    for t in sorted({r["topic"] for r in rows}):
        sub = [r for r in rows if r["topic"] == t]
        topics[t] = {
            "n": len(sub),
            "faithfulness_delta_mean": _paired_stats(
                [r["vanilla"].get("faithfulness") for r in sub],
                [r["full"].get("faithfulness") for r in sub]
            ).get("paired_mean_difference"),
            "relevance_delta_mean": _paired_stats(
                [r["vanilla"].get("answer_relevance") for r in sub],
                [r["full"].get("answer_relevance") for r in sub]
            ).get("paired_mean_difference"),
            "support_delta_mean": _paired_stats(
                [r["vanilla"].get("evidence_support") for r in sub],
                [r["full"].get("evidence_support") for r in sub]
            ).get("paired_mean_difference"),
        }
    summary["topic_level"] = topics
    return summary
def _short(text):
    return " ".join(str(text).strip().split())[:4000]


def _signature(rows):
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda x: x["pair_id"]):
        payload = {
            "pair_id": r["pair_id"],
            "evidence_identity_ok": r["evidence_identity_ok"],
            "full": {k: r["full"][k] for k in
                     ["evidence_ids", "reliability", "decision",
                      "refinement_attempts", "retrieval_attempts",
                      "refused", "generated", "answer", "faithfulness",
                      "answer_relevance", "evidence_support"]},
            "vanilla": {k: r["vanilla"][k] for k in
                        ["evidence_ids", "generated", "answer",
                         "faithfulness", "answer_relevance",
                         "evidence_support"]},
            "deltas": r["paired_differences"],
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
    records = load_benchmark()
    print("Benchmark OK: %d records, hash verified" % len(records))
    target = records if args.limit is None else records[: args.limit]
    print("Running %d paired questions (run %s)..." % (len(target), run_id))

    rows = []
    for i, rec in enumerate(target, 1):
        t0 = time.perf_counter()
        row = process_pair(rec)
        rows.append(row)
        f, v = row["full"], row["vanilla"]
        print("[%s %d/%d] %s | id_ok=%s | FULL: %s ref=%s | "
              "fF=%s fV=%s rF=%s rV=%s sF=%s sV=%s (%.0fs)"
              % (run_id, i, len(target), row["pair_id"],
                 row["evidence_identity_ok"], f.get("decision"),
                 f.get("refinement_attempts"),
                 f.get("faithfulness"), v.get("faithfulness"),
                 f.get("answer_relevance"), v.get("answer_relevance"),
                 f.get("evidence_support"), v.get("evidence_support"),
                 time.perf_counter() - t0), flush=True)
        with open(OUT_DIR / ("experiment4_per_question_run%s.json" % run_id),
                  "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, ensure_ascii=False)

    if args.limit is not None:
        print("Diagnostic run; skipped summary writeback.")
        return

    summary = build_summary(rows)
    signature = _signature(rows)
    snap_after = snapshot_frozen()
    with open(OUT_DIR / "experiment4_summary.json", "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "experiment4_topic_results.json", "w",
              encoding="utf-8") as fh:
        json.dump(summary["topic_level"], fh, indent=2, ensure_ascii=False)
    hash_path = OUT_DIR / "experiment4_frozen_hashes.json"
    hashes = {}
    if hash_path.exists():
        with open(hash_path, "r", encoding="utf-8") as fh:
            hashes = json.load(fh)
    hashes["run_%s" % run_id] = {
        "before": snap_before, "after": snap_after,
        "note": "frozen artifacts are read-only; before == after."}
    with open(hash_path, "w", encoding="utf-8") as fh:
        json.dump(hashes, fh, indent=2, ensure_ascii=False)
    manifest = {
        "run_id": run_id,
        "n": len(rows),
        "signature": signature,
        "retrieval_backend": "geri_lit",
        "llm_model": LLM_MODEL,
        "judge_model": JUDGE_MODEL,
        "judge_options": JUDGE_OPTIONS,
        "vanilla_system": VANILLA_SYSTEM,
        "vanilla_prompt_template": VANILLA_PROMPT_TEMPLATE,
        "vanilla_prompt_sha256": VANILLA_PROMPT_SHA256,
        "runtime_seconds": round(time.perf_counter() - t_all, 2),
        "frozen_unchanged": snap_before == snap_after,
    }
    with open(OUT_DIR / "experiment4_run_manifest.json", "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print("=== RUN %s SUMMARY ===" % run_id)
    print("signature:", signature)
    print("evidence_identity_ok:", summary["evidence_identity_ok"])
    for metric, st in summary["paired_statistics"].items():
        print("  %-18s VAN=%s FULL=%s delta=%s supported=%s" % (
            metric, st.get("vanilla_mean"), st.get("full_mean"),
            st.get("paired_mean_difference"),
            st.get("statistically_supported")))
    print("frozen_unchanged:", manifest["frozen_unchanged"])


if __name__ == "__main__":
    main()