#!/usr/bin/env python3
"""ElderDocAI - Task 3: Scientific Reliability Re-evaluation on GeriLit-Gold v1.1
(POST-Task-2 corrected relevance semantics).

Population: data/geri_lit_gold_v1_1.json (READ-ONLY, 121 questions).
Retrieval: frozen GeriLit stack (dense + BM25 + hybrid + CrossEncoder rerank)
           exactly as Tasks 10D-10H.
Reliability: current production implementation (evaluate_reliability +
            make_reliability_decision) with post-Task-2 semantics:
            similarity_score = dense cosine (semantic relevance);
            retrieval_score = CrossEncoder logit (ranking only).

The programmatic gate is SIMULATED with the same control flow as
scripts/rag_chat._run_gated_generation, but the LLM is never invoked; the
evaluation records whether generation would be permitted/blocked.

Outputs (created under this directory):
  task3_per_question.json        per-question results
  task3_summary.json             aggregates (reliability/factors/decisions/
                                 evidence behaviour/topics/determinism)
  task3_frozen_hashes.json       frozen-artifact integrity manifest
"""
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

META_DIR = Path(__file__).resolve().parent          # .../metadata/task3_reliability_evaluation
GERI_DIR = META_DIR.parent.parent                  # datasets/elderdocai_geriLit
REPO_DIR = GERI_DIR.parent.parent                  # repo root
for _p in (str(GERI_DIR), str(REPO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.geri_lit_adapter import to_legacy_evidence          # noqa: E402
from scripts.reliability_evaluation import evaluate_reliability  # noqa: E402
from scripts.reliability_config import load_reliability_config  # noqa: E402
from scripts.adaptive_decision_controller import make_reliability_decision  # noqa: E402
import scripts.rag_chat as rag_chat                              # noqa: E402

GOLD_11 = REPO_DIR / "data" / "geri_lit_gold_v1_1.json"
V11_SHA = "1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72"
OUT_PER_Q = META_DIR / "task3_per_question.json"
OUT_SUMMARY = META_DIR / "task3_summary.json"
OUT_HASHES = META_DIR / "task3_frozen_hashes.json"

CONFIG = load_reliability_config()
WEIGHTS = CONFIG["reliability_weights"]
THRESHOLDS = CONFIG["decision_thresholds"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stats(vals):
    vals = [float(v) for v in vals if v is not None]
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


def gate_simulate(query, retriever):
    """Replicate _run_gated_generation WITHOUT calling the LLM.

    Returns dict with evidence flow, reliability, decision, counters and a
    booleean 'generation_permitted' (True only when ACCEPT, or the single
    bounded REFINE variant; the LLM itself is never invoked here).
    """
    retrieval_calls = 0
    refine_attempts = 0
    refused = False
    why = ""

    raw = retriever.retrieve(query)
    retrieval_calls += 1
    chunks = to_legacy_evidence(raw)
    evidence = rag_chat.prepare_evidence(chunks)
    reliability = evaluate_reliability(query=query, evidence_items=evidence)
    decision = make_reliability_decision(reliability)

    guard = 0
    while guard < 8:
        guard += 1
        label = decision["decision"]
        if label == "ACCEPT":
            break
        if label == "REJECT":
            refused = True
            why = "REJECT"
            break
        if label == "REFINE" and refine_attempts < 1:
            refine_attempts += 1
            refined = rag_chat._refine_evidence(evidence, query)
            if not refined:
                refused = True
                why = "REFINE_EMPTY"
                evidence = []
            else:
                evidence = refined
            reliability = evaluate_reliability(query=query, evidence_items=evidence)
            decision = make_reliability_decision(reliability)
            continue
        if label == "RE-RETRIEVE" and retrieval_calls <= 1:
            retrieval_calls += 1
            raw2 = retriever.retrieve(query)
            if not raw2:
                refused = True
                why = "RERETRIEVE_EMPTY"
                break
            evidence = rag_chat.prepare_evidence(to_legacy_evidence(raw2))
            reliability = evaluate_reliability(query=query, evidence_items=evidence)
            decision = make_reliability_decision(reliability)
            continue
        if label == "REFINE" and evidence:
            break
        refused = True
        why = label or "UNKNOWN"
        break

    generation_permitted = not refused
    return {
        "retrieval_calls": retrieval_calls,
        "refine_attempts": refine_attempts,
        "refused": refused,
        "why_refused": why,
        "generation_permitted": generation_permitted,
        "evidence": evidence,
        "reliability": reliability,
        "decision": decision,
    }
def build_per_question(rec, retriever):
    q = rec["question"].strip()
    gold = sorted(rec["gold_relevant_chunk_ids"])
    sim = gate_simulate(q, retriever)
    ev = sim["evidence"]
    ev_ids = [e.get("chunk_id") for e in ev]
    first_gold_rank = None
    for i, cid in enumerate(ev_ids):
        if cid in set(gold):
            first_gold_rank = i + 1
            break
    return {
        "final_benchmark_id": rec["final_benchmark_id"],
        "candidate_id": rec.get("candidate_id"),
        "topic": rec.get("topic"),
        "pmcid": rec.get("pmcid"),
        "question": q,
        "gold_relevant_chunk_ids": gold,
        "evidence_chunk_ids": ev_ids,
        "first_gold_rank_evidence": first_gold_rank,
        "dense_similarity_mean": round(
            statistics.mean([e.get("similarity_score") or 0.0 for e in ev]), 6)
        if ev else None,
        "retrieval_score_mean": round(
            statistics.mean([e.get("retrieval_score") or 0.0 for e in ev]), 6)
        if ev else None,
        "authority": sim["reliability"]["authority"],
        "relevance": sim["reliability"]["relevance"],
        "support": sim["reliability"]["support"],
        "coverage": sim["reliability"]["coverage"],
        "consistency": sim["reliability"]["consistency"],
        "overall_reliability": sim["reliability"]["overall_reliability"],
        "decision": sim["decision"]["decision"],
        "generation_permitted": sim["generation_permitted"],
        "generation_blocked": sim["refused"],
        "refinement_attempts": sim["refine_attempts"],
        "retrieval_attempts": sim["retrieval_calls"],
        "final_evidence_count": len(ev),
    }


def signature(rows):
    canon = []
    for r in rows:
        canon.append({
            "id": r["final_benchmark_id"],
            "rel": r["overall_reliability"],
            "dec": r["decision"],
            "deny": r["generation_blocked"],
            "ev": r["evidence_chunk_ids"],
            "ref": r["refinement_attempts"],
            "ret": r["retrieval_attempts"],
        })
    return hashlib.sha256(
        json.dumps(canon, sort_keys=True).encode()).hexdigest()


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else "run1"
    ts = datetime.now(timezone.utc).isoformat()

    assert sha256(GOLD_11) == V11_SHA, "v1.1 SHA mismatch: STOP"
    gold = json.loads(GOLD_11.read_text(encoding="utf-8"))
    recs = gold["records"]
    assert len(recs) == 121
    assert len({r["final_benchmark_id"] for r in recs}) == 121
    assert all(r["gold_relevant_chunk_ids"] for r in recs)

    from geri_lit_retriever import (GeriLitRetriever, EMBEDDING_MODEL,
                                    RERANK_MODEL, DENSE_W, SPARSE_W)
    retriever = GeriLitRetriever()
    print("retriever:", EMBEDDING_MODEL, "|", RERANK_MODEL)

    rows = [build_per_question(r, retriever) for r in recs]
    sig = signature(rows)
    print("signature:", sig)

    from collections import Counter

    def cat(vals):
        return dict(sorted({k: {"count": c,
                                "pct": round(100.0 * c / max(1, len(vals)), 2)}
                            for k, c in Counter(vals).items()}.items()))

    overall = stats([r["overall_reliability"] for r in rows])
    factors = {f: stats([r[f] for r in rows])
               for f in ("authority", "relevance", "support",
                         "coverage", "consistency")}
    decisions = cat([r["decision"] for r in rows])
    evidence_n = [r["final_evidence_count"] for r in rows]
    evidence_behaviour = {
        "empty_evidence": sum(1 for r in rows if r["final_evidence_count"] == 0),
        "refinement_cases": sum(1 for r in rows if r["refinement_attempts"] > 0),
        "reretrieve_cases": sum(1 for r in rows if r["retrieval_attempts"] > 1),
        "reject_cases": sum(1 for r in rows if r["decision"] == "REJECT"),
        "generation_permitted": sum(1 for r in rows if r["generation_permitted"]),
        "generation_blocked": sum(1 for r in rows if r["generation_blocked"]),
        "evidence_count_mean": round(statistics.mean(evidence_n), 6),
        "evidence_count_median": round(statistics.median(evidence_n), 6),
    }
    topics = {}
    for t in sorted({r["topic"] for r in rows}):
        tr = [r for r in rows if r["topic"] == t]
        topics[t] = {
            "n": len(tr),
            "mean_reliability": round(
                statistics.mean([r["overall_reliability"] for r in tr]), 6),
            "mean_relevance": round(
                statistics.mean([r["relevance"] for r in tr]), 6),
            "decisions": cat([r["decision"] for r in tr]),
        }

    run_manifest = {}
    mf = META_DIR / "task3_run_manifest.json"
    if mf.exists():
        run_manifest = json.loads(mf.read_text(encoding="utf-8"))
    run_manifest[run_name] = {"signature": sig, "generated_at_utc": ts}

    summary = {
        "task": "phase3_task3",
        "run_name": run_name,
        "generated_at_utc": ts,
        "benchmark": {"status": gold.get("status"),
                      "version": gold.get("version"),
                      "n": len(rows),
                      "sha256": V11_SHA,
                      "topics": sorted({r["topic"] for r in rows})},
        "system": {
            "retrieval_backend": "geri_lit (dense+BM25+hybrid+CrossEncoder)",
            "embedding_model": EMBEDDING_MODEL,
            "cross_encoder": RERANK_MODEL,
            "llm_used": False,
            "hybrid_weights": [DENSE_W, SPARSE_W],
            "reliability_formula": (
                "0.3*authority+0.3*relevance+0.2*support"
                "+0.1*coverage+0.1*consistency"),
            "thresholds": {k: float(v) for k, v in THRESHOLDS.items()},
            "thresholds_tuned": False,
        },
        "reliability": overall,
        "factors": factors,
        "decision_distribution": decisions,
        "evidence_behaviour": evidence_behaviour,
        "topics": topics,
        "signature": sig,
        "run_manifest": run_manifest,
        "deterministic": len(run_manifest) >= 2 and len(
            {r["signature"] for r in run_manifest.values()}) == 1,
    }
    OUT_PER_Q.write_text(json.dumps(
        {"task": "phase3_task3", "run_name": run_name, "rows": rows},
        indent=2), encoding="utf-8")
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    mf.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    frozen = {
        "v1.0": sha256(REPO_DIR / "data/geri_lit_gold.json"),
        "v1.1": sha256(GOLD_11),
        "chunks": sha256(GERI_DIR / "chunks/chunks.jsonl"),
        "embeddings": sha256(GERI_DIR / "index/embeddings.npy"),
        "faiss": sha256(GERI_DIR / "index/faiss_index.bin"),
        "bm25": sha256(GERI_DIR / "index/bm25.pkl"),
        "row_mapping": sha256(GERI_DIR / "index/row_mapping.json"),
        "t10d_results": sha256(
            GERI_DIR / "metadata/task10d_retrieval_results.json"),
        "t10e_analysis": sha256(
            GERI_DIR / "metadata/task10e_failure_analysis.json"),
        "t10f_diag": sha256(
            GERI_DIR / "metadata/task10f_benchmark_reconstruction.json"),
        "t10g_diag": sha256(
            GERI_DIR / "metadata/task10g_finalization.json"),
        "t10h_results": sha256(
            GERI_DIR / "metadata/task10h_retrieval_results.json"),
        "reliability_config": sha256(
            REPO_DIR / "config/reliability_config.json"),
    }
    OUT_HASHES.write_text(json.dumps({
        "task": "phase3_task3",
        "run_name": run_name,
        "generated_at_utc": ts,
        "frozen_artifacts": frozen,
    }, indent=2), encoding="utf-8")

    print(json.dumps({
        "n": len(rows),
        "reliability": overall,
        "decisions": decisions,
        "evidence_behaviour": evidence_behaviour,
        "signature": sig,
        "deterministic_so_far": summary["deterministic"],
    }, indent=2))


if __name__ == "__main__":
    main()