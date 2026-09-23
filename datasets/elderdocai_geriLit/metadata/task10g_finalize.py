#!/usr/bin/env python3
"""ElderDocAI-GeriLit - Phase 3 Task 10G: human-review encoding + v1.1 finalization.

Encodes the COMPLETED human review (researcher approved all 121 v1.1
candidates) and freezes GeriLit-Gold v1.1.

NO new substantive judgment is made here: the researcher manually reviewed and
approved all 121 candidates. This script only records that decision:
  - review_status: PENDING_HUMAN_REVIEW -> ACCEPTED (121/121)
  - relevance_grade: 2 (Directly relevant) for every accepted record,
    consistent with the whole-set approval and the Task 9D v1.0 convention
  - revised_question: not introduced (no human revision recorded)
  - reviewer_notes / reviewed_at: recorded (single reviewer, no identity)

Outputs:
  data/geri_lit_gold_v1_1.json          (FROZEN v1.1 benchmark, n=121)
  data/geri_lit_gold_v1_1_review.json   (review-state artifact)
  metadata/task10g_finalization.json    (before/after hashes + signatures)

Frozen v1.0 / corpus / index / Task 10D / Task 10E / Task 10F artifacts are
READ-ONLY and verified unchanged. No retrieval evaluation is performed.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

META_DIR = Path(__file__).resolve().parent
GERI_DIR = META_DIR.parent
REPO_DIR = GERI_DIR.parent.parent

CAND_FILE = REPO_DIR / "data" / "geri_lit_gold_v1_1_candidate.json"
FINAL_FILE = REPO_DIR / "data" / "geri_lit_gold_v1_1.json"
REVIEW_FILE = REPO_DIR / "data" / "geri_lit_gold_v1_1_review.json"
BASELINE_FILE = META_DIR / "task10f_frozen_hashes.json"
DIAG_FILE = META_DIR / "task10g_finalization.json"

GOLD_V10_FILE = REPO_DIR / "data" / "geri_lit_gold.json"
EXPECT_V10 = "28ef54faeec3b9a1cb8c1c79c9a478d8ea3618574b2787a82a74c1460f209c62"

ACCEPTED = "ACCEPTED"
GRADE = 2
REVIEWER_NOTE = (
    "Manually reviewed and approved by the researcher (human reviewer). "
    "No automated relevance judgment was used for this decision."
)
TS_SENTINEL = "REVIEWED_AT_TS"   # normalized for deterministic signature


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ts = datetime.now(timezone.utc).isoformat()
    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))["baseline"]
    bl = lambda rel: baseline.get(rel, {}).get("sha256")  # noqa: E731

    # ------------------------------------------------ protected hashes
    protected = {
        "v1_0": (GOLD_V10_FILE, EXPECT_V10),
        "chunks": (GERI_DIR / "chunks/chunks.jsonl",
                   bl("chunks/chunks.jsonl")),
        "embeddings": (GERI_DIR / "index/embeddings.npy",
                       bl("index/embeddings.npy")),
        "faiss": (GERI_DIR / "index/faiss_index.bin",
                  bl("index/faiss_index.bin")),
        "bm25": (GERI_DIR / "index/bm25.pkl", bl("index/bm25.pkl")),
        "row_mapping": (GERI_DIR / "index/row_mapping.json",
                        bl("index/row_mapping.json")),
        "t10d_results": (META_DIR / "task10d_retrieval_results.json",
                         bl("metadata/task10d_retrieval_results.json")),
        "t10d_summary": (META_DIR / "task10d_summary.json",
                         bl("metadata/task10d_summary.json")),
        "t10e_analysis": (META_DIR / "task10e_failure_analysis.json",
                          bl("metadata/task10e_failure_analysis.json")),
        "t10e_report": (
            META_DIR / "phase3_task10e_retrieval_failure_analysis_report.md",
            bl("metadata/phase3_task10e_retrieval_failure_analysis_report.md")),
    }
    before = {}
    for k, (p, exp) in protected.items():
        before[k] = sha256(p)
        assert exp is not None and before[k] == exp, (
            "STOP: protected artifact %s changed: %s" % (k, before[k]))

    # Task 10F products: verified by before/after identity plus a content anchor
    # on the Task 10F records signature.
    t10f_products = {
        "t10f_candidate": CAND_FILE,
        "t10f_diag": META_DIR / "task10f_benchmark_reconstruction.json",
        "t10f_hashes": BASELINE_FILE,
        "t10f_validation":
            META_DIR / "validation/phase3_task10f_validation.json",
    }
    t10f_before = {k: sha256(p) for k, p in t10f_products.items()}
    t10f_diag = json.loads(
        (META_DIR / "task10f_benchmark_reconstruction.json")
        .read_text(encoding="utf-8"))
    cand_doc = json.loads(CAND_FILE.read_text(encoding="utf-8"))
    cands = cand_doc["candidates"]
    rec_sig = hashlib.sha256(
        json.dumps(cands, sort_keys=True).encode()).hexdigest()
    assert rec_sig == t10f_diag["candidate_signature_sha256"], (
        "STOP: candidate records no longer match the Task 10F signature")
    assert len(cands) == 121, "STOP: expected 121 candidates, got %d" % len(cands)
    assert all(c["review_status"] == "PENDING_HUMAN_REVIEW" for c in cands), (
        "STOP: candidate records are not all PENDING_HUMAN_REVIEW")
# ------------------------------------------------ build final records
    records, review_rows = [], []
    for i, c in enumerate(cands, start=1):
        rec = {k: c[k] for k in c}          # deep copy of every candidate field
        rec["final_benchmark_id"] = "GLG11-%03d" % i
        rec["review_status"] = ACCEPTED
        rec["relevance_grade"] = GRADE
        rec["reviewer_notes"] = REVIEWER_NOTE
        rec["reviewed_at"] = ts
        if "revised_question" not in rec:
            rec["revised_question"] = None
        records.append(rec)
        review_rows.append({
            "candidate_id": c["candidate_id"],
            "final_benchmark_id": rec["final_benchmark_id"],
            "original_v1_0_id": c["original_v1_0_id"],
            "original_candidate_id": c["original_candidate_id"],
            "question": c["question"],
            "pmcid": c["pmcid"],
            "topic": c["topic"],
            "gold_relevant_chunk_ids": list(c["gold_relevant_chunk_ids"]),
            "review_status": ACCEPTED,
            "relevance_grade": GRADE,
            "reviewer_notes": REVIEWER_NOTE,
            "reviewed_at": ts,
        })

    # ------------------------------------------------ write final benchmark
    final = {
        "dataset": "ElderDocAI-GeriLit-Gold",
        "version": "1.1",
        "status": "FROZEN",
        "task": "phase3_task10g",
        "generated_at_utc": ts,
        "description": (
            "Frozen GeriLit-Gold v1.1 retrieval benchmark. All 121 records "
            "were manually reviewed and approved by a single researcher "
            "(no automated relevance judgment). Evidence-first v1.1 record set "
            "derived from the 134 v1.0 records."),
        "final_size": len(records),
        "human_review": {
            "benchmark_version": "1.1",
            "human_review_state": "ACCEPTED_COMPLETED",
            "total_candidates": 121,
            "reviewed": 121,
            "accepted": 121,
            "rejected": 0,
            "pending": 0,
            "reviewer_count": 1,
            "review_method": "manual human review",
            "inter_rater_reliability": "NOT_APPLICABLE_SINGLE_REVIEWER",
        },
        "records": records,
    }
    FINAL_FILE.write_text(json.dumps(final, indent=2), encoding="utf-8")

    # ------------------------------------------------ review artifact
    review_doc = {
        "dataset": "ElderDocAI-GeriLit-Gold",
        "benchmark_version": "1.1",
        "task": "phase3_task10g",
        "generated_at_utc": ts,
        "human_review_state": (
            "ACCEPTED_COMPLETED - human review completed. All 121 candidates "
            "were manually reviewed and approved by the researcher "
            "(1 reviewer); no automated relevance judgment was used."),
        "review_summary": {
            "total_candidates": 121,
            "reviewed": 121,
            "accepted": 121,
            "rejected": 0,
            "pending": 0,
            "reviewer_count": 1,
            "review_method": "manual human review",
            "inter_rater_reliability": "NOT_APPLICABLE_SINGLE_REVIEWER",
        },
        "review_records": review_rows,
    }
    REVIEW_FILE.write_text(json.dumps(review_doc, indent=2), encoding="utf-8")

    # ------------------------------------------------ deterministic signature
    def norm(rec):
        r = dict(rec)
        r["reviewed_at"] = TS_SENTINEL
        r.pop("generated_at_utc", None)
        return r

    sig = hashlib.sha256(
        json.dumps([norm(r) for r in records], sort_keys=True).encode()
    ).hexdigest()

    # ------------------------------------------------ verify after
    after = {k: sha256(p) for k, (p, _) in protected.items()}
    t10f_after = {k: sha256(p) for k, p in t10f_products.items()}
    unchanged = all(after[k] == before[k] for k in protected)
    t10f_unchanged = all(t10f_after[k] == t10f_before[k]
                         for k in t10f_products)
    assert unchanged and t10f_unchanged, (
        "STOP: protected artifact changed during finalize")

    diag = {
        "task": "phase3_task10g",
        "generated_at_utc": ts,
        "generated_by": "metadata/task10g_finalize.py",
        "protected_before": before,
        "protected_after": after,
        "protected_unchanged": unchanged,
        "t10f_before": t10f_before,
        "t10f_after": t10f_after,
        "t10f_unchanged": t10f_unchanged,
        "final_sha256": sha256(FINAL_FILE),
        "review_sha256": sha256(REVIEW_FILE),
        "candidate_sha256": sha256(CAND_FILE),
        "final_benchmark_signature": sig,
        "final_size": len(records),
        "accepted": sum(1 for r in records if r["review_status"] == ACCEPTED),
        "rejected": 0,
        "pending": sum(1 for r in records
                       if r["review_status"] == "PENDING_HUMAN_REVIEW"),
    }
    DIAG_FILE.write_text(json.dumps(diag, indent=2), encoding="utf-8")

    print(json.dumps({
        "final_size": len(records),
        "accepted": diag["accepted"],
        "rejected": 0,
        "pending": diag["pending"],
        "final_sha256": diag["final_sha256"],
        "final_benchmark_signature": sig,
        "protected_unchanged": unchanged,
        "t10f_unchanged": t10f_unchanged,
        "final_file": str(FINAL_FILE),
        "review_file": str(REVIEW_FILE),
    }, indent=2))


if __name__ == "__main__":
    main()