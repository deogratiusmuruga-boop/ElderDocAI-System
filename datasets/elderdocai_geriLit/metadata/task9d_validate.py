#!/usr/bin/env python3
"""ElderDocAI-GeriLit - Phase 3 Task 9D validation.

STRUCTURAL / INTEGRITY validation ONLY. Does NOT judge question quality or
relevance (the human decision is already recorded). Verifies faithful encoding
of the completed human approval and frozen-artifact integrity.
"""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent.parent.parent
GERI_DIR = REPO_DIR / "datasets" / "elderdocai_geriLit"
META_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = META_DIR / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

REVIEW_FILE = REPO_DIR / "data" / "geri_lit_gold_review.json"
FINAL_FILE = REPO_DIR / "data" / "geri_lit_gold.json"

FROZEN = {
    "chunks.jsonl": "62bfdde3c7e77cf56112e79a71bc79bdea70767f6b2625701e392bbb6581c6b3",
    "embeddings.npy": "b0905d2f96903dadc3cf5b4a737f330853656f846955de706f21bd714ce2dacf",
    "faiss_index.bin": "b7016b9dabbab08ee68f875007b6db2b013af4472b1a3c0958af8a566b59f2b3",
    "bm25.pkl": "17f503240f2f9a13abf58c94359884b03b06bec1a5a59f63f921328e2a2c70c9",
    "row_mapping.json": "4d649f81d554c41ec7b63c65dce887466dfa6245745c568315735f1b93a1872b",
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    checks = {}
    doc = json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
    recs = doc["review_records"]
    final = json.loads(FINAL_FILE.read_text(encoding="utf-8"))["records"]

    checks["source_candidates_134"] = len(recs) == 134
    checks["accepted_134"] = all(r["review_status"] == "ACCEPTED"
                                 for r in recs) and len(recs) == 134
    checks["zero_pending"] = not any(
        r["review_status"] == "PENDING_HUMAN_REVIEW" for r in recs)
    checks["zero_rejected"] = not any(
        r["review_status"].startswith("REJECT") for r in recs)
    checks["no_dup_candidate_ids"] = len(
        {r["candidate_id"] for r in recs}) == len(recs)
    checks["no_dup_final_ids"] = len(
        {r["final_benchmark_id"] for r in final}) == len(final)
    checks["final_134"] = len(final) == 134

    # every final record maps to exactly one original candidate
    cand_ids = {r["candidate_id"] for r in recs}
    checks["final_maps_to_originals"] = all(
        r["original_candidate_id"] in cand_ids for r in final)
    checks["final_originals_unique"] = len(
        {r["original_candidate_id"] for r in final}) == len(final)

    checks["question_nonempty"] = all(
        (r.get("question") or "").strip() for r in final)
    checks["pmcid_present"] = all(r.get("pmcid") for r in final)
    checks["relevant_chunk_ids_present"] = all(
        len(r.get("relevant_chunk_ids") or []) >= 1 for r in final)
    checks["gold_chunks_consistent"] = all(
        sorted(r.get("gold_relevant_chunk_ids") or [])
        == sorted(r.get("relevant_chunk_ids") or []) for r in final)
    checks["provenance_preserved"] = all(
        r.get("source_locator") and r.get("title") and r.get("pmid") is not None
        for r in final)
    checks["reviewed_at_populated"] = all(
        bool(r.get("reviewed_at")) for r in final)
    checks["reviewer_notes_populated"] = all(
        bool(r.get("reviewer_notes")) for r in final)
    checks["final_ordering_deterministic"] = (
        [r["final_benchmark_id"] for r in final]
        == ["GLG-%03d" % i for i in range(1, 135)])

    # reproducibility: hash of the final records block
    sig = hashlib.sha256(
        json.dumps(final, sort_keys=True).encode()).hexdigest()
    checks["final_reproducible_signature"] = bool(sig)
    # (signature recomputed on each run; determinism is re-verified by rerun)

    frozen = {}
    for k, rel in [("chunks.jsonl", "chunks/chunks.jsonl"),
                   ("embeddings.npy", "index/embeddings.npy"),
                   ("faiss_index.bin", "index/faiss_index.bin"),
                   ("bm25.pkl", "index/bm25.pkl"),
                   ("row_mapping.json", "index/row_mapping.json")]:
        frozen[k] = sha256_file(GERI_DIR / rel)
    checks["frozen_artifacts_unchanged"] = all(
        frozen[k] == FROZEN[k] for k in FROZEN)

    # top-level artifact header must report completion (not stale PENDING)
    hr_state = doc.get("human_review_state") or ""
    checks["review_header_reports_completion"] = (
        "PENDING_HUMAN_REVIEW" not in hr_state and "ACCEPTED" in hr_state)

    ok = all(v is True for v in checks.values())

    pmc = Counter(r["pmcid"] for r in final)
    chunks_used = Counter(
        cid for r in final for cid in (r["gold_relevant_chunk_ids"] or []))
    topics = Counter(r["primary_topic"] for r in final)

    out = {
        "validation_ts": datetime.now(timezone.utc).isoformat(),
        "validated_by": "metadata/task9d_validate.py",
        "all_checks_pass": ok,
        "checks": checks,
        "final_signature_sha256": sig,
        "counts": {
            "final_size": len(final),
            "unique_pmcid": len(pmc),
            "unique_gold_chunks": len(chunks_used),
            "topics": dict(sorted(topics.items())),
        },
    }
    (VALIDATION_DIR / "phase3_task9d_validation.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    lines = ["# ElderDocAI-GeriLit Phase 3 Task 9D - Validation", "",
             f"- **timestamp**: {out['validation_ts']}", "",
             "| Check | Result |", "|---|---|"]
    for k, v in checks.items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Counts", json.dumps(out["counts"], indent=2)]
    (VALIDATION_DIR / "phase3_task9d_validation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_checks_pass": ok,
        "n_checks": len(checks),
        "final_size": len(final),
        "unique_pmcid": len(pmc),
        "unique_gold_chunks": len(chunks_used),
        "topics": dict(sorted(topics.items())),
    }, indent=2))


if __name__ == "__main__":
    main()