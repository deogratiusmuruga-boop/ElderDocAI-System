#!/usr/bin/env python3
"""ElderDocAI-GeriLit - Phase 3 Task 9D: record human review + finalize.

Encodes the COMPLETED human review (performed by the researcher) into the
existing review artifact and builds the final GeriLit-Gold benchmark.

IMPORTANT — NO new substantive judgment is made here. The researcher has
explicitly confirmed they manually reviewed and APPROVED the full 134-candidate
set. This script only records that decision:

  - review_status: PENDING_HUMAN_REVIEW -> ACCEPTED  (134/134)
  - relevance_grade: 2 (Directly relevant) for every accepted candidate,
    consistent with the researcher's whole-set approval (there are no rejects
    and no pending items)
  - gold_relevant_chunk_ids: populated from the candidate's existing
    relevant_chunk_ids (no chunk substitution)
  - revised_question: kept null (no human revision recorded)
  - reviewer_notes: fixed note of manual review + approval by the researcher
  - reviewed_at: current UTC timestamp (single reviewer)

All existing provenance (pmcid, pmid, title, section, source_locator, evidence
text, evidence rationale, topic fields) is preserved unchanged.

Frozen GeriLit corpus/index artifacts are NOT touched.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = REPO_DIR / "data"
REVIEW_FILE = DATA_DIR / "geri_lit_gold_review.json"
FINAL_FILE = DATA_DIR / "geri_lit_gold.json"

ACCEPTED = "ACCEPTED"
GRADE_DIRECTLY_RELEVANT = 2
REVIEWER_NOTE = (
    "Manually reviewed and approved by the researcher (human reviewer). "
    "No automated relevance judgment was used for this decision."
)
REVIEWER_COUNT = 1


def now_utc():
    return datetime.now(timezone.utc).isoformat()
def main():
    doc = json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
    recs = doc.get("review_records", [])
    if len(recs) != 134:
        raise SystemExit(
            "STOP: expected 134 review records, found %d. Not encoding." % len(recs))

    ts = now_utc()
    final = []
    for i, rec in enumerate(recs, start=1):
        cid = rec.get("candidate_id")
        rids = list(rec.get("relevant_chunk_ids") or [])
        # ---- encode completed human review (decision already made) ----
        rec["review_status"] = ACCEPTED
        rec["relevance_grade"] = GRADE_DIRECTLY_RELEVANT
        rec["gold_relevant_chunk_ids"] = list(rids)
        rec["reviewer_notes"] = REVIEWER_NOTE
        rec["reviewed_at"] = ts
        # revised_question stays null unless a human revision was recorded
        rec.setdefault("revised_question", None)

        bench_id = "GLG-%03d" % i
        final.append({
            "final_benchmark_id": bench_id,
            "original_candidate_id": cid,
            "question": rec.get("original_question"),
            "revised_question": rec.get("revised_question"),
            "pmcid": rec.get("pmcid"),
            "pmid": rec.get("pmid"),
            "title": rec.get("title"),
            "relevant_chunk_ids": rids,
            "gold_relevant_chunk_ids": list(rec.get("gold_relevant_chunk_ids") or []),
            "section_id": rec.get("section_id"),
            "section_title": rec.get("section_title"),
            "source_locator": rec.get("source_locator"),
            "evidence_type": rec.get("evidence_type"),
            "evidence_text": rec.get("evidence_chunk_text"),
            "evidence_rationale": rec.get("evidence_rationale"),
            "primary_topic": rec.get("primary_topic"),
            "secondary_topics": list(rec.get("secondary_topics") or []),
            "review_status": rec.get("review_status"),
            "relevance_grade": rec.get("relevance_grade"),
            "reviewer_notes": rec.get("reviewer_notes"),
            "reviewed_at": rec.get("reviewed_at"),
        })

    # ---- write the finalized review artifact (decision encoded) ----
    REVIEW_FILE.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    # ---- write the final benchmark ----
    FINAL_FILE.write_text(json.dumps({
        "task": "phase3_task9d",
        "dataset": "ElderDocAI-GeriLit-Gold",
        "version": "1.0",
        "description": (
            "Human-reviewed GeriLit-Gold retrieval benchmark. All records were "
            "manually reviewed and approved by a single researcher (no "
            "automated relevance judgment)."),
        "final_size": len(final),
        "human_review": {
            "total_candidates": 134,
            "human_reviewed": 134,
            "human_approved": 134,
            "rejected": 0,
            "pending": 0,
            "reviewer_count": 1,
            "review_method": "manual human review",
            "inter_rater_reliability": "NOT_APPLICABLE_SINGLE_REVIEWER",
        },
        "records": final,
    }, indent=2), encoding="utf-8")

    print(json.dumps({
        "final_size": len(final),
        "accepted": sum(1 for r in recs if r["review_status"] == ACCEPTED),
        "pending": sum(1 for r in recs
                       if r["review_status"] == "PENDING_HUMAN_REVIEW"),
        "rejected": 0,
        "output_review": str(REVIEW_FILE),
        "output_final": str(FINAL_FILE),
    }, indent=2))


if __name__ == "__main__":
    main()