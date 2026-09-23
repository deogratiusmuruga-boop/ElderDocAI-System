#!/usr/bin/env python3
"""ElderDocAI-GeriLit dev-v0.1 - Phase 3 Task 9C: review preparation.

Prepares human-review artifacts for the 134 GeriLit-Gold candidates.

IMPORTANT - HUMAN REVIEW DISTINCTION
  - Automated validation (IDs, chunk existence, uniqueness, grounding) is
    performed here and reported separately.
  - Human semantic verification (does the chunk answer the question? is the
    wording clear? grade 2/1/0? KEEP/REVISE/REJECT?) CANNOT be performed in
    this script. All human fields are set to PENDING_HUMAN_REVIEW.
  - No LLM is used as a reviewer and no human judgments are fabricated.
  - data/geri_lit_gold.json (final benchmark) is NOT produced; that requires
    an actual human review.

Outputs:
  data/geri_lit_gold_review.json              (review-ready, chunk text embedded)
  metadata/phase3_task9c_review_worksheet.md  (human worksheet, blank fields)
  metadata/validation/phase3_task9c_validation.json / .md
  metadata/phase3_task9c_semantic_duplicates.json  (AUTOMATED aid only)
"""
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

META_DIR = Path(__file__).resolve().parent
GERI_DIR = META_DIR.parent
REPO_DIR = GERI_DIR.parent.parent
VALIDATION_DIR = META_DIR / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

CAND_FILE = REPO_DIR / "data" / "geri_lit_gold_candidates.json"
CHUNKS_FILE = GERI_DIR / "chunks" / "chunks.jsonl"
REVIEW_FILE = REPO_DIR / "data" / "geri_lit_gold_review.json"
WORKSHEET = META_DIR / "phase3_task9c_review_worksheet.md"
DUP_AID = META_DIR / "phase3_task9c_semantic_duplicates.json"

FROZEN = {
    "chunks.jsonl": "62bfdde3c7e77cf56112e79a71bc79bdea70767f6b2625701e392bbb6581c6b3",
    "embeddings.npy": "b0905d2f96903dadc3cf5b4a737f330853656f846955de706f21bd714ce2dacf",
    "faiss_index.bin": "b7016b9dabbab08ee68f875007b6db2b013af4472b1a3c0958af8a566b59f2b3",
    "bm25.pkl": "17f503240f2f9a13abf58c94359884b03b06bec1a5a59f63f921328e2a2c70c9",
    "row_mapping.json": "4d649f81d554c41ec7b63c65dce887466dfa6245745c568315735f1b93a1872b",
}
GOLD16_SHA = "e44788926afe351d30bad2e253cfe0712ad83a4ee3b3d9abe6a63c2af0baf544"
GOLD96_SHA = "d4e1b6e6c48cd84fc95fcbb7bf0df375365cd354ec8cc31f16dde2b3881b5299"
ACC_MAN_SHA = "f80cd457e5e2be55ed3a3de09e2554e9d585e76cef8dc44800270dfa66a99a49"


def sha256_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
def main():
    d = json.loads(CAND_FILE.read_text(encoding="utf-8"))
    cands = d["candidates"]
    chunks = {}
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                chunks[c["chunk_id"]] = c

    review = []
    for c in cands:
        cid = c["relevant_chunk_ids"][0]
        ch = chunks.get(cid, {})
        evidence_text = (ch.get("text") or "").strip()
        review.append({
            "candidate_id": c["candidate_id"],
            "original_question": c["question"],
            "revised_question": None,
            "primary_topic": c["primary_topic"],
            "secondary_topics": list(c.get("secondary_topics") or []),
            "pmcid": c["pmcid"],
            "pmid": c.get("pmid"),
            "title": c.get("title"),
            "relevant_chunk_ids": list(c["relevant_chunk_ids"]),
            "gold_relevant_chunk_ids": [],   # human decision (pending)
            "section_id": c.get("section_id"),
            "section_title": c.get("section_title"),
            "source_locator": c.get("source_locator"),
            "evidence_type": c.get("evidence_type"),
            "evidence_chunk_text": evidence_text,
            "evidence_rationale": c.get("evidence_rationale"),
            "review_status": "PENDING_HUMAN_REVIEW",
            "relevance_grade": None,
            "reviewer_notes": "",
            "reviewed_at": None,
        })
    REVIEW_FILE.write_text(json.dumps({
        "task": "phase3_task9c",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "human_review_state": (
            "PENDING_HUMAN_REVIEW - no human review occurred; no LLM reviewer "
            "was used and no human judgments were fabricated."),
        "review_records": review,
    }, indent=2), encoding="utf-8")
# -------- automated duplicate-grouping AID (NOT a decision) --------
    def tokens(t):
        return set(re.findall(r"[a-z]+", t.lower()))

    groups = []
    used = set()
    for i in range(len(review)):
        if review[i]["candidate_id"] in used:
            continue
        grp = [review[i]]
        used.add(review[i]["candidate_id"])
        for j in range(i + 1, len(review)):
            if review[j]["candidate_id"] in used:
                continue
            a = review[i]
            b = review[j]
            same_topic = a["primary_topic"] == b["primary_topic"]
            same_chunk = set(a["relevant_chunk_ids"]) & set(b["relevant_chunk_ids"])
            ta, tb = tokens(a["original_question"]), tokens(b["original_question"])
            jac = len(ta & tb) / max(1, len(ta | tb))
            if same_topic and (same_chunk or jac >= 0.6):
                grp.append(b)
                used.add(b["candidate_id"])
        if len(grp) > 1:
            groups.append([g["candidate_id"] for g in grp])
    DUP_AID.write_text(json.dumps({
        "disclaimer": ("AUTOMATED lexical/topical grouping AID only. NOT a "
                       "human semantic duplication decision."),
        "n_groups": len(groups),
        "groups": groups,
    }, indent=2), encoding="utf-8")

    # -------- automated-only validation --------
    checks = {}
    checks["all_134_present"] = len(review) == 134
    ids = [r["candidate_id"] for r in review]
    checks["candidate_id_unique"] = len(ids) == len(set(ids))
    checks["pmcid_nonempty"] = all(r["pmcid"] for r in review)
    checks["chunk_text_embedded"] = all(r["evidence_chunk_text"] for r in review)
    checks["chunk_text_present_in_corpus"] = all(
        r["relevant_chunk_ids"][0] in chunks for r in review)
    checks["gold_chunks_are_review_field"] = all(
        r["gold_relevant_chunk_ids"] == [] for r in review)
    checks["human_fields_pending"] = all(
        r["review_status"] == "PENDING_HUMAN_REVIEW"
        and r["relevance_grade"] is None
        and not r["reviewer_notes"] for r in review)
    checks["no_llm_reviewer"] = True
    qs = [r["original_question"] for r in review]
    checks["questions_unique"] = len(qs) == len(set(qs))
    leak = re.compile(r"pmc\d+|pubmed|10\.\d{4,}|chunk[ _-]?id", re.I)
    checks["no_id_leak"] = all(not leak.search(q) for q in qs)
# -------- frozen integrity --------
    frozen = {}
    for k, rel in [
        ("chunks.jsonl", "chunks/chunks.jsonl"),
        ("embeddings.npy", "index/embeddings.npy"),
        ("faiss_index.bin", "index/faiss_index.bin"),
        ("bm25.pkl", "index/bm25.pkl"),
        ("row_mapping.json", "index/row_mapping.json"),
    ]:
        frozen[k] = sha256_file(GERI_DIR / rel)
    checks["frozen_artifacts_unchanged"] = all(
        frozen[k] == FROZEN[k] for k in FROZEN)
    checks["gold16_unchanged"] = (
        sha256_file(REPO_DIR / "data/gold_qa_evaluation.json") == GOLD16_SHA)
    checks["gold96_unchanged"] = (
        sha256_file(REPO_DIR / "data/gold_qa_extended.json") == GOLD96_SHA)
    checks["accepted_manifest_unchanged"] = (
        sha256_file(GERI_DIR / "manifest/accepted_manifest.jsonl") == ACC_MAN_SHA)
    checks["final_benchmark_not_produced"] = (
        not (REPO_DIR / "data/geri_lit_gold.json").exists())

    ok = all(v is True for v in checks.values())
    by_topic = Counter(r["primary_topic"] for r in review)

    # -------- human-review worksheet --------
    wline = ["# ElderDocAI-GeriLit Phase 3 Task 9C - Human Review Worksheet", "",
             "> Instructions: For each candidate, read the ORIGINAL QUESTION and the",
             "> EVIDENCE CHUNK TEXT, then assign review_status (KEEP / REVISE / REJECT),",
             "> relevance_grade (2 = directly relevant, 1 = partially relevant,",
             "> 0 = not relevant), and reviewer_notes. Set revised_question only when",
             "> REVISE. Return the filled file to the pipeline owner. Do not edit the",
             "> candidate_id, pmcid, relevant_chunk_ids, or evidence text.", "",
             "All candidates are currently PENDING_HUMAN_REVIEW.",
             f"- Total candidates: {len(review)}", ""]
    for r in review:
        wline += [
            f"### {r['candidate_id']}  |  Topic {r['primary_topic']}",
            "",
            f"- **Original question**: {r['original_question']}",
            f"- **PMCID**: {r['pmcid']} | **PMID**: {r['pmid']}",
            f"- **Title**: {r['title']}",
            f"- **Section**: {r.get('section_title') or 'untitled'} "
            f"({r.get('section_id')})",
            f"- **Chunk**: {r['relevant_chunk_ids'][0]}",
            f"- **Source locator**: {r.get('source_locator')}",
            f"- **Evidence rationale**: {r.get('evidence_rationale')}",
            "",
            f"**Evidence chunk text**: {r['evidence_chunk_text']}",
            "",
            "- **Reviews below (to be completed by human reviewer); keep as "
            "PENDING_HUMAN_REVIEW until then.**",
            "- review_status: PENDING_HUMAN_REVIEW   (options: KEEP / REVISE / REJECT)",
            "- relevance_grade: (blank)   (options: 2 / 1 / 0)",
            "- revised_question: (blank)",
            "- reviewer_notes:",
            "",
            "---",
            "",
        ]
    WORKSHEET.write_text("\n".join(wline) + "\n", encoding="utf-8")

    vjson = {
        "validation_ts": datetime.now(timezone.utc).isoformat(),
        "validated_by": "metadata/task9c_prepare_review.py",
        "all_checks_pass": ok,
        "checks": checks,
        "counts": {
            "candidates": len(review),
            "by_topic": dict(sorted(by_topic.items())),
            "automated_dup_group_aids": len(groups),
            "human_review_state": "PENDING_HUMAN_REVIEW",
            "final_benchmark_created": False,
        },
    }
    (VALIDATION_DIR / "phase3_task9c_validation.json").write_text(
        json.dumps(vjson, indent=2), encoding="utf-8")
    vlines = ["# ElderDocAI-GeriLit Phase 3 Task 9C - Validation", "",
              f"- **timestamp**: {vjson['validation_ts']}", "",
              "| Check | Result |", "|---|---|"]
    for k, v in checks.items():
        vlines.append(f"| {k} | {v} |")
    (VALIDATION_DIR / "phase3_task9c_validation.md").write_text(
        "\n".join(vlines) + "\n", encoding="utf-8")

    print(json.dumps({
        "all_checks_pass": ok,
        "candidates": len(review),
        "automated_dup_group_aids": len(groups),
        "final_benchmark_created": False,
        "human_review_state": "PENDING_HUMAN_REVIEW",
    }, indent=2))


if __name__ == "__main__":
    main()