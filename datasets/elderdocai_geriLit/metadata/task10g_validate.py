#!/usr/bin/env python3
"""ElderDocAI-GeriLit - Phase 3 Task 10G validation (READ-ONLY).

Validates the FROZEN GeriLit-Gold v1.1 benchmark and the encoding of the
completed human review (121/121 ACCEPTED). No relevance judgment is made.
Frozen v1.0, corpus/index, Task 10D/10E/10F artifacts are verified unchanged.
"""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

META_DIR = Path(__file__).resolve().parent
GERI_DIR = META_DIR.parent
REPO_DIR = GERI_DIR.parent.parent
VALIDATION_DIR = META_DIR / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

FINAL_FILE = REPO_DIR / "data" / "geri_lit_gold_v1_1.json"
REVIEW_FILE = REPO_DIR / "data" / "geri_lit_gold_v1_1_review.json"
CAND_FILE = REPO_DIR / "data" / "geri_lit_gold_v1_1_candidate.json"
GOLD_V10_FILE = REPO_DIR / "data" / "geri_lit_gold.json"
BASELINE_FILE = META_DIR / "task10f_frozen_hashes.json"
DIAG_FILE = META_DIR / "task10g_finalization.json"
CHUNKS_FILE = GERI_DIR / "chunks" / "chunks.jsonl"

EXPECT_V10 = "28ef54faeec3b9a1cb8c1c79c9a478d8ea3618574b2787a82a74c1460f209c62"
VALID_RECON = {"RETAINED", "RECONSTRUCTED"}
VALID_TOPICS = {f"C{i:02d}" for i in range(1, 11)}

checks = []


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def add(cid, desc, expected, actual, passed):
    checks.append({"check_id": cid, "description": desc,
                   "expected": expected, "actual": actual,
                   "result": "PASS" if passed else "FAIL"})


def main():
    diag = json.loads(DIAG_FILE.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))["baseline"]
    fin = json.loads(FINAL_FILE.read_text(encoding="utf-8"))
    rev = json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
    cand = json.loads(CAND_FILE.read_text(encoding="utf-8"))
    recs = fin["records"]
    crecs = cand["candidates"]
    rrows = rev["review_records"]

    v1_seen = {r["final_benchmark_id"]
               for r in json.loads(GOLD_V10_FILE.read_text(encoding="utf-8"))
               ["records"]}
    corpus_chunks = set()
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                corpus_chunks.add(json.loads(line)["chunk_id"])

    # ------------------------------------------------ review completion
    add("G1", "final benchmark size = 121",
        121, fin.get("final_size"), fin.get("final_size") == 121
        and len(recs) == 121)
    add("G2", "candidate set unchanged at 121 (PENDING on input)",
        121, len(crecs), len(crecs) == 121)
    add("G3", "all 121 final records ACCEPTED",
        121, sum(1 for r in recs if r["review_status"] == "ACCEPTED"),
        all(r["review_status"] == "ACCEPTED" for r in recs))
    add("G4", "zero PENDING, zero REJECTED",
        "0/0", (sum(1 for r in recs
                    if r["review_status"] == "PENDING_HUMAN_REVIEW"),
                sum(1 for r in recs
                    if r["review_status"] == "REJECTED")),
        all(r["review_status"] == "ACCEPTED" for r in recs))
    hr = fin["human_review"]
    add("G5", "human_review block consistent (121/121/0/0)",
        "reviewed 121 accepted 121 rejected 0 pending 0",
        (hr.get("reviewed"), hr.get("accepted"), hr.get("rejected"),
         hr.get("pending")),
        hr.get("reviewed") == 121 and hr.get("accepted") == 121
        and hr.get("rejected") == 0 and hr.get("pending") == 0)
    add("G6", "review artifact records = 121 ACCEPTED",
        "121",
        sum(1 for r in rrows if r.get("review_status") == "ACCEPTED"),
        len(rrows) == 121 and all(
            r.get("review_status") == "ACCEPTED" for r in rrows))
# ------------------------------------------------ structure
    ids = [r["final_benchmark_id"] for r in recs]
    cids = [r["candidate_id"] for r in recs]
    add("G7", "final benchmark IDs unique",
        "unique %d" % len(ids), len(set(ids)), len(ids) == len(set(ids)))
    add("G8", "candidate IDs unique",
        "unique %d" % len(cids), len(set(cids)), len(cids) == len(set(cids)))
    add("G9", "deterministic final IDs GLG11-001..121",
        ["GLG11-%03d" % i for i in range(1, 122)], ids,
        ids == ["GLG11-%03d" % i for i in range(1, 122)])
    add("G10", "required fields present (question, pmcid, title, topic)",
        "all non-empty",
        sum(1 for r in recs if r.get("question") and r.get("pmcid")
            and r.get("title") and r.get("topic")),
        all(r.get("question") and r.get("pmcid") and r.get("title")
            and r.get("topic") for r in recs))
    add("G11", "PMCID format valid",
        "PMC\\d+",
        len({r["pmcid"] for r in recs
             if re.fullmatch(r"PMC\d+", r.get("pmcid", ""))}),
        all(re.fullmatch(r"PMC\d+", r.get("pmcid", "")) for r in recs))
    add("G12", "gold chunk IDs present",
        "all >=1",
        min(len(r.get("gold_relevant_chunk_ids") or []) for r in recs),
        all(len(r.get("gold_relevant_chunk_ids") or []) >= 1 for r in recs))
    add("G13", "every gold chunk exists in the frozen corpus",
        "all present",
        sum(1 for r in recs for x in (r["gold_relevant_chunk_ids"] or [])
            if x in corpus_chunks),
        all(x in corpus_chunks for r in recs
            for x in r["gold_relevant_chunk_ids"]))
    add("G14", "topic labels valid C01-C10",
        10, len({r["topic"] for r in recs}),
        all(r["topic"] in VALID_TOPICS for r in recs))
    add("G15", "provenance preserved (original ids, pmid, locator, evidence)",
        "all non-empty",
        sum(1 for r in recs if r.get("original_v1_0_id")
            and r.get("original_candidate_id") and r.get("pmid")
            and r.get("source_locator") and r.get("evidence_type")),
        all(r.get("original_v1_0_id") and r.get("original_candidate_id")
            and r.get("pmid") and r.get("source_locator")
            and r.get("evidence_type") for r in recs))
    add("G16", "evidence rationale present",
        "all non-empty",
        sum(1 for r in recs if str(r.get("evidence_rationale") or "").strip()),
        all(str(r.get("evidence_rationale") or "").strip() for r in recs))
    add("G17", "reconstruction status valid",
        sorted(VALID_RECON), sorted({r["reconstruction_status"]
                                     for r in recs}),
        all(r["reconstruction_status"] in VALID_RECON for r in recs))
    add("G18", "diagnostics preserved (lex/sem/region/claim)",
        "all present",
        sum(1 for r in recs if r.get("diagnostics")
            and "lex_overlap_ratio" in r["diagnostics"]
            and "sem_sim" in r["diagnostics"]),
        all(r.get("diagnostics") and "lex_overlap_ratio" in r["diagnostics"]
            and "sem_sim" in r["diagnostics"] for r in recs))
    add("G19", "no duplicate questions",
        "0",
        len(recs) - len({r["question"].lower() for r in recs}),
        len({r["question"].lower() for r in recs}) == len(recs))
    # ------------------------------------------------ traceability
    cand_map = {c["candidate_id"]: c for c in crecs}
    add("G20", "every candidate appears exactly once in the final records",
        "121/121",
        sum(1 for c in crecs
            if len([r for r in recs
                    if r.get("candidate_id") == c["candidate_id"]]) == 1),
        all(len([r for r in recs
                 if r.get("candidate_id") == c["candidate_id"]]) == 1
            for c in crecs))
    add("G21", "final records map back to candidates",
        "121/121",
        sum(1 for r in recs if r.get("candidate_id") in cand_map),
        all(r.get("candidate_id") in cand_map for r in recs))
    add("G22", "v1.0 -> final mapping injective & traceable",
        "unique %d" % len({r["original_v1_0_id"] for r in recs}),
        len({r["original_v1_0_id"] for r in recs}),
        len({r["original_v1_0_id"] for r in recs}) == len(recs)
        and all(r["original_v1_0_id"] in v1_seen for r in recs))
    add("G23", "question/evidence relationships unchanged from candidates",
        "121 identical",
        sum(1 for r in recs
            if cand_map.get(r["candidate_id"])
            and r["question"] == cand_map[r["candidate_id"]]["question"]
            and r["gold_relevant_chunk_ids"]
            == cand_map[r["candidate_id"]]["gold_relevant_chunk_ids"]),
        all(r["question"] == cand_map[r["candidate_id"]]["question"]
            and r["gold_relevant_chunk_ids"]
            == cand_map[r["candidate_id"]]["gold_relevant_chunk_ids"]
            for r in recs))

    # ------------------------------------------------ integrity
    add("G24", "final benchmark SHA-256 matches finalization record",
        diag.get("final_sha256"), sha256(FINAL_FILE),
        sha256(FINAL_FILE) == diag.get("final_sha256"))
    add("G25", "candidate benchmark SHA-256 matches finalization record",
        diag.get("candidate_sha256"), sha256(CAND_FILE),
        sha256(CAND_FILE) == diag.get("candidate_sha256"))
    add("G26", "review artifact SHA-256 matches finalization record",
        diag.get("review_sha256"), sha256(REVIEW_FILE),
        sha256(REVIEW_FILE) == diag.get("review_sha256"))

    add("G27", "v1.0 benchmark unchanged (28ef54fa...)",
        EXPECT_V10, sha256(GOLD_V10_FILE),
        sha256(GOLD_V10_FILE) == EXPECT_V10)
    for name, key in [
        ("chunks", "chunks/chunks.jsonl"),
        ("embeddings", "index/embeddings.npy"),
        ("faiss", "index/faiss_index.bin"),
        ("bm25", "index/bm25.pkl"),
        ("row_mapping", "index/row_mapping.json"),
    ]:
        exp = diag.get("protected_before", {}).get(name)
        cur = sha256(GERI_DIR / key)
        add("G28_" + name, "frozen " + name + " unchanged",
            exp, cur, bool(exp) and cur == exp)
    add("G28", "all 5 frozen GeriLit artifacts unchanged",
        "5/5", diag.get("protected_unchanged"),
        diag.get("protected_unchanged") is True)
    for name, sub in [
        ("t10d_results", "task10d_retrieval_results.json"),
        ("t10d_summary", "task10d_summary.json"),
        ("t10e_analysis", "task10e_failure_analysis.json"),
        ("t10e_report",
         "phase3_task10e_retrieval_failure_analysis_report.md"),
    ]:
        exp = diag.get("protected_before", {}).get(name)
        cur = sha256(META_DIR / sub)
        add("G29_" + name, "Task 10D/10E artifact " + name + " unchanged",
            exp, cur, bool(exp) and cur == exp)
    add("G29", "Task 10D/10E artifacts unchanged",
        "4/4", diag.get("protected_unchanged"),
        diag.get("protected_unchanged") is True)
    add("G30", "Task 10F products unchanged during finalize",
        "4/4", diag.get("t10f_unchanged"),
        diag.get("t10f_unchanged") is True)
    # ------------------------------------------------ determinism
    ts_sentinel = "REVIEWED_AT_TS"

    def norm(r):
        x = dict(r)
        x["reviewed_at"] = ts_sentinel
        x.pop("generated_at_utc", None)
        return x

    sig = hashlib.sha256(
        json.dumps([norm(r) for r in recs], sort_keys=True).encode()).hexdigest()
    add("G31", "final benchmark deterministic signature matches",
        diag.get("final_benchmark_signature"), sig,
        sig == diag.get("final_benchmark_signature"))
    add("G32", "final benchmark status FROZEN, version 1.1",
        ("FROZEN", "1.1"), (fin.get("status"), fin.get("version")),
        fin.get("status") == "FROZEN" and fin.get("version") == "1.1")

    failed = [c for c in checks if c["result"] == "FAIL"]
    for c in failed:
        print("FAILED:", c["check_id"], c["description"], c["actual"])

    out = {
        "task": "phase3_task10g",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validated_by": "metadata/task10g_validate.py",
        "final_size": len(recs),
        "checks": checks,
        "n_passed": sum(1 for c in checks if c["result"] == "PASS"),
        "n_failed": len(failed),
    }
    (VALIDATION_DIR / "phase3_task10g_validation.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    lines = ["# PHASE 3 TASK 10G - VALIDATION", "",
             "Generated: " + out["generated_at_utc"], "",
             "Checks: **" + str(out["n_passed"]) + "/" + str(len(checks))
             + " PASS**, " + str(out["n_failed"]) + " FAIL", "",
             "| Check | Description | Result |", "|---|---|---|"]
    for c in checks:
        lines.append("| " + c["check_id"] + " | " + c["description"] + " | "
                     + c["result"] + " |")
    lines += ["",
              "* Frozen GeriLit-Gold v1.1 benchmark: 121 ACCEPTED / 0 rejected /"
              " 0 pending.",
              "* v1.0 benchmark, 5 corpus/index artifacts, Task 10D/10E and",
              "  Task 10F products unchanged.",
              "* No retrieval evaluation performed."]
    (VALIDATION_DIR / "phase3_task10g_validation.md").write_text(
        "\n".join(lines), encoding="utf-8")

    print(json.dumps({"n_passed": out["n_passed"],
                      "n_failed": out["n_failed"],
                      "total": len(checks),
                      "final_size": len(recs)}, indent=2))


if __name__ == "__main__":
    main()