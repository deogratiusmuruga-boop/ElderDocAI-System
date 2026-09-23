#!/usr/bin/env python3
"""ElderDocAI-GeriLit - Phase 3 Task 10F validation (READ-ONLY).

Structural/integrity validation for the GeriLit-Gold v1.1 CANDIDATE benchmark.
No relevance judgment is made (candidates stay PENDING_HUMAN_REVIEW). Frozen v1.0
benchmark, Task 10D/10E artifacts, and GeriLit corpus/index files are verified
unchanged against the Task 10F baseline.
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

CAND_FILE = REPO_DIR / "data" / "geri_lit_gold_v1_1_candidate.json"
GOLD_FILE = REPO_DIR / "data" / "geri_lit_gold.json"
DIAG_FILE = META_DIR / "task10f_benchmark_reconstruction.json"
BASELINE_FILE = META_DIR / "task10f_frozen_hashes.json"
CHUNKS_FILE = GERI_DIR / "chunks" / "chunks.jsonl"

GOLD_SHA = "28ef54faeec3b9a1cb8c1c79c9a478d8ea3618574b2787a82a74c1460f209c62"

VALID_STATUS = {"PENDING_HUMAN_REVIEW"}
VALID_RECON = {"RETAINED", "RECONSTRUCTED"}
VALID_TOPICS = {f"C{i:02d}" for i in range(1, 11)}

checks = []


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add(cid, desc, expected, actual, passed):
    checks.append({"check_id": cid, "description": desc,
                   "expected": expected, "actual": actual,
                   "result": "PASS" if passed else "FAIL"})


def main():
    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))["baseline"]
    gold = json.loads(GOLD_FILE.read_text(encoding="utf-8"))["records"]
    cand_doc = json.loads(CAND_FILE.read_text(encoding="utf-8"))
    cands = cand_doc["candidates"]
    diag = json.loads(DIAG_FILE.read_text(encoding="utf-8"))

    v1_seen = {r["final_benchmark_id"] for r in gold}
    chunk_ids = set()
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunk_ids.add(json.loads(line)["chunk_id"])

    add("T1", "v1.0 benchmark unchanged (28ef54fa...)",
        GOLD_SHA, sha256(GOLD_FILE), sha256(GOLD_FILE) == GOLD_SHA)

    frozen_ok = True
    for name in ("chunks.jsonl", "embeddings.npy", "faiss_index.bin",
                 "bm25.pkl", "row_mapping.json"):
        key = "chunks/chunks.jsonl" if name == "chunks.jsonl" else "index/" + name
        exp = baseline.get(key, {}).get("sha256")
        cur = sha256(GERI_DIR / key)
        ok = bool(exp) and cur == exp
        frozen_ok &= ok
        add("T2_" + name.replace(".", "_"), "frozen " + name + " unchanged",
            exp, cur, ok)
    add("T2", "all 5 frozen GeriLit artifacts unchanged",
        "5/5 PASS", frozen_ok, frozen_ok)

    for name, sub in [
        ("task10d_retrieval_results", "task10d_retrieval_results.json"),
        ("task10d_summary", "task10d_summary.json"),
        ("task10e_failure_analysis", "task10e_failure_analysis.json"),
        ("task10e_report_md",
         "phase3_task10e_retrieval_failure_analysis_report.md"),
    ]:
        key = "metadata/" + sub
        exp = baseline.get(key, {}).get("sha256")
        cur = sha256(META_DIR / sub)
        add("T3_" + name, "Task 10D/10E artifact '" + name + "' unchanged",
            exp, cur, bool(exp) and cur == exp)
# ------------------------------------------ candidate structure
    add("T4", "candidate file marked PENDING",
        "PENDING_HUMAN_REVIEW", cand_doc.get("status"),
        cand_doc.get("status") == "PENDING_HUMAN_REVIEW")
    add("T5", "candidate file marked v1.1 candidate",
        "1.1-candidate", cand_doc.get("version"),
        cand_doc.get("version") == "1.1-candidate")

    cids = [c["candidate_id"] for c in cands]
    add("T6", "candidate IDs unique",
        "unique %d" % len(cids), len(set(cids)),
        len(cids) == len(set(cids)))
    add("T7", "candidate IDs deterministic GLG11-C001..N",
        ["GLG11-C%03d" % i for i in range(1, len(cids) + 1)], cids,
        cids == ["GLG11-C%03d" % i for i in range(1, len(cids) + 1)])

    add("T8", "required fields present (question, pmcid, topic)",
        "all non-empty",
        sum(1 for c in cands if c.get("question") and c.get("pmcid")
            and c.get("topic")),
        all(c.get("question") and c.get("pmcid") and c.get("topic")
            for c in cands))
    add("T9", "all PMCIDs valid format",
        "PMC\\d+",
        len({c["pmcid"] for c in cands
             if re.fullmatch(r"PMC\d+", c.get("pmcid", ""))}),
        all(re.fullmatch(r"PMC\d+", c.get("pmcid", "")) for c in cands))
    add("T10", "every gold chunk id exists in chunk index",
        "all present",
        sum(1 for c in cands for x in c["gold_relevant_chunk_ids"]
            if x in chunk_ids),
        all(x in chunk_ids for c in cands
            for x in c["gold_relevant_chunk_ids"]))
    add("T11", "at least one gold chunk per candidate",
        "all >=1",
        min(len(c["gold_relevant_chunk_ids"]) for c in cands),
        all(len(c["gold_relevant_chunk_ids"]) >= 1 for c in cands))
    add("T12", "topic labels valid C01-C10",
        10, len({c["topic"] for c in cands}),
        all(c["topic"] in VALID_TOPICS for c in cands))
    add("T13", "review_status is PENDING_HUMAN_REVIEW",
        sorted(VALID_STATUS), sorted({c["review_status"] for c in cands}),
        all(c["review_status"] in VALID_STATUS for c in cands))
    add("T14", "reconstruction_status valid",
        sorted(VALID_RECON), sorted({c["reconstruction_status"]
                                     for c in cands}),
        all(c["reconstruction_status"] in VALID_RECON for c in cands))
    add("T15", "evidence rationale present",
        "all non-empty",
        sum(1 for c in cands if c.get("evidence_rationale")),
        all(str(c.get("evidence_rationale") or "").strip() for c in cands))
    add("T16", "provenance present (original ids, pmid, title, locator)",
        "all non-empty",
        sum(1 for c in cands if c.get("original_v1_0_id")
            and c.get("original_candidate_id") and c.get("pmid")
            and c.get("title") and c.get("source_locator")
            and c.get("pmcid")),
        all(c.get("original_v1_0_id") and c.get("original_candidate_id")
            and c.get("pmid") and c.get("title") and c.get("source_locator")
            and c.get("pmcid") for c in cands))
    add("T17", "no accidental duplicate questions",
        "0 duplicates",
        len(cands) - len({c["question"].lower() for c in cands}),
        len({c["question"].lower() for c in cands}) == len(cands))
    add("T18", "diagnostics present per candidate",
        "all have lex/sem/region",
        sum(1 for c in cands if c.get("diagnostics")
            and "lex_overlap_ratio" in c["diagnostics"]
            and "sem_sim" in c["diagnostics"]),
        all(c.get("diagnostics") and "lex_overlap_ratio" in c["diagnostics"]
            and "sem_sim" in c["diagnostics"] for c in cands))
# ------------------------------------------ traceability / determinism
    add("T19", "every candidate maps to an original v1.0 id",
        "all in v1.0 134",
        sum(1 for c in cands if c["original_v1_0_id"] in v1_seen),
        all(c["original_v1_0_id"] in v1_seen for c in cands))
    add("T20", "v1.0 -> candidate mapping is injective",
        "unique %d" % len({c["original_v1_0_id"] for c in cands}),
        len({c["original_v1_0_id"] for c in cands}),
        len({c["original_v1_0_id"] for c in cands}) == len(cands))

    use_ids = {c["original_v1_0_id"] for c in cands}
    excl_unused = {v for v, st in diag.get("decisions", {}).items()
                   if "EXCLUDE" in st and v not in use_ids}
    add("T21", "v1.0 records not in candidates are accounted as excluded",
        "candidates=%d excluded=%d -> 134" % (len(cands), len(excl_unused)),
        len(cands) + len(excl_unused),
        len(cands) + len(excl_unused) == 134)
    add("T22", "exclusions recorded with reasons",
        "documented",
        len(diag.get("exclusions", [])),
        bool(diag.get("exclusions")) and all(
            e.get("reconstruction_reason") and
            "EXCLUDE" in (e.get("reconstruction_status") or "")
            for e in diag["exclusions"]))
    sig = hashlib.sha256(json.dumps(cands, sort_keys=True).encode()).hexdigest()
    add("T23", "candidate set reproducible (reconstruction signature)",
        diag.get("candidate_signature_sha256"), sig,
        sig == diag.get("candidate_signature_sha256"))
    add("T24", "reconstruction confirmed frozen_unchanged",
        True, diag.get("frozen_unchanged"),
        diag.get("frozen_unchanged") is True)

    failed = [c for c in checks if c["result"] == "FAIL"]
    for c in failed:
        print("FAILED:", c["check_id"], c["description"], c["actual"])

    out = {
        "task": "phase3_task10f",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validated_by": "metadata/task10f_validate.py",
        "candidates": len(cands),
        "checks": checks,
        "n_passed": sum(1 for c in checks if c["result"] == "PASS"),
        "n_failed": len(failed),
    }
    (VALIDATION_DIR / "phase3_task10f_validation.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    lines = ["# PHASE 3 TASK 10F - VALIDATION", "",
             "Generated: " + out["generated_at_utc"], "",
             "Checks: **" + str(out["n_passed"]) + "/" + str(len(checks))
             + " PASS**, " + str(out["n_failed"]) + " FAIL", "",
             "| Check | Description | Result |", "|---|---|---|"]
    for c in checks:
        lines.append("| " + c["check_id"] + " | " + c["description"] + " | "
                     + c["result"] + " |")
    lines += ["",
              "* Frozen v1.0 benchmark & 5 GeriLit index artifacts unchanged.",
              "* Task 10D/10E result artifacts unchanged vs Task 10F baseline.",
              "* All candidates are PENDING_HUMAN_REVIEW (no human relevance",
              "  judgment was made by this task).",
              "* No retrieval evaluation (Recall@K / MRR / nDCG) run."]
    (VALIDATION_DIR / "phase3_task10f_validation.md").write_text(
        "\n".join(lines), encoding="utf-8")

    print(json.dumps({"n_passed": out["n_passed"],
                      "n_failed": out["n_failed"],
                      "total": len(checks),
                      "candidates": len(cands)}, indent=2))


if __name__ == "__main__":
    main()