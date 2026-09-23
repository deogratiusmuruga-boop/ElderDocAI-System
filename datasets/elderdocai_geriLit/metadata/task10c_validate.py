#!/usr/bin/env python3
"""ElderDocAI-GeriLit - Phase 3 Task 10C validation suite.

Runs controlled deterministic checks over the Task 10B integration without
model/network dependence (router selection, adapter, fallback, both-mode,
invalid-backend) and verifies frozen-artifact hashes + source diff scoping.

Checks are plain and reproducible; each records
check_id/description/expected/actual/pass.
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent.parent.parent
GERI_DIR = REPO_DIR / "datasets" / "elderdocai_geriLit"
VALIDATION_DIR = Path(__file__).resolve().parent / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

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
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO_DIR), *args],
                                   text=True, errors="replace")


CHECKS = []


def record(check_id, desc, expected, actual):
    passed = (expected == actual)
    CHECKS.append({
        "check_id": check_id, "description": desc,
        "expected": expected, "actual": actual,
        "pass": passed,
    })
def main():
    from unittest import mock as _u_mock
    import scripts.retrieval_router as router
    import scripts.geri_lit_adapter as ad

    env_backup = os.environ.pop(router.BACKEND_ENV, None)

    record("C01", "default backend legacy (env absent)",
           "legacy", router._enabled_backend(None))
    record("C02", "explicit legacy", "legacy", router._enabled_backend("legacy"))
    record("C03", "explicit geri_lit", "geri_lit",
           router._enabled_backend("geri_lit"))
    record("C04", "both backend allowed", "both", router._enabled_backend("both"))
    record("C05", "invalid backend -> legacy", "legacy",
           router._enabled_backend("INVALID_VALUE"))

    with _u_mock.patch("scripts.hybrid_retriever.hybrid_search",
                       return_value=[{"chunk_id": "legacy-1"}]) as m, \
         _u_mock.patch.object(router, "_geri_lit_retrieve") as gr:
        out = router.retrieve("q", backend="both", fallback=False)
        record("C06", "both -> legacy-only (no fusion)", ["legacy-1"],
               [x["chunk_id"] for x in out])
        record("C07", "both never invokes geri_lit path", False,
               gr.called)

    with _u_mock.patch.object(router, "_geri_lit",
                              side_effect=RuntimeError("boom")), \
         _u_mock.patch("scripts.hybrid_retriever.hybrid_search",
                       return_value=[{"chunk_id": "legacy-fb"}]) as hs:
        out = router.retrieve("q", backend="geri_lit", fallback=True)
        record("C08", "geri_lit failure -> legacy fallback", ["legacy-fb"],
               [x["chunk_id"] for x in out])

    with _u_mock.patch.object(router, "_geri_lit",
                              side_effect=RuntimeError("boom")):
        out = router.retrieve("q", backend="geri_lit", fallback=False)
        record("C09", "no-fallback -> empty evidence", [], out)

    sample = [{"chunk_id": "X", "pmcid": "PMC1", "pmid": "1",
               "doi": "10.x/y", "title": "T", "journal": "J",
               "publication_year": 2020, "region": "BODY",
               "section_id": "s", "section_title": "Intro",
               "content_type": "PROSE", "evidence_type": "RCT",
               "source_locator": "PMC1:BODY:s:0..1", "text": "Evidence text.",
               "dense_score": .8, "sparse_score": .2, "hybrid_score": .5,
               "rerank_score": .9, "source_document": "PMC1 (J, 2020)",
               "row_id": 0}]
    a1 = ad.to_legacy_evidence(sample)[0]
    a2 = ad.to_legacy_evidence(sample)[0]
    record("C10", "adapter deterministic", True, a1 == a2)
    record("C11", "rerank->similarity", .9, a1["similarity_score"])
    record("C12", "authority explicit", ad.AUTHORITY_PMC,
           a1["authority_score"])
    record("C13", "category PMC", "PMC", a1["document_category"])
    record("C14", "provenance preserved", sample[0]["source_locator"],
           a1["source_locator"])
    record("C15", "input not mutated", .9, sample[0]["rerank_score"])

    frozen = {}
    for k, rel in [("chunks.jsonl", "chunks/chunks.jsonl"),
                   ("embeddings.npy", "index/embeddings.npy"),
                   ("faiss_index.bin", "index/faiss_index.bin"),
                   ("bm25.pkl", "index/bm25.pkl"),
                   ("row_mapping.json", "index/row_mapping.json")]:
        frozen[k] = sha256_file(GERI_DIR / rel)
    record("C16", "frozen GeriLit unchanged", True,
           all(frozen[k] == FROZEN[k] for k in FROZEN))
    record("C17", "Gold16 unchanged", True,
           sha256_file(REPO_DIR / "data/gold_qa_evaluation.json") == GOLD16_SHA)
    record("C18", "Gold96 unchanged", True,
           sha256_file(REPO_DIR / "data/gold_qa_extended.json") == GOLD96_SHA)
    record("C19", "accepted_manifest unchanged", True,
           sha256_file(GERI_DIR / "manifest/accepted_manifest.jsonl")
           == ACC_MAN_SHA)

    diff0 = git("diff", "--unified=0", "--", "scripts/rag_chat.py")
    changed_lines = [ln for ln in diff0.splitlines()
                     if (ln.startswith("+") or ln.startswith("-"))
                     and not ln.startswith(("+++", "---"))
                     and not ln.startswith(("diff ", "index ", "@@"))]
    record("C20", "rag_chat diff = 4 changed lines", 4, len(changed_lines))
    record("C21", "new router/adapter exist", True,
           (REPO_DIR / "scripts/retrieval_router.py").exists()
           and (REPO_DIR / "scripts/geri_lit_adapter.py").exists())
    staged = git("diff", "--cached", "--name-only").strip()
    record("C22", "nothing staged", "", staged)

    if env_backup is not None:
        os.environ[router.BACKEND_ENV] = env_backup

    ok = all(c["pass"] for c in CHECKS)
    out = {
        "validation_ts": datetime.now(timezone.utc).isoformat(),
        "validated_by": "metadata/task10c_validate.py",
        "all_checks_pass": ok,
        "n_checks": len(CHECKS),
        "checks": CHECKS,
    }
    (VALIDATION_DIR / "phase3_task10c_validation.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    lines = ["# ElderDocAI-GeriLit Phase 3 Task 10C - Validation", "",
             f"- **timestamp**: {out['validation_ts']}", "",
             "| ID | Description | Expected | Actual | PASS |",
             "|---|---|---|---|--|"]
    for c in CHECKS:
        lines.append("| %s | %s | %s | %s | %s |" % (
            c["check_id"], c["description"], c["expected"],
            c["actual"], c["pass"]))
    (VALIDATION_DIR / "phase3_task10c_validation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"all_checks_pass": ok, "n_checks": len(CHECKS)},
                     indent=2))


if __name__ == "__main__":
    main()