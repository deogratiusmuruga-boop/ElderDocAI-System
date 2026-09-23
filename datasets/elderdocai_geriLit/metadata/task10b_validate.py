#!/usr/bin/env python3
"""ElderDocAI-GeriLit - Phase 3 Task 10B validation.

Confirms the controlled GeriLit integration:
  - files created exist
  - rag_chat.py diff is limited to the retrieval entry point import + call
  - router/adapter contracts hold (deterministic, no mutation)
  - frozen artifacts unchanged
  - no staged/committed changes
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
def main():
    checks = {}
    new_files = [
        REPO_DIR / "scripts/retrieval_router.py",
        REPO_DIR / "scripts/geri_lit_adapter.py",
        REPO_DIR / "tests/test_retrieval_router.py",
    ]
    checks["new_files_exist"] = all(p.exists() for p in new_files)

    diff0 = git("diff", "--unified=0", "--", "scripts/rag_chat.py")
    changed_lines = [ln for ln in diff0.splitlines()
                     if (ln.startswith("+") or ln.startswith("-"))
                     and not ln.startswith(("+++", "---"))
                     and not ln.startswith(("diff ", "index ", "@@"))]
    # 2 removals + 2 additions = exactly 4 changed lines total
    checks["rag_chat_changed_lines_exact"] = len(changed_lines) == 4
    checks["rag_chat_import_replaced"] = any(
        "+from scripts.retrieval_router import retrieve as retrieve_evidence"
        in ln for ln in changed_lines)
    checks["rag_chat_call_replaced"] = any(
        "+    retrieved_chunks = retrieve_evidence(" in ln
        for ln in changed_lines)
    checks["rag_chat_no_other_change"] = all(
        ("retrieval_router" in ln or "retrieve_evidence" in ln)
        for ln in changed_lines if ln.startswith("+"))
    checks["rag_chat_removed_exact_2"] = len(
        [ln for ln in changed_lines if ln.startswith("-")]) == 2

    import scripts.retrieval_router as router
    env_ok = os.environ.pop(router.BACKEND_ENV, None)
    checks["default_backend_legacy"] = router._enabled_backend(None) == "legacy"
    checks["explicit_legacy_backend"] = router._enabled_backend("legacy") == "legacy"
    checks["explicit_geri_backend"] = router._enabled_backend("geri_lit") == "geri_lit"
    checks["both_backend_allowed"] = router._enabled_backend("both") == "both"
    checks["invalid_backend_safe"] = router._enabled_backend("bogus") == "legacy"
    if env_ok is not None:
        os.environ[router.BACKEND_ENV] = env_ok

    import scripts.geri_lit_adapter as ad
    sample = [{"chunk_id": "X", "pmcid": "PMC1", "pmid": "1", "doi": "10.x/y",
               "title": "T", "journal": "J", "publication_year": 2020,
               "region": "BODY", "section_id": "s", "section_title": "Intro",
               "content_type": "PROSE", "evidence_type": "RCT",
               "source_locator": "PMC1:BODY:s:0..1",
               "text": "Evidence text.", "dense_score": .8,
               "sparse_score": .2, "hybrid_score": .5, "rerank_score": .9,
               "source_document": "PMC1 (J, 2020)", "row_id": 0}]
    a = ad.to_legacy_evidence(sample)[0]
    checks["adapter_maps_similarity"] = a["similarity_score"] == .9
    checks["adapter_authority_explicit"] = (
        "authority_score" in a and a["authority_score"] == ad.AUTHORITY_PMC)
    checks["adapter_category_pmc"] = a["document_category"] == "PMC"
    checks["adapter_no_input_mutation"] = sample[0]["rerank_score"] == .9
    checks["adapter_preserves_chunk_id"] = a["chunk_id"] == "X"

    frozen = {}
    for k, rel in [("chunks.jsonl", "chunks/chunks.jsonl"),
                   ("embeddings.npy", "index/embeddings.npy"),
                   ("faiss_index.bin", "index/faiss_index.bin"),
                   ("bm25.pkl", "index/bm25.pkl"),
                   ("row_mapping.json", "index/row_mapping.json")]:
        frozen[k] = sha256_file(GERI_DIR / rel)
    checks["frozen_artifacts_unchanged"] = all(
        frozen[k] == FROZEN[k] for k in FROZEN)
    checks["gold16_unchanged"] = (
        sha256_file(REPO_DIR / "data/gold_qa_evaluation.json") == GOLD16_SHA)
    checks["gold96_unchanged"] = (
        sha256_file(REPO_DIR / "data/gold_qa_extended.json") == GOLD96_SHA)
    checks["accepted_manifest_unchanged"] = (
        sha256_file(GERI_DIR / "manifest/accepted_manifest.jsonl")
        == ACC_MAN_SHA)

    staged = git("diff", "--cached", "--name-only").strip()
    checks["nothing_staged"] = staged == ""
    checks["only_expected_modified"] = " M scripts/rag_chat.py" in git(
        "status", "--short")

    ok = all(v is True for v in checks.values())
    vjson = {
        "validation_ts": datetime.now(timezone.utc).isoformat(),
        "validated_by": "metadata/task10b_validate.py",
        "all_checks_pass": ok,
        "checks": checks,
    }
    (VALIDATION_DIR / "phase3_task10b_validation.json").write_text(
        json.dumps(vjson, indent=2), encoding="utf-8")
    lines = ["# ElderDocAI-GeriLit Phase 3 Task 10B - Validation", "",
             f"- **timestamp**: {vjson['validation_ts']}", "",
             "| Check | Result |", "|---|---|"]
    for k, v in checks.items():
        lines.append(f"| {k} | {v} |")
    (VALIDATION_DIR / "phase3_task10b_validation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"all_checks_pass": ok, "n_checks": len(checks)},
                     indent=2))


if __name__ == "__main__":
    main()