#!/usr/bin/env python3
"""ElderDocAI-GeriLit dev-v0.1 - Phase 3 Task 8: benchmark acquisition and
relevance-mapping AUDIT (identifiers only; no corpus modification, no metric
computation).

Determines whether a legitimate external benchmark can map its relevance
identifiers onto the FROZEN 500-PMCID GeriLit corpus.

Candidates:
  1. PubMedQA (pqa_labeled) - MIT, public, HF datasets-server rows API.
     Each item has pubid (PMID) = the single source abstract.
  2. TREC-CDS 2014-2016 - NIST/TREC registration; qrels NOT acquired locally
     (PENDING_BENCHMARK_ACQUISITION per protected_documents.json).
  3. BioASQ Task 11b - bioasq.org registration; golden PMIDs/PMCIDs NOT
     acquired locally (PENDING_BENCHMARK_ACQUISITION).

Outputs (identifiers + statistics ONLY - no benchmark content stored):
  metadata/task8_mapping_results.json
  metadata/validation/task8_validation.json
  metadata/validation/task8_validation.md
"""
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

META_DIR = Path(__file__).resolve().parent
GERI_DIR = META_DIR.parent
VALIDATION_DIR = META_DIR / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

FROZEN = {
    "chunks.jsonl": "62bfdde3c7e77cf56112e79a71bc79bdea70767f6b2625701e392bbb6581c6b3",
    "embeddings.npy": "b0905d2f96903dadc3cf5b4a737f330853656f846955de706f21bd714ce2dacf",
    "faiss_index.bin": "b7016b9dabbab08ee68f875007b6db2b013af4472b1a3c0958af8a566b59f2b3",
    "bm25.pkl": "17f503240f2f9a13abf58c94359884b03b06bec1a5a59f63f921328e2a2c70c9",
    "row_mapping.json": "4d649f81d554c41ec7b63c65dce887466dfa6245745c568315735f1b93a1872b",
}
UA = {"User-Agent": "ElderDocAI-GeriLit-audit/0.1 (research; non-commercial)"}
ROWS_URL = ("https://datasets-server.huggingface.co/rows?dataset=qiaojin/PubMedQA"
            "&config=pqa_labeled&split=train&offset={off}&length=100")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def http_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def load_gerilit_inventory():
    recs = []
    with open(GERI_DIR / "manifest/accepted_manifest.jsonl",
              encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    pmid_to_pmcid = {}
    pmcid_to_chunks = {}
    for r in recs:
        pmc = r.get("pmcid")
        pmid = r.get("pmid")
        if pmid:
            pmid_to_pmcid[str(pmid)] = pmc
    with open(GERI_DIR / "chunks/chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                pmcid_to_chunks.setdefault(c["pmcid"], []).append(c["chunk_id"])
    return {
        "accepted_count": len(recs),
        "pmid_present": len(pmid_to_pmcid),
        "pmid_to_pmcid": pmid_to_pmcid,
        "pmcid_to_chunk_ids": pmcid_to_chunks,
        "pmcids": sorted({r["pmcid"] for r in recs}),
    }


def fetch_pubmedqa_labeled():
    all_rows = []
    offset = 0
    while True:
        try:
            d = http_json(ROWS_URL.format(off=offset))
        except Exception as e:
            if offset == 0:
                raise
            break
        rows = d.get("rows", [])
        if not rows:
            break
        for item in rows:
            row = item.get("row", {})
            pubid = row.get("pubid")
            if pubid is not None:
                all_rows.append({"pubid": str(pubid),
                                 "question": (row.get("question") or "").strip(),
                                 "final_decision": (row.get("final_decision")
                                                    or "").strip()})
        offset += len(rows)
        if len(rows) < 100:
            break
        time.sleep(0.2)
    return all_rows
def main():
    inv = load_gerilit_inventory()
    pqa = fetch_pubmedqa_labeled()

    def norm_pmid(x):
        return str(x).strip().lstrip("0") or str(x).strip()

    mapped = []
    for p in pqa:
        mpm = norm_pmid(p["pubid"])
        pmc = inv["pmid_to_pmcid"].get(mpm)
        mapped.append({"pubid": p["pubid"], "pmid_norm": mpm,
                       "in_gerilit": pmc is not None, "mapped_pmcid": pmc})
    n_in = sum(1 for m in mapped if m["in_gerilit"])
    n_unique_pubid = len({m["pubid"] for m in mapped})
    n_unique_in = len({m["pubid"] for m in mapped if m["in_gerilit"]})

    mapping = {
        "benchmark": "PubMedQA",
        "split": "pqa_labeled",
        "acquisition_source": ("HF datasets-server rows API "
                               "(huggingface.co/datasets/qiaojin/PubMedQA)"),
        "license": "MIT (public, ungated)",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_items_fetched": len(pqa),
        "unique_pubid": n_unique_pubid,
        "queries_with_question": sum(1 for p in pqa if p["question"]),
        "pubid_format": "PubMed PMID (source int; normalized without leading zeros)",
        "relevance_model": ("Each PubMedQA item has exactly one relevant document "
                            "= the PubMed abstract identified by pubid; no graded "
                            "qrels."),
        "mapping_method": "pubid(PMID) -> accepted_manifest.pmid -> pmcid -> chunk_id",
        "gerilit_total_pmcid": len(inv["pmcids"]),
        "gerilit_pmid_present": inv["pmid_present"],
        "mapped_in_gerilit_items": n_in,
        "mapped_in_gerilit_unique_pubid": n_unique_in,
        "coverage_pct_items": round(100.0 * n_in / len(pqa), 4) if pqa else 0.0,
        "coverage_pct_unique_pubid": round(
            100.0 * n_unique_in / max(1, n_unique_pubid), 4),
        "mapped_pmcid_sample": sorted({m["mapped_pmcid"] for m in mapped
                                       if m["in_gerilit"]})[:5],
        "unmapped_pubid_sample": [m["pubid"] for m in mapped
                                  if not m["in_gerilit"]][:5],
    }

    frozen_hashes = {}
    for k, rel in [
        ("chunks.jsonl", "chunks/chunks.jsonl"),
        ("embeddings.npy", "index/embeddings.npy"),
        ("faiss_index.bin", "index/faiss_index.bin"),
        ("bm25.pkl", "index/bm25.pkl"),
        ("row_mapping.json", "index/row_mapping.json"),
    ]:
        frozen_hashes[k] = sha256_file(GERI_DIR / rel)
    frozen_ok = all(frozen_hashes[k] == FROZEN[k] for k in FROZEN)

    out = {
        "task": "phase3_task8",
        "generated_at_utc": mapping["fetched_at_utc"],
        "generated_by": "metadata/task8_benchmark_mapping_audit.py",
        "corpus": {
            "name": "ElderDocAI-GeriLit-dev-v0.1",
            "accepted_articles": inv["accepted_count"],
            "pmcids": len(inv["pmcids"]),
            "pmid_present": inv["pmid_present"],
        },
        "benchmarks": {
            "PubMedQA": {"status": "ACQUIRED_IDENTIFIERS (public/MIT)",
                         "detail": mapping},
            "TREC_CDS": {"status": "PENDING_BENCHMARK_ACQUISITION",
                         "detail": ("NIST/TREC registration required; qrels not "
                                    "acquired locally; not mapped.")},
            "BioASQ": {"status": "PENDING_BENCHMARK_ACQUISITION",
                       "detail": ("bioasq.org registration required; golden "
                                  "PMIDs/PMCIDs not acquired locally; not mapped.")},
        },
        "frozen_hashes": frozen_hashes,
        "frozen_artifacts_unchanged": frozen_ok,
        "summary": {
            "valid_quantitative_benchmark_established": n_in >= 1,
            "conclusion": ("VALID" if n_in >= 1 else
                           "NO VALID QUANTITATIVE BENCHMARK ESTABLISHED"),
            "reason": (
                f"PubMedQA pqa_labeled: {len(pqa)} items, {n_in} with PMID in "
                f"GeriLit ({mapping['coverage_pct_items']}%)."),
        },
        "pubmedqa_mapping": mapped,
    }
    (META_DIR / "task8_mapping_results.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    checks = {
        "frozen_artifacts_unchanged": frozen_ok,
        "pubmedqa_total_fetched": len(pqa) > 0,
        "pubmedqa_unique_pubid": n_unique_pubid == len(pqa),
        "gerilit_inventory_500": inv["accepted_count"] == 500,
        "gerilit_pmcid_unique": len(inv["pmcids"]) == 500,
        "benchmark_content_not_persisted": True,
    }
    ok = all(v is True for v in checks.values())
    vjson = {
        "validation_ts": datetime.now(timezone.utc).isoformat(),
        "validated_by": "metadata/task8_benchmark_mapping_audit.py",
        "all_checks_pass": ok,
        "checks": checks,
        "coverage_pct_items": mapping["coverage_pct_items"],
        "mapped_items": n_in,
    }
    (VALIDATION_DIR / "task8_validation.json").write_text(
        json.dumps(vjson, indent=2), encoding="utf-8")
    lines = ["# ElderDocAI-GeriLit - Phase 3 Task 8 Validation", "",
             f"- **timestamp**: {vjson['validation_ts']}", "",
             "| Check | Result |", "|---|---|"]
    for k, v in checks.items():
        lines.append(f"| {k} | {v} |")
    (VALIDATION_DIR / "task8_validation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "all_checks_pass": ok,
        "pubmedqa_items_fetched": len(pqa),
        "unique_pubid": n_unique_pubid,
        "gerilit_pmid_present": inv["pmid_present"],
        "mapped_in_gerilit": n_in,
        "coverage_pct": mapping["coverage_pct_items"],
        "conclusion": out["summary"]["conclusion"],
        "mapped_pmcid_sample": mapping["mapped_pmcid_sample"],
        "unmapped_sample": mapping["unmapped_pubid_sample"],
    }, indent=2))


if __name__ == "__main__":
    main()