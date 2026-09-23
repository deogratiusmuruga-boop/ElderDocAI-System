#!/usr/bin/env python3
"""ElderDocAI-GeriLit dev-v0.1 - Task 2 validation (Task 2P + content integrity 2Q).

Validates manifests/statistics/raw integrity. No chunking / no RAG.
"""
import json
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(r"C:\Users\chosun\Documents\ElderDocAI-System\datasets\elderdocai_geriLit")
META = BASE / "metadata"
MAN = BASE / "manifest"
RAW = BASE / "raw"
VAL = META / "validation"
VAL.mkdir(parents=True, exist_ok=True)

ALLOWED_STATUS = {"ACCEPTED", "EXCLUDED", "RESERVE"}
ALLOWED_LIC = {"PERMISSIVE", "RESTRICTED", "UNKNOWN"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def content_check(meta):
    pid = meta["pmcid"]
    v = meta["raw_file"]
    path = None
    if v:
        d = meta.get("source_id") or pid
        path = RAW / d / v
    res = {"pmcid": pid, "file_exists": bool(path and path.exists())}
    if not res["file_exists"]:
        res["ok"] = False
        res["error"] = "missing_file"
        return res
    b = path.stat().st_size
    res["bytes"] = b
    res["nonzero"] = b > 0
    xml = path.read_text(encoding="utf-8", errors="replace")
    res["has_article"] = "<article" in xml
    res["has_pmcid_in_text"] = bool(re.search(r"PMC\s*\d+", xml))
    res["no_html_error_page"] = not bool(
        re.search(r"<!DOCTYPE html|<html\b|<head\b|Access Denied|404 Not Found",
                  xml, re.I))
    t_xml = re.search(r"<article-title>(.*?)</article-title>", xml, re.S | re.I)
    t_xml = re.sub(r"<[^>]+>", " ", t_xml.group(1)).strip() if t_xml else ""
    t_meta = (meta.get("title") or "").strip()
    nt = lambda s: re.sub(r"[^a-z0-9]+", " ", re.sub(r"&[a-z#0-9]+;", " ", s.lower())).strip()
    res["title_match_norm"] = (nt(t_xml)[:80] == nt(t_meta)[:80]) if t_xml and t_meta else None
    res["ok"] = all((res["nonzero"], res["has_article"], res["has_pmcid_in_text"],
                     res["no_html_error_page"]))
    return res


def main():
    accepted = [json.loads(l) for l in
                (MAN / "accepted_manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    excluded = [json.loads(l) for l in
                (MAN / "excluded_manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    stats = json.loads((META / "corpus_statistics.json").read_text(encoding="utf-8"))
    taxo = json.loads((META / "topic_taxonomy.json").read_text(encoding="utf-8"))
    protected = json.loads((META / "protected_documents.json").read_text(encoding="utf-8"))
    topics = {t["topic_id"] for t in taxo["topics"]}

    checks = {}
    checks["corpus_version_exists"] = bool(stats.get("corpus_version"))
    checks["accepted_count"] = len(accepted)
    checks["accepted_unique_source_id"] = len({m["source_id"] for m in accepted}) == len(accepted)
    checks["accepted_unique_pmcid"] = len({m["pmcid"] for m in accepted}) == len(accepted)
    pmids = [m["pmid"] for m in accepted if m["pmid"]]
    dois = [m["doi"] for m in accepted if m["doi"]]
    checks["accepted_unique_pmid"] = len(pmids) == len(set(pmids))
    checks["accepted_unique_doi"] = len(dois) == len(set(dois))
    checks["valid_inclusion_status"] = all(m["inclusion_status"] in ALLOWED_STATUS for m in accepted)
    checks["valid_license_categories"] = all(m["license_category"] in ALLOWED_LIC for m in accepted)
    bad_lic = [m["pmcid"] for m in accepted if m["license_category"] not in ("PERMISSIVE",)]
    checks["permissive_only_accepted"] = len(bad_lic) == 0
    protected_ids = set()
    for b in protected.get("benchmarks", []):
        protected_ids.update(b.get("protected_ids") or [])
    protected_in_corpus = [m["pmcid"] for m in accepted if m["pmcid"] in protected_ids]
    checks["benchmark_protected_ids_count"] = len(protected_ids)
    checks["benchmark_protected_excluded_from_corpus"] = len(protected_in_corpus) == 0
    checks["topic_taxonomy_valid"] = all(
        m["topic_ids"] and all(t in topics for t in m["topic_ids"]) for m in accepted)
    checks["publication_year_valid"] = all(
        (isinstance(m["publication_year"], int) and 1900 <= m["publication_year"] <= 2100)
        or m["publication_year"] is None for m in accepted)
    res = [content_check(m) for m in accepted]
    checks["content_integrity_ok"] = sum(1 for r in res if r["ok"]) == len(accepted)
    checks["content_zero_bytes"] = sum(1 for r in res if not r.get("nonzero", False))
    checks["content_missing_file"] = sum(1 for r in res if not r["file_exists"])
    checks["content_html_error_page"] = sum(1 for r in res if not r.get("no_html_error_page", False))
    title_mismatch = [r["pmcid"] for r in res if r.get("title_match_norm") is False]
    checks["content_title_mismatch"] = len(title_mismatch)
    checks["content_title_mismatch_ids"] = title_mismatch[:10]
    checks["unknown_license_in_accepted"] = sum(1 for m in accepted if m["license_category"] == "UNKNOWN")
    checks["excluded_have_reason"] = all(
        (e.get("reason") or e.get("exclusion_reason")) for e in excluded)
    checks["manifests_sha256"] = {
        "accepted": sha(MAN / "accepted_manifest.jsonl"),
        "excluded": sha(MAN / "excluded_manifest.jsonl"),
        "statistics": sha(META / "corpus_statistics.json")}
    return checks, res


if __name__ == "__main__":
    checks, content = main()
    out = {
        "validation_ts": datetime.now(timezone.utc).isoformat(),
        "validated_by": "validate_task2.py",
        "all_checks_pass": all(v is not False for v in checks.values()),
        "checks": checks,
        "content_integrity_summary": {
            "n_checked": len(content),
            "n_ok": sum(1 for r in content if r["ok"]),
            "n_failed": sum(1 for r in content if not r["ok"]),
        },
    }
    with open(VAL / "task2_validation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    md = ["# ElderDocAI-GeriLit Dev Corpus — Task 2 Validation", ""]
    for k, v in checks.items():
        md.append(f"- **{k}**: {v}")
    md.append("")
    (VAL / "task2_validation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))