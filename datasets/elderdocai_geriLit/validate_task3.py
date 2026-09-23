#!/usr/bin/env python3
"""ElderDocAI-GeriLit dev-v0.1 - Phase 3 Task 3 validation (18 checks).

Validates JATS-enriched metadata, section inventory, structural audit,
evidence-type enrichment, corpus integrity, determinism, and absence of
mutation of Task-2 artifacts and raw files. No chunking / no RAG.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(r"C:\Users\chosun\Documents\ElderDocAI-System\datasets\elderdocai_geriLit")
META = BASE / "metadata"
MAN = BASE / "manifest"
RAW = BASE / "raw"
VAL = META / "validation"
VAL.mkdir(parents=True, exist_ok=True)

ET_CV = [
    "GUIDELINE_OR_CONSENSUS", "SYSTEMATIC_REVIEW", "META_ANALYSIS",
    "RANDOMIZED_CONTROLLED_TRIAL", "OBSERVATIONAL_STUDY",
    "DIAGNOSTIC_OR_VALIDATION_STUDY", "QUALITATIVE_OR_MIXED_METHODS",
    "NARRATIVE_REVIEW", "PROTOCOL", "METHODS_OR_TECHNICAL",
    "EDITORIAL_OR_COMMENTARY", "CASE_REPORT_OR_CASE_SERIES", "OTHER", "UNKNOWN",
]
CONF_CV = ["HIGH", "MEDIUM", "LOW"]

EXPECTED_SHA = {
    "jats_enriched_metadata.json":
        "655f1f4a63e64a0d55682a8d207e2059393b756b77ee8abac182b7ab1dad7825",
    "jats_section_inventory.json":
        "1dac58f9b5d7bf481e1b8a13d8a6f1d9ed119bdbb8f99d4b574a9d5b4b087e7c",
    "jats_structural_audit.json":
        "59799ac33474b589f505ddbd999a1166b515d0d07c1eef8ab756f4be9d87fc5c",
    "evidence_type_enrichment.json":
        "670ee5fbcf33c597256b287e82f073b56719b05320d92d05c17df79a01620568",
}
ACCEPTED_MANIFEST_SHA = ("f80cd457e5e2be55ed3a3de09e2554e9d585e76cef8dc44800270dfa66a99a49")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_json(name):
    return json.loads((META / name).read_text(encoding="utf-8"))
def main():
    checks = {}
    accepted = [json.loads(l) for l in
                (MAN / "accepted_manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()]
    enriched = load_json("jats_enriched_metadata.json")
    sections = load_json("jats_section_inventory.json")
    audits = load_json("jats_structural_audit.json")
    evs = load_json("evidence_type_enrichment.json")

    manifest_pmcids = [r["pmcid"] for r in accepted]
    checks["accepted_count_is_500"] = len(manifest_pmcids) == 500
    checks["enriched_metadata_count_is_500"] = len(enriched) == 500
    checks["section_inventory_count_is_500"] = len(sections) == 500
    checks["evidence_type_enrichment_count_is_500"] = len(evs) == 500
    four_sets = (set(manifest_pmcids) == set(enriched.keys()) ==
                 set(sections.keys()) == set(audits.keys()) == set(evs.keys()))
    checks["pmcid_set_identity_across_artifacts"] = (
        len(set(manifest_pmcids)) == 500 and four_sets)
    checks["no_accepted_pmcid_added"] = set(enriched.keys()) == set(manifest_pmcids)
    checks["no_accepted_pmcid_removed"] = set(manifest_pmcids) == set(enriched.keys())
    current_sha = {n: sha(META / n) for n in EXPECTED_SHA}
    checks["deterministic_output_hashes_match"] = current_sha == EXPECTED_SHA
    n_records = {"jats_enriched_metadata.json": len(enriched),
                 "jats_section_inventory.json": len(sections),
                 "jats_structural_audit.json": len(audits),
                 "evidence_type_enrichment.json": len(evs)}
    checks["recorded_hashes"] = {n: {"sha256": current_sha[n],
                                     "size_bytes": (META / n).stat().st_size,
                                     "records": n_records[n]}
                                 for n in EXPECTED_SHA}
    empty_titles = [p for p, b in enriched.items()
                    if not (b.get("article") or {}).get("title")]
    checks["no_empty_required_title"] = len(empty_titles) == 0
    parse_fail = [p for p, a in audits.items() if not a["xml_parse_success"]]
    checks["xml_parse_success_all"] = len(parse_fail) == 0
    bad_labels = [p for p, e in evs.items() if e.get("evidence_type") not in ET_CV]
    checks["evidence_type_labels_in_cv"] = len(bad_labels) == 0
    bad_conf = [p for p, e in evs.items()
                if e.get("evidence_type_confidence") not in (CONF_CV + [None])]
    checks["confidence_labels_valid"] = len(bad_conf) == 0
    checks["unknown_allowed_and_present"] = any(
        e.get("evidence_type") == "UNKNOWN" for e in evs.values())
    no_reason = [p for p, e in evs.items()
                 if e.get("evidence_type") != "UNKNOWN"
                 and not e.get("evidence_type_reason")]
    checks["reasons_exist_for_non_unknown"] = len(no_reason) == 0
    dup_sec = [p for p, a in audits.items()
               if any("duplicate_section_ids" in (i.get("code") or "")
                      for i in a["issues"])]
    checks["no_duplicate_section_ids_expected"] = len(dup_sec) == 0
    bad_order = []
    for p, b in enriched.items():
        orders = [a["order"] for a in b["authors"]]
        if orders and list(orders) != list(range(1, len(orders) + 1)):
            bad_order.append(p)
    checks["author_order_preserved"] = len(bad_order) == 0
    act_sha = sha(MAN / "accepted_manifest.jsonl")
    checks["task2_accepted_manifest_unchanged"] = act_sha == ACCEPTED_MANIFEST_SHA
    checks["accepted_manifest_sha"] = act_sha
    raw_changed = []
    for r in accepted:
        raw_file = r.get("raw_file")
        src = r.get("source_id")
        expected = r.get("raw_sha256")
        p = RAW / src / raw_file if src and raw_file else None
        if p is None or not p.exists() or sha(p) != expected:
            raw_changed.append(r["pmcid"])
    checks["no_raw_file_modified"] = len(raw_changed) == 0
    checks["raw_changed_ids"] = raw_changed[:10]
    missing_fields = [p for p, e in evs.items()
                      if "evidence_type" not in e
                      or "evidence_type_confidence" not in e
                      or "evidence_type_reason" not in e
                      or "evidence_type_source" not in e]
    checks["evidence_fields_present"] = len(missing_fields) == 0
    checks["missing_evidence_field_ids"] = missing_fields[:10]
    booleans = {k: v for k, v in checks.items()
                if isinstance(v, bool) and not k.endswith("_ids")}
    out = {
        "validation_ts": datetime.now(timezone.utc).isoformat(),
        "validated_by": "validate_task3.py",
        "all_checks_pass": all(booleans.values()),
        "checks": checks,
        "n_boolean_checks": len(booleans),
    }
    with open(VAL / "phase3_task3_validation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    md = ["# ElderDocAI-GeriLit Dev Corpus - Phase 3 Task 3 Validation", ""]
    for k, v in checks.items():
        md.append("- **%s**: %s" % (k, v))
    md.append("")
    (VAL / "phase3_task3_validation.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"all_checks_pass": out["all_checks_pass"],
                      "n_boolean_checks": len(booleans)}, indent=2))
    return out


if __name__ == "__main__":
    main()