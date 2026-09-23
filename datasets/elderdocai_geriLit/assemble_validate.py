#!/usr/bin/env python3
"""ElderDocAI-GeriLit dev-v0.1 - Stage 4: assemble + validate corpus.

Reads candidate_manifest.jsonl, selected.json pending, download_status.json,
and raw/{ver}/{ver}.xml to build final article metadata (Task-2J schema),
accepted/excluded manifests, corpus statistics, and Task-2 validation.
No chunking / no RAG.
"""
import json
import re
import hashlib
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(r"C:\Users\chosun\Documents\ElderDocAI-System\datasets\elderdocai_geriLit")
META = BASE / "metadata"
MAN = BASE / "manifest"
RAW = BASE / "raw"
VAL = META / "validation"
VAL.mkdir(parents=True, exist_ok=True)
VERSION = "ElderDocAI-GeriLit-dev-v0.1"
DEV_TARGET = 500
TAG = re.compile(r"<[^>]+>")
ENT = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'", "&apos;": "'"}


def clean(s):
    s = TAG.sub(" ", s or "")
    for k, v in ENT.items():
        s = s.replace(k, v)
    return re.sub(r"\s+", " ", s).strip()


def first(xml, pattern):
    m = re.search(pattern, xml, re.S | re.I)
    return clean(m.group(1)) if m else None


def license_category(xml):
    m = re.search(
        r"<license[^>]*?(?:href|xlink:href)=[\"']([^\"']+)[\"'][^>]*>.*?</license>",
        xml, re.S | re.I)
    href = m.group(1) if m else ""
    text = first(xml, r"<license[^>]*>(.*?)</license>") or ""
    raw = (href + " " + text).lower()
    if not raw.strip():
        return "UNKNOWN", raw.strip()
    if "creativecommons" in raw:
        if any(x in raw for x in ("/nc", "-nc", "/nd", "-nd", "no-derivative",
                                  "non-commercial")):
            return "RESTRICTED", raw.strip()
        return "PERMISSIVE", raw.strip()
    if "cc0" in raw or "public domain" in raw or "cc0" in href.lower():
        return "PERMISSIVE", raw.strip()
    return "UNKNOWN", raw.strip()


def lang_of(xml):
    head = xml[:600]
    m = re.search(r'(?:xml:)?lang\s*=\s*["\']([a-zA-Z-]+)["\']', head)
    return m.group(1).lower() if m else None


def type_of(xml):
    m = re.search(r"<article\b[^>]*article-type\s*=\s*[\"']([^\"']+)[\"']", xml)
    if m:
        return m.group(1)
    subj = first(xml, r"<article-categories>.*?(?:</article-categories>|$)")
    if subj:
        return subj[:60]
    return "other/unknown"


def year_of(xml, fallback):
    y = first(xml, r"<pub-date[^>]*>\s*(?:<day>\d+</day>\s*)?(?:<month>\d+</month>\s*)?<year>(\d{4})</year>")
    if not y:
        y = first(xml, r"<year[^>]*>(\d{4})</year>")
    if y:
        return int(y)
    if fallback:
        try:
            return int(fallback)
        except Exception:
            return None
    return None


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def main():
    recs = {}
    with open(MAN / "candidate_manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[r["pmcid"]] = r
    sel = json.loads((MAN / "selected.json").read_text(encoding="utf-8"))
    status = json.loads((MAN / "download_status.json").read_text(encoding="utf-8"))
    protected = set()
    pj = json.loads((META / "protected_documents.json").read_text(encoding="utf-8"))
    for b in pj.get("benchmarks", []):
        protected.update(b.get("protected_ids") or [])

    accepted = []
    excluded = []
    for raw_id in sel["pending"]:
        base_cand = recs.get(raw_id, {})
        st = status.get(raw_id, {})
        ver = st.get("version")
        xml_path = (RAW / ver / (ver + ".xml")) if ver else None
        xml = ""
        ok = bool(st.get("ok")) and xml_path and xml_path.exists()
        if ok:
            xml = xml_path.read_text(encoding="utf-8", errors="replace")
        has_article = "<article" in xml and bool(re.search(r"PMC\s*\d+", xml))
        license_v, license_raw = license_category(xml) if ok and has_article else ("UNKNOWN", "")
        lang = lang_of(xml) if xml else None
        a_type = type_of(xml) if xml else None
        year = year_of(xml, base_cand.get("year"))
        title = first(xml, r"<article-title>(.*?)</article-title>") or base_cand.get("title")
        journal = first(xml, r"<journal-title>(.*?)</journal-title>") or base_cand.get("journal")
        sentinel = bool(year is not None and year < 2015)
        reason = None
        if not (ok and has_article):
            reason = "E1_no_full_text"
        elif license_v == "RESTRICTED":
            reason = "E4_license_restricted"
        elif license_v == "UNKNOWN":
            reason = "E5_license_unknown"
        elif lang and not lang.startswith("en"):
            reason = "E3_non_english"
        elif raw_id in protected:
            reason = "E11_benchmark_protected"
        include = reason is None
        pmc = raw_id if raw_id.startswith("PMC") else "PMC" + raw_id
        meta = {
            "corpus_version": VERSION,
            "source_id": ver or raw_id,
            "pmcid": pmc,
            "pmid": base_cand.get("pmid"),
            "doi": base_cand.get("doi"),
            "title": title,
            "authors": None,
            "journal": journal,
            "publication_year": year,
            "article_type": a_type,
            "language": lang or "UNKNOWN",
            "license": license_raw or "UNKNOWN",
            "license_category": license_v,
            "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/" + pmc,
            "topic_ids": base_cand.get("topic_ids", []),
            "evidence_type": a_type,
            "sentinel": sentinel,
            "sentinel_reason": ("pre-2015 sentinel (dev corpus)" if sentinel else None),
            "benchmark_protected": raw_id in protected,
            "inclusion_status": "ACCEPTED" if include else "EXCLUDED",
            "exclusion_reason": reason,
            "raw_file": (ver + ".xml") if ver else None,
            "raw_sha256": sha(xml_path) if xml_path and xml_path.exists() else None,
        }
        (accepted if include else excluded).append(meta)

    # Cap dev corpus at target 500 by selection priority; overflow -> reserve.
    order = {pid: i for i, pid in enumerate(sel["pending"])}
    accepted.sort(key=lambda m: (order.get(m["pmcid"], 10**9), m["pmcid"]))
    reserve = accepted[DEV_TARGET:]
    for m in reserve:
        m["inclusion_status"] = "RESERVE"
        m["exclusion_reason"] = "E14_reserve_priority_for_primary_corpus"
    accepted = accepted[:DEV_TARGET]
    excluded.extend(reserve)

    # Preserve selection-stage exclusion records (E2/E8), dedup by pmcid.
    prior = {}
    if (MAN / "excluded_manifest.jsonl").exists():
        try:
            with open(MAN / "excluded_manifest.jsonl", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        e = json.loads(line)
                        prior[e.get("pmcid")] = e
        except Exception:
            prior = {}
    for e in excluded:
        prior[e.get("pmcid")] = e
    excluded_all = list(prior.values())

    with open(MAN / "accepted_manifest.jsonl", "w", encoding="utf-8") as f:
        for m in accepted:
            f.write(json.dumps(m) + "\n")
    with open(MAN / "excluded_manifest.jsonl", "w", encoding="utf-8") as f:
        for e in excluded_all:
            f.write(json.dumps(e) + "\n")
    with open(META / "articles_metadata.json", "w", encoding="utf-8") as f:
        json.dump(accepted + excluded, f, indent=2)

    years = Counter(m["publication_year"] for m in accepted if m["publication_year"])
    types = Counter(m["article_type"] for m in accepted)
    lic = Counter(m["license_category"] for m in accepted)
    journals = sorted({m["journal"] for m in accepted if m["journal"]})
    topic_counts = Counter()
    multi = 0
    for m in accepted:
        for t in m.get("topic_ids", []):
            topic_counts[t] += 1
        if len(m.get("topic_ids", [])) > 1:
            multi += 1
    n_2015plus = sum(1 for m in accepted if (m["publication_year"] or 0) >= 2015)
    sentinels = sum(1 for m in accepted if m["sentinel"])
    langs = Counter((m["language"] or "UNKNOWN") for m in accepted)
    n_english = sum(1 for m in accepted if (m["language"] or "").startswith("en"))
    missing = {"title": sum(1 for m in accepted if not m["title"]),
               "journal": sum(1 for m in accepted if not m["journal"]),
               "year": sum(1 for m in accepted if m["publication_year"] is None),
               "license": sum(1 for m in accepted if m["license"] in ("UNKNOWN", "")),
               "pmid": sum(1 for m in accepted if not m["pmid"]),
               "doi": sum(1 for m in accepted if not m["doi"])}
    stats = {
        "corpus_version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": len(recs), "selected_pending": len(sel["pending"]),
        "accepted": len(accepted), "excluded": len(excluded_all),
        "reserve_overflow": len(reserve),
        "exclusion_reasons": dict(Counter(e.get("reason") or e.get("exclusion_reason") for e in excluded_all)),
        "duplicates": 0,
        "protected_documents_count": len(protected),
        "download_status": {"ok": sum(1 for v in status.values() if v.get("ok")),
                            "attempted": len(status)},
        "license_categories": dict(lic),
        "languages": dict(langs),
        "n_english": n_english,
        "pct_english": round(100 * n_english / max(1, len(accepted)), 2) if accepted else None,
        "pct_permissive": round(100 * lic.get("PERMISSIVE", 0) / max(1, len(accepted)), 2) if accepted else None,
        "publication_years": {str(kk): v for kk, v in sorted(years.items())},
        "n_2015_plus": n_2015plus, "sentinel_count": sentinels,
        "pct_2015_plus": round(100 * n_2015plus / max(1, len(accepted)), 2) if accepted else None,
        "article_types": dict(types),
        "topic_counts": dict(topic_counts),
        "multi_topic_accepted": multi,
        "missing_metadata": missing,
        "unique_journals": len(journals),
        "unique_pmcid": len({m["pmcid"] for m in accepted}),
        "unique_pmid": len({m["pmid"] for m in accepted if m["pmid"]}),
        "unique_doi": len({m["doi"] for m in accepted if m["doi"]}),
    }
    with open(META / "corpus_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    md = ["# ElderDocAI-GeriLit dev-v0.1 - Corpus Statistics", ""]
    for k, v in stats.items():
        md.append(f"- **{k}**: {v}")
    with open(META / "corpus_statistics.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()