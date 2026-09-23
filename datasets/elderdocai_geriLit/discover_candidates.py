#!/usr/bin/env python3
"""ElderDocAI-GeriLit dev-v0.1 - Stage 1: PMC OA discovery.

Official NCBI services only: esearch(db=pmc, open access[filter]) per topic,
esummary(db=pmc) metadata, idconv for PMID. Writes manifest/candidate_manifest.jsonl
and metadata/discovery_log.json. No full-text download in this stage.
"""
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(r"C:\Users\chosun\Documents\ElderDocAI-System\datasets\elderdocai_geriLit")
META = BASE / "metadata"
MAN = BASE / "manifest"
MAN.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (ElderDocAI-dev-research; dev corpus stage1)"}
SLEEP = 0.37  # politeness > 3 req/s (no API key)


def get(url, timeout=150):
    req = urllib.request.Request(url, headers=UA)
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", "replace")
        except Exception as e:
            last = e
            time.sleep(2 + 2 * attempt)
    raise last


def esearch(db, term, retmax):
    q = urllib.parse.urlencode({"db": db, "term": term, "retmode": "json",
                                "retmax": retmax, "sort": "relevance"})
    return json.loads(get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + q))


def esummary(db, ids):
    out = {}
    for i in range(0, len(ids), 150):
        chunk = ids[i:i + 150]
        q = urllib.parse.urlencode({"db": db, "id": ",".join(chunk), "retmode": "json"})
        j = json.loads(get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + q))
        out.update(j.get("result", {}))
        time.sleep(SLEEP)
    return out


def idconv(ids):
    out = {}
    for i in range(0, len(ids), 40):
        chunk = ids[i:i + 40]
        q = urllib.parse.urlencode({"ids": ",".join(chunk), "format": "json"})
        j = json.loads(get("https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?" + q))
        for rec in j.get("records", []):
            out[rec.get("pmcid")] = rec.get("pmid")
        time.sleep(SLEEP)
    return out


def main():
    taxo = json.loads((META / "topic_taxonomy.json").read_text(encoding="utf-8"))
    per_topic = 200
    candidates = {}
    discovery = {"corpus": "ElderDocAI-GeriLit",
                 "version": "ElderDocAI-GeriLit-dev-v0.1",
                 "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                 "mechanism": ("NCBI eutils esearch(db=pmc), open_access[Filter] AND "
                               "(cc0_license[Filter] OR cc_by_license[Filter] OR "
                               "cc_by-sa_license[Filter]) per PMC Cloud Service README"),
                 "license_note": ("PERMISSIVE group = cc0/cc_by/cc_by-sa; nc/*-nd excluded "
                                  "under license_policy.md; initial over-broad run "
                                  "('open access[filter]') discarded as it included "
                                  "non-OA-Subset items."),
                 "queries": []}
    license_q = ("open_access[Filter] AND "
                 "(cc0_license[Filter] OR cc_by_license[Filter] OR "
                 "cc_by-sa_license[Filter])")
    for topic in taxo["topics"]:
        term = topic["search_terms"] + " AND " + license_q
        try:
            res = esearch("pmc", term, per_topic)
            ids = res["esearchresult"].get("idlist", [])
            discovery["queries"].append({"topic_id": topic["topic_id"],
                                         "query": term,
                                         "count": res["esearchresult"].get("count"),
                                         "retmax": per_topic,
                                         "ids": len(ids)})
            for pid in ids:
                rec = candidates.setdefault(pid, {"pmcid": pid, "topic_ids": []})
                if topic["topic_id"] not in rec["topic_ids"]:
                    rec["topic_ids"].append(topic["topic_id"])
        except Exception as e:
            discovery["queries"].append({"topic_id": topic["topic_id"],
                                         "query": term, "error": str(e)[:200]})
        time.sleep(SLEEP)

    summary = esummary("pmc", list(candidates.keys()))
    for pid, rec in candidates.items():
        s = summary.get(pid, {})
        rec["title"] = s.get("title")
        rec["journal"] = s.get("fulljournalname") or s.get("source")
        pub = (s.get("pubdate") or s.get("epubdate") or "")
        rec["year"] = pub[:4] if pub[:4].isdigit() else None
        rec["epubdate"] = s.get("epubdate")
        for aid in s.get("articleids", []):
            if not isinstance(aid, dict):
                continue
            t = aid.get("idtype")
            if t == "doi":
                rec["doi"] = aid.get("value")
            elif t == "pmid":
                rec.setdefault("pmid", aid.get("value"))
    missing_pmid = [pid for pid, r in candidates.items() if not r.get("pmid")]
    if missing_pmid:
        conv = idconv(missing_pmid)
        for pid in missing_pmid:
            candidates[pid]["pmid"] = conv.get(pid) or candidates[pid].get("pmid")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    discovery["candidate_count"] = len(candidates)
    with open(META / "discovery_log.json", "w", encoding="utf-8") as f:
        json.dump(discovery, f, indent=2, default=str)
    with open(MAN / "candidate_manifest.jsonl", "w", encoding="utf-8") as f:
        for pid in sorted(candidates):
            rec = candidates[pid]
            rec.setdefault("doi", None)
            rec.setdefault("pmid", None)
            rec["discovered_at"] = stamp
            f.write(json.dumps(rec) + "\n")
    print(json.dumps({"candidate_count": len(candidates),
                      "with_pmid": sum(1 for r in candidates.values() if r.get("pmid")),
                      "with_doi": sum(1 for r in candidates.values() if r.get("doi")),
                      "queries": discovery["queries"]}, indent=2))


if __name__ == "__main__":
    main()