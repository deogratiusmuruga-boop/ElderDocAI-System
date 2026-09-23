#!/usr/bin/env python3
"""Probe: PMC-database OA search -> esummary -> oa.fcgi (license + tarball link)."""
import json
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (ElderDocAI-dev-research)"}


def get(url, timeout=90):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")


def main():
    qs = {
        "pmc_oa_only": "open%20access%5Bfilter%5D",
        "pmc_age_oa": ("%22aged%22%5BMeSH%5D%20AND%20open%20access%5Bfilter%5D"),
        "pmc_polypharm_oa": ("polypharmacy%20AND%20%22older%20adults%22%20AND%20"
                             "open%20access%5Bfilter%5D"),
    }
    picks = {}
    for k, v in qs.items():
        try:
            j = json.loads(get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
                               "esearch.fcgi?db=pmc&term=" + v +
                               "&retmode=json&retmax=3"))
            ids = j["esearchresult"].get("idlist", [])
            print(k, "count:", j["esearchresult"].get("count"), "ids:", ids)
            if ids:
                picks[k] = ids[0]
        except Exception as e:
            print(k, "ERR", str(e)[:200])
    if not picks:
        print("no PMC results; stopping")
        return
    pid = picks["pmc_polypharm_oa"] if "pmc_polypharm_oa" in picks else list(picks.values())[0]
    try:
        s = json.loads(get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
                           "esummary.fcgi?db=pmc&id=" + pid + "&retmode=json"))
        rec = s["result"][pid]
        print("esummary keys:", sorted(rec.keys()))
        for kk in ("title", "fulljournalname", "pubdate", "epubdate", "source",
                   "doi", "pmid", "pubtype"):
            print("  ", kk, "=", rec.get(kk))
    except Exception as e:
        print("esummary ERR", str(e)[:300])
    try:
        oc = get("https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=" + pid)
        print("OA response:\n", oc[:2500])
    except Exception as e:
        print("oa.fcgi ERR", str(e)[:300])


if __name__ == "__main__":
    main()