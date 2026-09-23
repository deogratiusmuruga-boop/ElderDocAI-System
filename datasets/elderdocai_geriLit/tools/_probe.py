#!/usr/bin/env python3
"""Probe: verify NCBI E-utilities + PMC OA Web Service access (no downloads of
substantial data). Part of Phase 3 Task 2 verification."""
import json
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (ElderDocAI-dev-research; contact: elderdocai-dev)"}


def get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")


def main():
    qs = {
        "oa_f_lower": "open%20access%5Bfilter%5D",
        "oa_f_cap": "open%20access%5BFilter%5D",
        "oa_nospace": "openaccess%5BFilter%5D",
        "oa_pubmed_pmc": "pubmed%20pmc%20open%20access%5Bfilter%5D",
        "aging_and_oa_f": "aging%5BMeSH%5D%20AND%20open%20access%5BFilter%5D",
        "aging_and_oa_ns": "aging%5BMeSH%5D%20AND%20openaccess%5BFilter%5D",
    }
    for k, v in qs.items():
        try:
            j = json.loads(get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
                               "esearch.fcgi?db=pubmed&term=" + v +
                               "&retmode=json&retmax=3"))
            print(k, "count:", j["esearchresult"]["count"],
                  "ids:", j["esearchresult"]["idlist"][:3])
        except Exception as e:
            print(k, "ERR", str(e)[:200])
    # resolve PMCID from the first OA-qualified query that returns results
    best = None
    for k, v in qs.items():
        try:
            j = json.loads(get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
                               "esearch.fcgi?db=pubmed&term=" + v +
                               "&retmode=json&retmax=5"))
            n = int(j["esearchresult"]["count"])
            print(k, "count:", n, "ids:", j["esearchresult"]["idlist"][:3])
            if n > 0 and best is None:
                best = v
        except Exception as e:
            print(k, "ERR", str(e)[:160])
    print("chosen query:", best)
    if best is None:
        print("no OA-qualified query; stopping probe")
        return
    try:
        j = json.loads(get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
                           "esearch.fcgi?db=pubmed&term=" + best +
                           "&retmode=json&retmax=5"))
        ids = j["esearchresult"]["idlist"]
        c = json.loads(get("https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
                           "?ids=" + ",".join(ids) + "&format=json"))
    except Exception as e:
        print("idconv ERR", str(e)[:300])
        return
    pmc = None
    for rec in c.get("records", []):
        print("  idconv:", rec.get("pmid"), "->", rec.get("pmcid"),
              "err:", rec.get("error"))
        if pmc is None and rec.get("pmcid"):
            pmc = rec["pmcid"]
    if pmc:
        oc = get("https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=" + pmc)
        print("OA response for", pmc, ":\n", oc[:2500])
    else:
        print("no PMCID resolved; stopping probe")


if __name__ == "__main__":
    main()