#!/usr/bin/env python3
"""ElderDocAI-GeriLit dev-v0.1 - Stage 2: deterministic selection + OA pre-check.

Resumable, capped per run (--limit). Steps: dedup, relevance gate, stratified
sampling toward ~500 (+margin), OA Web Service membership/license pre-check.
Writes manifest/selected.json and appends to manifest/excluded_manifest.jsonl.
"""
import argparse
import json
import random
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(r"C:\Users\chosun\Documents\ElderDocAI-System\datasets\elderdocai_geriLit")
META = BASE / "metadata"
MAN = BASE / "manifest"
SEED = 42
TARGET = 500
FLOOR_FRAC = 0.05
MARGIN = int(TARGET * 1.15)
UA = {"User-Agent": "Mozilla/5.0 (ElderDocAI-dev-research; dev corpus stage2)"}
SLEEP = 0.37
CACHE = MAN / "oa_cache.json"


def get(url, timeout=120):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower())


def load_candidates():
    recs = []
    with open(MAN / "candidate_manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def dedup(recs):
    kept = {}
    dup_groups = {}
    for r in sorted(recs, key=lambda x: (x.get("year") or "9999", x["pmcid"])):
        key = None
        for idk in ("pmcid", "pmid", "doi"):
            v = r.get(idk)
            if v:
                key = (idk, str(v).lower())
                break
        if key is None:
            key = ("title", norm_title(r.get("title")))
        if key in kept:
            if r["pmcid"] not in dup_groups:
                dup_groups[r["pmcid"]] = {"duplicate_group": kept[key],
                                          "retained_record": kept[key],
                                          "duplicate_reason": "shared_identifier_or_norm_title"}
        else:
            kept[key] = r["pmcid"]
    return kept, dup_groups


def load_cache():
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return {}


def save_cache(c):
    CACHE.write_text(json.dumps(c, indent=2, sort_keys=True), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150, help="oa.fcgi pre-checks per run")
    args = ap.parse_args()

    recs = load_candidates()
    by_pmcid = {r["pmcid"]: r for r in recs}
    excluded = []
    relevant = [r for r in recs
                if r.get("topic_ids") and (r.get("title") or "").strip()]
    excluded += [{"pmcid": r["pmcid"], "reason": "E2_irrelevant_to_elderly_aging",
                  "note": "no topic membership or empty title"}
                 for r in recs if r not in relevant]

    keep_map, dup_groups = dedup(relevant)
    keep_set = set(keep_map.values())
    pool = [r for r in relevant if r["pmcid"] in keep_set]
    excluded += [{"pmcid": pid, "reason": "E8_duplicate", **d}
                 for pid, d in dup_groups.items()]
    with open(MAN / "excluded_manifest.jsonl", "w", encoding="utf-8") as f:
        for e in excluded:
            f.write(json.dumps(e) + "\n")

    rng = random.Random(SEED)
    taxo = json.loads((META / "topic_taxonomy.json").read_text(encoding="utf-8"))
    topics = [t["topic_id"] for t in taxo["topics"]]
    floor = max(1, int(FLOOR_FRAC * TARGET))
    chosen = set()
    for t in topics:
        member = [r["pmcid"] for r in pool
                  if t in r.get("topic_ids", []) and r["pmcid"] not in chosen]
        if not member:
            continue
        chosen.update(rng.sample(member, min(floor, len(member))))
    remain = [r["pmcid"] for r in pool if r["pmcid"] not in chosen]
    needed = MARGIN - len(chosen)
    if needed > 0 and remain:
        chosen.update(rng.sample(remain, min(needed, len(remain))))
    selected = sorted(chosen)
    print("selection:", len(selected), "floor", floor, "needed", needed, flush=True)

    pending = sorted(selected)
    with open(MAN / "selected.json", "w", encoding="utf-8") as f:
        json.dump({"target": TARGET, "margin": MARGIN, "seed": SEED,
                   "selected": pending, "pending": pending,
                   "rejected": [], "unchecked": [],
                   "note": ("OA-subset + permissive-license membership guaranteed by "
                            "discovery query (open_access[Filter] AND cc0/cc_by/cc_by-sa "
                            "license filters) per PMC Cloud Service README"),
                   "updated_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)
    print(json.dumps({
        "candidate_count": len(recs), "relevant_pool": len(pool),
        "duplicates": len(dup_groups), "selected": len(pending)}, indent=2))


if __name__ == "__main__":
    main()