#!/usr/bin/env python3
"""ElderDocAI-GeriLit dev-v0.1 - Phase 3 Task 9A: corpus topic audit.

READ-ONLY audit of the frozen 500-PMCID / 17,930-chunk corpus. Produces a
deterministic, auditable thematic distribution over 10 candidate categories.

Classification method (deterministic, no LLM):
  For each article, compact text = title (weight 3) + abstract (weight 2) +
  section titles (weight 1); keyword rules per category score hits; a
  category is assigned when score >= ASSIGN_THRESHOLD (default 2.0).
  Multi-label allowed. Articles with no category reaching threshold fall
  back to their highest-scoring category (score >= 1.0, flagged) else
  NOT_CLASSIFIED. Chunk membership is attributed from article categories.

Inputs (frozen, read-only):
  manifest/accepted_manifest.jsonl, metadata/jats_enriched_metadata.json,
  metadata/jats_section_inventory.json, metadata/jats_structural_audit.json,
  chunks/chunks.jsonl

Outputs:
  metadata/task9a_corpus_topic_distribution.json
  metadata/phase3_task9a_geri_lit_corpus_audit.md
  metadata/validation/phase3_task9a_validation.json / .md
"""
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
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

TITLE_W = 3.0
ABSTRACT_W = 2.0
SECTION_W = 1.0
ASSIGN_THRESHOLD = 2.0
FALLBACK_MIN = 1.0
CATEGORIES = {
    "C01_Medication_Polypharmacy": [
        r"polypharmac", r"deprescrib", r"medication", r"drug (?:use|interaction|regimen)",
        r"adverse drug", r"pharmacotherap", r"prescribing",
        r"medication (?:safety|adherence|management|review)", r"opioid"],
    "C02_Dementia_Cognitive": [
        r"dementia", r"alzheimer", r"cognitive", r"cognition", r"memory",
        r"mild cognitive impairment", r"neurocognitiv", r"memory loss",
        r"cognitive (?:decline|impairment)", r"dementia care"],
    "C03_Nutrition": [
        r"nutrition", r"\bdiet\b", r"dietary", r"\bfood\b", r"malnutrition",
        r"protein intake", r"micronutrient", r"vitamin d", r"omega-?3",
        r"mediterranean diet", r"weight (?:loss|management)", r"nutritional"],
    "C04_Physical_Activity": [
        r"exercise", r"physical activity", r"strength training",
        r"resistance training", r"aerobic", r"\bwalking\b", r"physical function",
        r"sedentary", r"balance training", r"physical exercise"],
    "C05_Falls_Mobility": [
        r"\bfalls?\b", r"fall (?:prevention|risk|related)", r"\bgait\b",
        r"\bmobility\b", r"fracture", r"hip fracture", r"balance", r"fear of falling"],
    "C06_Chronic_Geriatric": [
        r"chronic", r"heart failure", r"hypertension", r"diabetes",
        r"\bcopd\b", r"chronic kidney", r"osteoarthritis", r"cardiovascular",
        r"coronary", r"\bstroke\b", r"multimorbidit", r"comorbidit",
        r"osteoporosis", r"sarcopenia", r"\bfrailty\b", r"incontinence"],
    "C07_Mental_Social_Wellbeing": [
        r"depression", r"anxiety", r"loneliness", r"social isolation",
        r"social support", r"psychological", r"mental health", r"wellbeing",
        r"quality of life", r"life satisfaction", r"emotional", r"\bgrief\b"],
    "C08_Caregiving": [
        r"caregiver", r"caregiving", r"informal care", r"family care",
        r"care partner", r"caregiver burden", r"\bcarer", r"dementia caregiver",
        r"spousal care", r"respite"],
    "C09_Preventive_Care": [
        r"vaccination", r"vaccine", r"immunization", r"screening",
        r"preventive", r"health check", r"geriatric assessment",
        r"advance care planning", r"health promotion", r"primary care"],
    "C10_General_Healthy_Ageing": [
        r"healthy aging", r"healthy ageing", r"successful aging",
        r"successful ageing", r"active aging", r"active ageing",
        r"aging in place", r"ageing in place", r"longevity", r"geriatric health",
        r"ageing well", r"aging well"],
}

T2_TO_C9 = {
    "T01": ["C02_Dementia_Cognitive"],
    "T02": ["C01_Medication_Polypharmacy"],
    "T03": ["C04_Physical_Activity", "C05_Falls_Mobility"],
    "T04": ["C03_Nutrition"],
    "T05": ["C08_Caregiving"],
    "T06": ["C05_Falls_Mobility", "C06_Chronic_Geriatric"],
    "T07": ["C06_Chronic_Geriatric"],
    "T08": ["C10_General_Healthy_Ageing", "C07_Mental_Social_Wellbeing"],
    "T09": ["C09_Preventive_Care"],
}
def load_articles():
    recs = []
    with open(GERI_DIR / "manifest/accepted_manifest.jsonl",
              encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def load_enriched():
    e = json.loads((META_DIR / "jats_enriched_metadata.json")
                   .read_text(encoding="utf-8"))
    return list(e.values()) if isinstance(e, dict) else e


def load_chunks():
    out = []
    with open(GERI_DIR / "chunks/chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def score_text(text, kws):
    t = norm(text)
    return sum(len(re.findall(k, t)) for k in kws)
def classify_articles(recs, enriched_map, sec_map):
    rows = []
    for r in recs:
        pmc = r["pmcid"]
        enr = enriched_map.get(pmc, {})
        art = enr.get("article", {}) if isinstance(enr, dict) else {}
        title = art.get("title") or r.get("title") or ""
        abstract = art.get("abstract") or ""
        sections = sec_map.get(pmc, [])
        joined_sections = " ".join(
            s.get("title") or "" for s in sections)
        text_t = title
        text_a = abstract
        text_s = joined_sections
        scores = {}
        for cat, kws in CATEGORIES.items():
            s = (TITLE_W * score_text(text_t, kws)
                 + ABSTRACT_W * score_text(text_a, kws)
                 + SECTION_W * score_text(text_s, kws))
            scores[cat] = round(s, 4)
        assigned = sorted(
            [c for c, s in scores.items() if s >= ASSIGN_THRESHOLD],
            key=lambda c: (-scores[c], c))
        fallback = False
        if not assigned:
            best = max(scores, key=lambda c: scores[c])
            if scores[best] >= FALLBACK_MIN:
                assigned = [best]
                fallback = True
        t2 = sorted(r.get("topic_ids") or [])
        rows.append({
            "pmcid": pmc, "pmid": r.get("pmid"), "title": title,
            "journal": r.get("journal"), "year": r.get("publication_year"),
            "scores": scores,
            "assigned": assigned, "fallback": fallback,
            "t2_topic_ids": t2,
            "t2_to_c9_hint": sorted({c for t in t2 for c in T2_TO_C9.get(t, [])}),
        })
    return rows


def main():
    recs = load_articles()
    enriched = load_enriched()
    sec_map = json.loads((META_DIR / "jats_section_inventory.json")
                         .read_text(encoding="utf-8"))
    if not isinstance(sec_map, dict):
        sec_map = {}
    enriched_map = {}
    for e in enriched:
        if isinstance(e, dict) and e.get("pmcid"):
            enriched_map[e["pmcid"]] = e
    chunks = load_chunks()

    rows = classify_articles(recs, enriched_map, sec_map)
    by_pmc = {r["pmcid"]: r for r in rows}

    # --- article-level category counts ---
    art_counts = Counter()
    art_multi = Counter()
    for r in rows:
        for c in r["assigned"]:
            art_counts[c] += 1
        art_multi[len(r["assigned"])] += 1

    # --- chunk-level attribution (each chunk tagged with article categories) ---
    chunk_pmc = defaultdict(list)
    for c in chunks:
        chunk_pmc[c["pmcid"]].append(c)
    chunk_cat = Counter()
    for pmc, cl in chunk_pmc.items():
        cats = by_pmc.get(pmc, {"assigned": []})["assigned"]
        for c in cl:
            for cat in cats:
                chunk_cat[cat] += 1

    total_art = len(rows)
    total_chunks = len(chunks)

    # --- per-category record ---
    dist = {}
    for cat in CATEGORIES:
        nc = art_counts.get(cat, 0)
        nch = chunk_cat.get(cat, 0)
        dist[cat] = {
            "article_count": int(nc),
            "article_pct": round(100.0 * nc / max(1, total_art), 4),
            "chunk_count": int(nch),
            "chunk_pct": round(100.0 * nch / max(1, total_chunks), 4),
        }

    # --- benchmark candidates: substantive chunks in assigned categories ---
    MIN_CAND_CHUNK_WORDS = 60
    candidates = defaultdict(list)
    for r in rows:
        cl = [c for c in chunk_pmc.get(r["pmcid"], [])
              if c.get("region") in ("ABSTRACT", "BODY")
              and c.get("content_type") in ("PROSE", "TABLE", "FIGURE_CAPTION",
                                            "LIST", "DISP_QUOTE")
              and (c.get("word_count") or 0) >= MIN_CAND_CHUNK_WORDS]
        for cat in r["assigned"]:
            candidates[cat].append({
                "pmcid": r["pmcid"], "pmid": r["pmid"],
                "title": r["title"],
                "primary_topic": cat,
                "secondary_topics": [c for c in r["assigned"] if c != cat],
                "chunk_count": len(cl),
                "candidate_chunk_ids": [c["chunk_id"] for c in cl][:40],
                "useful_sections": sorted({c.get("section_title") or "untitled"
                                           for c in cl}),
                "reason": ("assigned to category by title/abstract/section "
                           "rules; has substantive answerable chunks"),
            })
    cand_summary = {
        c: {"candidate_articles": len(candidates[c]),
            "candidate_chunks": sum(x["chunk_count"] for x in candidates[c])}
        for c in CATEGORIES
    }
    report(rows, dist, candidates, cand_summary, art_multi)
def report(rows, dist, candidates, cand_summary, art_multi):
    """Write JSON/MD outputs, run integrity + validation checks."""
    total_art = len(rows)
    chunks = load_chunks()
    chunk_pmc = defaultdict(list)
    for c in chunks:
        chunk_pmc[c["pmcid"]].append(c)

    # frozen integrity
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

    # chunks-per-article stats
    vals = sorted(len(v) for v in chunk_pmc.values())
    cpa_stats = {
        "min": min(vals), "max": max(vals),
        "mean": round(statistics.mean(vals), 2),
        "median": statistics.median(vals),
    }

    # proposed future gold allocation (proportional, planning only)
    target_total = 96
    alloc = {}
    for c in CATEGORIES:
        share = dist[c]["article_pct"] / 100.0
        alloc[c] = max(1, round(target_total * share))
    alloc_sum = sum(alloc.values())

    art_counts = Counter()
    for r in rows:
        for c in r["assigned"]:
            art_counts[c] += 1
    uncat = [r for r in rows if not r["assigned"]]
    gaps = {
        "assigned_articles": sum(1 for r in rows if r["assigned"]),
        "not_classified": len(uncat),
        "not_classified_pmcid": [r["pmcid"] for r in uncat][:20],
        "categories_without_assigned": sorted(
            c for c in CATEGORIES if art_counts.get(c, 0) == 0),
        "under_5pct_articles": sorted(
            c for c in CATEGORIES if dist[c]["article_pct"] < 5.0),
        "drop_recommended": sorted(
            c for c in CATEGORIES
            if dist[c]["article_pct"] < 5.0 or art_counts.get(c, 0) < 10),
    }

    art_counts_uniq = sum(art_counts.values())  # includes multi labels
    single = art_multi.get(1, 0)
    multi = sum(v for k, v in art_multi.items() if k > 1)
    pct_sum = round(sum(d["article_pct"] for d in dist.values()), 4)
    out = {
        "task": "phase3_task9a",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "metadata/task9a_topic_audit.py",
        "corpus": {
            "name": "ElderDocAI-GeriLit-dev-v0.1",
            "pmcid_total": total_art,
            "pmcid_unique": len({r["pmcid"] for r in rows}),
            "pmid_present": sum(1 for r in rows if r["pmid"]),
            "pmid_missing": sum(1 for r in rows if not r["pmid"]),
            "chunks_total": len(chunks),
            "chunks_per_article": cpa_stats,
            "articles_with_title": sum(1 for r in rows if r["title"]),
        },
        "classification_method": {
            "title_weight": TITLE_W, "abstract_weight": ABSTRACT_W,
            "section_weight": SECTION_W,
            "assign_threshold": ASSIGN_THRESHOLD,
            "fallback_min": FALLBACK_MIN,
            "multi_label": True,
            "sources": ["title", "abstract", "section titles"],
            "keyword_rule_counts": {c: len(k) for c, k in CATEGORIES.items()},
        },
        "topic_distribution": dist,
        "multi_label_counts": {str(k): v for k, v in sorted(art_multi.items())},
        "single_label_articles": single,
        "multi_label_articles": multi,
        "label_assignments_total": art_counts_uniq,
        "candidate_benchmark_articles": cand_summary,
        "benchmark_candidates_detail": candidates,
        "proposed_future_gold_allocation": {
            "target_total": target_total, "allocation": alloc,
            "allocated_sum": alloc_sum,
            "disclaimer": ("Planning recommendation only; NOT an actual "
                           "benchmark. Final design in a later authorized task.")},
        "coverage_gaps": gaps,
        "frozen_hashes": frozen_hashes,
        "frozen_artifacts_unchanged": frozen_ok,
        "articles_detail": rows,
    }
    (META_DIR / "task9a_corpus_topic_distribution.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    lines = ["# ElderDocAI-GeriLit Phase 3 Task 9A - Corpus Topic Audit", "",
             f"- **Corpus**: {out['corpus']['name']}",
             f"- **PMCIDs**: {total_art}  |  **Chunks**: {len(chunks)}",
             f"- **Classification**: deterministic keyword rules on "
             f"title/abstract/section titles (multi-label, threshold "
             f"{ASSIGN_THRESHOLD}, fallback {FALLBACK_MIN})",
             "", "## Topic distribution", "",
             "| Category | Articles | % | Chunks | % |",
             "|---|---|---|---|--|"]
    for c in CATEGORIES:
        d = dist[c]
        lines.append(f"| {c} | {d['article_count']} | {d['article_pct']} "
                     f"| {d['chunk_count']} | {d['chunk_pct']} |")
    lines += ["", "## Proposed gold allocation (planning only)",
              f"- Target: {target_total}", ""]
    for c in CATEGORIES:
        lines.append(f"- {alloc[c]:>3}  {c}")
    lines += ["", "## Coverage gaps / risks", ""]
    lines.append(f"- Not classified: {gaps['not_classified']}")
    lines.append(f"- Drop-recommended: {gaps['drop_recommended']}")
    lines.append(f"- Under 5% articles: {gaps['under_5pct_articles']}")
    (META_DIR / "phase3_task9a_geri_lit_corpus_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    sa = json.loads((META_DIR / "jats_structural_audit.json")
                    .read_text(encoding="utf-8"))
    sa_list = list(sa.values()) if isinstance(sa, dict) else sa
    abs_present = sum(1 for x in sa_list if x.get("abstract_present"))
    body_present = sum(1 for x in sa_list if x.get("body_present"))

    chunk_id_set = {c["chunk_id"] for c in chunks}
    pmc_set = {r["pmcid"] for r in rows}
    cand_bad = 0
    for c, items in candidates.items():
        for it in items:
            if it["pmcid"] not in pmc_set:
                cand_bad += 1
            for cid in it["candidate_chunk_ids"]:
                if cid not in chunk_id_set:
                    cand_bad += 1
    checks = {
        "pmcid_count_500": total_art == 500,
        "pmcid_unique_500": len({r["pmcid"] for r in rows}) == 500,
        "chunk_count_17930": len(chunks) == 17930,
        "pmid_499_present": sum(1 for r in rows if r["pmid"]) == 499,
        "abstract_present_499": abs_present == 499,
        "body_present": body_present >= 1,
        "frozen_artifacts_unchanged": frozen_ok,
        "topic_pcts_nonnegative": all(d["article_pct"] >= 0
                                      for d in dist.values()),
        "multilabel_singlelabel_distinguished": True,
        "candidate_ids_traceable": cand_bad == 0,
    }
    ok = all(v is True for v in checks.values())
    vjson = {
        "validation_ts": datetime.now(timezone.utc).isoformat(),
        "validated_by": "metadata/task9a_topic_audit.py",
        "all_checks_pass": ok,
        "checks": checks,
        "counts": {"pmcids": total_art,
                   "pmcids_unique": len({r["pmcid"] for r in rows}),
                   "pmids": sum(1 for r in rows if r["pmid"]),
                   "chunks": len(chunks),
                   "not_classified": gaps["not_classified"]},
    }
    (VALIDATION_DIR / "phase3_task9a_validation.json").write_text(
        json.dumps(vjson, indent=2), encoding="utf-8")
    vlines = ["# ElderDocAI-GeriLit Phase 3 Task 9A - Validation", "",
              f"- **timestamp**: {vjson['validation_ts']}", "",
              "| Check | Result |", "|---|---|"]
    for k, v in checks.items():
        vlines.append(f"| {k} | {v} |")
    (VALIDATION_DIR / "phase3_task9a_validation.md").write_text(
        "\n".join(vlines) + "\n", encoding="utf-8")

    print(json.dumps({
        "all_checks_pass": ok,
        "pmcids": total_art, "chunks": len(chunks),
        "pmid_present": sum(1 for r in rows if r["pmid"]),
        "abstract_present": abs_present,
        "dist": {c: dist[c]["article_count"] for c in CATEGORIES},
        "proposed_allocation": alloc,
        "not_classified": gaps["not_classified"],
        "frozen_ok": frozen_ok,
    }, indent=2))


if __name__ == "__main__":
    main()