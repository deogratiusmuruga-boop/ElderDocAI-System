#!/usr/bin/env python3
"""ElderDocAI-GeriLit - Phase 3 Task 10E: failure + benchmark-validity
diagnostic (READ-ONLY).

Diagnoses why Task 10D retrieval metrics were low. NO benchmark/corpus/index/
retrieval/production changes. NO optimization. Diagnostic measurements only.

Produces:
  metadata/task10e_failure_analysis.json
  metadata/phase3_task10e_retrieval_failure_analysis_report.md
  metadata/validation/phase3_task10e_validation.{json,md}
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

META_DIR = Path(__file__).resolve().parent
GERI_DIR = META_DIR.parent
REPO_DIR = GERI_DIR.parent.parent
if str(GERI_DIR) not in sys.path:
    sys.path.insert(0, str(GERI_DIR))

GOLD_FILE = REPO_DIR / "data" / "geri_lit_gold.json"
RES10D_FILE = META_DIR / "task10d_retrieval_results.json"
CHUNKS_FILE = GERI_DIR / "chunks" / "chunks.jsonl"

CONFIGS = ("dense", "sparse", "hybrid", "rerank")
K_EVAL = (1, 3, 5)

STOPWORDS = set("""
a an and are as at be by for from how in is it of on or the to what when
where which who why with you your this that these those their its not no can
may do does did has have had been being were was also more most such other any
age older adults elderly geriatric care about into than then if but so because
""".split())


def norm_tokens(text):
    return [t for t in re.findall(r"[a-z]+", text.lower())
            if t not in STOPWORDS and len(t) > 1]
def load_gold():
    doc = json.loads(GOLD_FILE.read_text(encoding="utf-8"))
    return doc["records"]


def load_10d():
    return json.loads(RES10D_FILE.read_text(encoding="utf-8"))["results"]


def load_chunks():
    out = {}
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                out[c["chunk_id"]] = c
    return out


def stats(vals):
    vals = sorted(float(v) for v in vals)
    if not vals:
        return {"n": 0, "min": None, "mean": None, "median": None,
                "max": None}
    return {"n": len(vals), "min": round(vals[0], 6),
            "mean": round(sum(vals) / len(vals), 6),
            "median": round(vals[len(vals) // 2], 6),
            "max": round(vals[-1], 6)}


def main():
    recs = load_gold()
    res = load_10d()
    chunks = load_chunks()
    assert len(recs) == 134 and len(res) == 134

    from geri_lit_retriever import GeriLitRetriever  # noqa: E402
    retriever = GeriLitRetriever()

    rows = []
    for rec, r in zip(recs, res):
        gid = r["final_benchmark_id"]
        gold_ids = list(r["gold_relevant_chunk_ids"])
        gold_set = set(gold_ids)
        q = r["question"].strip()
        q_toks = norm_tokens(q)
        q_content = set(q_toks)

        per_cfg = {}
        for cfg in CONFIGS:
            rank = r["rankings"][cfg]
            pos = [i + 1 for i, c in enumerate(rank) if c in gold_set]
            per_cfg[cfg] = {
                "rank1": 1 if 1 in pos else 0,
                "rank3": 1 if any(p <= 3 for p in pos) else 0,
                "rank5": 1 if any(p <= 5 for p in pos) else 0,
                "first_rank": min(pos) if pos else None,
            }

        any5 = any(per_cfg[c]["rank5"] for c in CONFIGS)
        n_hit_cfgs = sum(1 for c in CONFIGS if per_cfg[c]["rank5"])

        gold_chunk = chunks.get(gold_ids[0], {})
        gold_text = (gold_chunk.get("text") or "").strip()
        gold_toks = norm_tokens(gold_text)
        gold_content = set(gold_toks)
        overlap = q_content & gold_content
        lex_ratio = (len(overlap) / len(q_content) if q_content else 0.0)

        vq = retriever.embed_query(q) if q else None
        vg = (retriever.embed_query(gold_text[:1000])
              if gold_text else None)
        sem = (float((vq[0] * vg[0]).sum())
               if vq is not None and vg is not None and gold_text else None)

        comp = 0
        if q_content:
            for cid, c in chunks.items():
                if c["pmcid"] != gold_chunk.get("pmcid"):
                    continue
                if cid == gold_ids[0]:
                    continue
                if q_content & set(norm_tokens(c.get("text") or "")):
                    comp += 1

        rows.append({
            "final_benchmark_id": gid,
            "original_candidate_id": r.get("original_candidate_id"),
            "question": q,
            "primary_topic": r.get("primary_topic"),
            "secondary_topics": r.get("secondary_topics", []),
            "pmcid": rec.get("pmcid"),
            "gold_chunk_ids": gold_ids,
            "per_config": per_cfg,
            "any_config_top5": any5,
            "n_config_hits_top5": n_hit_cfgs,
            "q_token_count": len(q_toks),
            "gold_token_count": len(gold_toks),
            "q_content_words": len(q_content),
            "overlap_content_tokens": len(overlap),
            "lex_overlap_ratio": round(lex_ratio, 6),
            "q_content_words_in_gold": len(overlap),
            "sem_sim": round(sem, 6) if sem is not None else None,
            "gold_section_title": gold_chunk.get("section_title"),
            "gold_section_id": gold_chunk.get("section_id"),
            "gold_region": gold_chunk.get("region"),
            "gold_content_type": gold_chunk.get("content_type"),
            "gold_evidence_type": gold_chunk.get("evidence_type"),
            "gold_word_count": len(gold_toks),
            "gold_has_descriptive_title": bool(
                (gold_chunk.get("section_title") or "").strip()
                and gold_chunk["section_title"].lower() != "abstract"),
            "competing_chunks_in_article": comp,
        })
    catA = [r for r in rows if r["n_config_hits_top5"] >= 3]
    catB = [r for r in rows if 1 <= r["n_config_hits_top5"] <= 2]
    catC = [r for r in rows if r["n_config_hits_top5"] == 0]

    cfg_summary = {}
    for cfg in CONFIGS:
        cfg_summary[cfg] = {
            "top1_hits": sum(1 for r in rows
                             if r["per_config"][cfg]["rank1"]),
            "top3_hits": sum(1 for r in rows
                             if r["per_config"][cfg]["rank3"]),
            "top5_hits": sum(1 for r in rows
                             if r["per_config"][cfg]["rank5"]),
            "no_top5": sum(1 for r in rows
                           if not r["per_config"][cfg]["rank5"]),
        }
    overall = {
        "n_questions": len(rows),
        "any_config_top5": sum(1 for r in rows if r["any_config_top5"]),
        "none_config_top5": sum(1 for r in rows
                                if not r["any_config_top5"]),
    }

    hit_rows = [r for r in rows if r["any_config_top5"]]
    miss_rows = [r for r in rows if not r["any_config_top5"]]
    lex_data = {
        "hits": stats([r["lex_overlap_ratio"] for r in hit_rows]),
        "misses": stats([r["lex_overlap_ratio"] for r in miss_rows]),
    }
    sem_data = {
        "hits": stats([r["sem_sim"] for r in hit_rows
                       if r["sem_sim"] is not None]),
        "misses": stats([r["sem_sim"] for r in miss_rows
                         if r["sem_sim"] is not None]),
    }

    topics = {}
    for t in sorted({r["primary_topic"] for r in rows}):
        tr = [r for r in rows if r["primary_topic"] == t]
        topics[t] = {
            "n": len(tr),
            "top1_hits_any": sum(1 for r in tr if any(
                r["per_config"][c]["rank1"] for c in CONFIGS)),
            "top3_hits_any": sum(1 for r in tr if any(
                r["per_config"][c]["rank3"] for c in CONFIGS)),
            "top5_hits_any": sum(1 for r in tr if r["any_config_top5"]),
            "no_top5": sum(1 for r in tr if not r["any_config_top5"]),
            "mean_lex_overlap": round(
                sum(r["lex_overlap_ratio"] for r in tr) / max(1, len(tr)), 6),
            "mean_q_len": round(
                sum(r["q_token_count"] for r in tr) / max(1, len(tr)), 3),
            "mean_gold_len": round(
                sum(r["gold_word_count"] for r in tr) / max(1, len(tr)), 3),
        }

    def first_n(group, n=5):
        return [row["final_benchmark_id"] for row in group[:n]]

    g1 = [r for r in rows if any(r["per_config"][c]["rank1"]
                                 for c in CONFIGS)]
    g2 = [r for r in rows if any(0 < (r["per_config"][c]["first_rank"]
                                      or 99) <= 5 for c in CONFIGS)
          and not any(r["per_config"][c]["rank1"] for c in CONFIGS)]

    out = {
        "task": "phase3_task10e",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "metadata/task10e_failure_analysis.py",
        "dataset": {"questions": len(rows),
                    "pmcid": len({r["pmcid"] for r in rows}),
                    "gold_chunks": len({c for r in rows
                                        for c in r["gold_chunk_ids"]})},
        "overall": overall,
        "per_config": cfg_summary,
        "categories": {"A_all_or_most": len(catA),
                       "B_some": len(catB),
                       "C_none": len(catC)},
        "lexical": lex_data,
        "semantic": sem_data,
        "topics": topics,
        "representative_examples": {
            "group1_top1": first_n(g1),
            "group2_ranks2to5": first_n(g2),
            "group3_miss_some": first_n(catB),
            "group4_all_fail": first_n(catC, 8),
        },
        "per_query": rows,
    }
    (META_DIR / "task10e_failure_analysis.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps({
        "dataset": out["dataset"],
        "overall": overall,
        "per_config": cfg_summary,
        "categories": out["categories"],
        "lexical_hits_mean": lex_data["hits"]["mean"],
        "lexical_misses_mean": lex_data["misses"]["mean"],
        "semantic_hits_mean": sem_data["hits"]["mean"],
        "semantic_misses_mean": sem_data["misses"]["mean"],
        "topics_covered": len(topics),
    }, indent=2))


if __name__ == "__main__":
    main()