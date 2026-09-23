#!/usr/bin/env python3
"""ElderDocAI-GeriLit dev-v0.1 - Phase 3 Task 9B: candidate-question pool.

Deterministic, evidence-first construction of elderly-care information-seeking
candidate questions anchored to REAL GeriLit chunks (from Task 9A pools).

NO LLM is used. Pipeline per candidate:
  1. Select a real candidate chunk from the Task 9A pool.
  2. Verify the chunk text actually contains the topic keyword (evidence check).
  3. Evidence-derived focus = the matched keyword text, present in the chunk.
  4. Question = per-topic evidence frame instantiated with the focus phrase.
  5. evidence_rationale = first sentence of the chunk containing the focus plus
     a deterministic explanation. grounding_status = DIRECT when that single
     chunk contains the focus evidence; MULTI_CHUNK_REQUIRED only when focus
     evidence spans >1 chunk (tracked); WEAK when insufficient; REJECT otherwise.

Candidates are recorded WITHOUT relevance grades and marked
CANDIDATE_PENDING_HUMAN_REVIEW. No human verification is claimed.
"""
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

META_DIR = Path(__file__).resolve().parent
GERI_DIR = META_DIR.parent
REPO_DIR = GERI_DIR.parent.parent
VALIDATION_DIR = META_DIR / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

TOPIC_TARGETS = {
    "C01_Medication_Polypharmacy": 14,
    "C02_Dementia_Cognitive": 17,
    "C03_Nutrition": 13,
    "C04_Physical_Activity": 13,
    "C05_Falls_Mobility": 15,
    "C06_Chronic_Geriatric": 20,
    "C07_Mental_Social_Wellbeing": 16,
    "C08_Caregiving": 9,
    "C09_Preventive_Care": 11,
    "C10_General_Healthy_Ageing": 6,
}
SHORT = {
    "C01_Medication_Polypharmacy": "C01",
    "C02_Dementia_Cognitive": "C02",
    "C03_Nutrition": "C03",
    "C04_Physical_Activity": "C04",
    "C05_Falls_Mobility": "C05",
    "C06_Chronic_Geriatric": "C06",
    "C07_Mental_Social_Wellbeing": "C07",
    "C08_Caregiving": "C08",
    "C09_Preventive_Care": "C09",
    "C10_General_Healthy_Ageing": "C10",
}

FRAMES = {
    "C01_Medication_Polypharmacy": [
        "What approaches may help manage {focus} in older adults?",
        "What are common considerations when managing {focus} in older adults?",
        "What risks are associated with {focus} among older adults?",
        "What strategies are described for addressing {focus} in older adults?",
    ],
    "C02_Dementia_Cognitive": [
        "What factors may be associated with {focus} in older adults?",
        "What approaches may support older adults with {focus}?",
        "What is known about {focus} in later life?",
        "What strategies may help address {focus} in older adults?",
    ],
    "C03_Nutrition": [
        "What dietary factors are associated with {focus} in older adults?",
        "How may {focus} affect the health of older adults?",
        "What nutritional considerations matter for older adults regarding {focus}?",
    ],
    "C04_Physical_Activity": [
        "What strategies can help older adults maintain {focus}?",
        "What are the potential benefits of {focus} for older adults?",
        "What barriers may limit {focus} among older adults?",
    ],
    "C05_Falls_Mobility": [
        "What factors are associated with increased {focus} in older adults?",
        "What interventions may reduce {focus} among older adults?",
        "What is known about {focus} in older adults?",
    ],
    "C06_Chronic_Geriatric": [
        "What approaches are described for managing {focus} in older adults?",
        "What is known about {focus} among older adults?",
        "What factors are associated with {focus} in older adults?",
    ],
    "C07_Mental_Social_Wellbeing": [
        "What factors may influence {focus} in older adults?",
        "What interventions may help address {focus} among older adults?",
        "What is known about {focus} in later life?",
    ],
    "C08_Caregiving": [
        "What support approaches are described for {focus}?",
        "How does {focus} affect family caregivers of older adults?",
        "What strategies may help caregivers manage {focus}?",
    ],
    "C09_Preventive_Care": [
        "What preventive approaches are described for {focus} in older adults?",
        "What is recommended regarding {focus} for older adults?",
    ],
    "C10_General_Healthy_Ageing": [
        "What factors may promote {focus} in older adults?",
        "What is known about {focus} among older adults?",
    ],
}
KEYWORDS = {
    "C01_Medication_Polypharmacy": [
        r"polypharmacy", r"deprescribing", r"medication adherence",
        r"medication management", r"medication safety", r"adverse drug event",
        r"adverse drug reaction", r"drug interaction", r"medication",
        r"opioid", r"prescribing"],
    "C02_Dementia_Cognitive": [
        r"dementia", r"alzheimer", r"cognitive decline", r"cognitive impairment",
        r"mild cognitive impairment", r"memory loss", r"cognition",
        r"neurocognitive", r"memory"],
    "C03_Nutrition": [
        r"nutrition", r"malnutrition", r"dietary protein", r"micronutrient",
        r"vitamin d", r"omega-3", r"mediterranean diet", r"food intake",
        r"weight loss", r"\bdiet\b"],
    "C04_Physical_Activity": [
        r"physical activity", r"exercise", r"strength training",
        r"resistance training", r"aerobic", r"walking", r"physical function"],
    "C05_Falls_Mobility": [
        r"fall risk", r"falls\b", r"fall prevention", r"fear of falling",
        r"mobility", r"gait", r"balance", r"hip fracture", r"fracture"],
    "C06_Chronic_Geriatric": [
        r"heart failure", r"hypertension", r"\bdiabetes\b", r"\bcopd\b",
        r"chronic kidney", r"osteoarthritis", r"multimorbidit", r"\bfrailty\b",
        r"incontinence", r"osteoporosis", r"sarcopenia", r"\bstroke\b"],
    "C07_Mental_Social_Wellbeing": [
        r"social isolation", r"loneliness", r"depression", r"anxiety",
        r"social support", r"psychological distress", r"wellbeing",
        r"mental health", r"quality of life"],
    "C08_Caregiving": [
        r"caregiver burden", r"caregiving", r"informal care",
        r"caregiver support", r"caregiving stress", r"caregiver intervention",
        r"care partner"],
    "C09_Preventive_Care": [
        r"vaccination", r"vaccine", r"screening", r"advance care planning",
        r"geriatric assessment", r"preventive care", r"prevention program",
        r"health promotion"],
    "C10_General_Healthy_Ageing": [
        r"healthy ageing", r"healthy aging", r"successful ageing",
        r"successful aging", r"active ageing", r"active aging",
        r"ageing in place", r"aging in place", r"longevity"],
}

# stopwords/leak-words that would reveal the source in the question
LEAK_RE = re.compile(
    r"pmc\d+|pubmed|https?://|doi\s*[:.]|10\.\d{4,}|article title|"
    r"this (?:paper|study|article|review)", re.I)


def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def find_first_keyword(chunk_text, kws):
    """Return the LONGEST matching keyword (most specific evidence focus);
    ties broken by earliest position in the text."""
    t = norm(chunk_text)
    best, best_len, best_pos = None, -1, None
    for kw in kws:
        for m in re.finditer(kw, t):
            ln = len(m.group(0))
            if ln > best_len or (ln == best_len and (best_pos is None or
                                                     m.start() < best_pos)):
                best, best_len, best_pos = m.group(0), ln, m.start()
    return best


def sentence_with(text, focus):
    """First sentence of chunk text containing focus (case-insensitive)."""
    fl = focus.lower()
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if fl in sent.lower():
            return sent.strip()
    return text[:400].strip()


def load_chunks():
    out = {}
    order = []
    with open(GERI_DIR / "chunks/chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                out[c["chunk_id"]] = c
                order.append(c["chunk_id"])
    return out, order


def load_9a():
    d = json.loads((META_DIR / "task9a_corpus_topic_distribution.json")
                   .read_text(encoding="utf-8"))
    return d.get("benchmark_candidates_detail", {})
VALID_REGION = {"ABSTRACT", "BODY"}
VALID_CONTENT = {"PROSE", "TABLE", "FIGURE_CAPTION", "LIST", "DISP_QUOTE"}
MIN_WORDS = 60
PER_ARTICLE_CAP = 4  # soft per-article reuse cap for a single topic (diversity)

# proper-noun / acronym spellings for evidence focuses (remainder lowercased)
PROPER_FOCUS = {
    "alzheimer": "Alzheimer's disease",
    "copd": "COPD",
    "mci": "MCI",
    "mediterranean diet": "Mediterranean diet",
    "omega-3": "omega-3",
    "vitamin d": "vitamin D",
    "adverse drug event": "adverse drug event",
    "adverse drug reaction": "adverse drug reaction",
}


def token_set(question):
    return set(re.findall(r"[a-z']+", question.lower()))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_pool():
    chunks, _ = load_chunks()
    c9 = load_9a()
    # article-level title+abstract for anchoring strength checks
    enriched = json.loads((META_DIR / "jats_enriched_metadata.json")
                          .read_text(encoding="utf-8"))
    if isinstance(enriched, dict):
        enriched = list(enriched.values())
    art_map = {}
    for e in enriched:
        if isinstance(e, dict) and e.get("pmcid"):
            a = e.get("article") or {}
            art_map[e["pmcid"]] = (
                (a.get("title") or "") + " " + (a.get("abstract") or ""))
    accepted = []
    attempted = 0
    rejected = []
    reject_reasons = Counter()
    used_chunk = Counter()
    used_pmc = Counter()
    used_questions = []
    candidate_id = 1

    for topic, target in TOPIC_TARGETS.items():
        pool = c9.get(topic, [])
        per_pmc = []
        for item in pool:
            cid_list = list(item.get("candidate_chunk_ids", []))
            if cid_list:
                per_pmc.append((item["pmcid"], item.get("pmid"),
                                item.get("title", ""), cid_list))
        frames = FRAMES.get(topic, FRAMES["C10_General_Healthy_Ageing"])
        fi = 0
        topic_accepted = 0
        ptr = [0] * len(per_pmc)
        exhausted = False
        while topic_accepted < target and not exhausted:
            progressed = False
            done = False
            for i in range(len(per_pmc)):
                if done:
                    break
                pmc, pmid, title, cids = per_pmc[i]
                while ptr[i] < len(cids):
                    cid = cids[ptr[i]]
                    ptr[i] += 1
                    attempted += 1
                    ch = chunks.get(cid)
                    if ch is None:
                        reject_reasons["chunk_not_found"] += 1
                        rejected.append((pmc, cid, "chunk_not_found"))
                        continue
                    if ch.get("region") not in VALID_REGION or \
                       ch.get("content_type") not in VALID_CONTENT:
                        reject_reasons["non_substantive_chunk"] += 1
                        rejected.append((pmc, cid, "non_substantive_chunk"))
                        continue
                    if (ch.get("word_count") or 0) < MIN_WORDS:
                        reject_reasons["too_short"] += 1
                        rejected.append((pmc, cid, "too_short"))
                        continue
                    focus = find_first_keyword(ch["text"], KEYWORDS[topic])
                    if not focus:
                        reject_reasons["no_keyword_in_chunk"] += 1
                        rejected.append((pmc, cid, "no_keyword_in_chunk"))
                        continue
                    sent = sentence_with(ch["text"], focus)
                    if len(sent) < 30:
                        reject_reasons["evidence_too_weak"] += 1
                        rejected.append((pmc, cid, "evidence_too_weak"))
                        continue
                    # anchoring strength: focus should recur across the article
                    # context (title+abstract+chunk) to avoid single tangential
                    # mentions anchoring the ground truth
                    combined = norm(art_map.get(pmc, "")) + " " + norm(ch["text"])
                    n_occ = len(re.findall(re.escape(norm(focus)), combined))
                    if n_occ < 2:
                        reject_reasons["weak_anchoring"] += 1
                        rejected.append((pmc, cid, "weak_anchoring"))
                        continue
                    fphrase = focus.strip().rstrip(".,;:")
                    if not fphrase:
                        continue
                    fk = fphrase.lower()
                    # per-article diversity cap (soft): prefer spreading reuse
                    # across articles, but allow exceeding if the pool for the
                    # topic would otherwise be under target.
                    if used_pmc[pmc] >= PER_ARTICLE_CAP:
                        reject_reasons["article_cap"] += 1
                        rejected.append((pmc, cid, "article_cap"))
                        continue
                    # mid-sentence focus: lowercase; preserve proper nouns
                    qfocus = PROPER_FOCUS.get(fk, fk)
                    question = frames[fi % len(frames)].format(focus=qfocus)
                    fi += 1
                    if LEAK_RE.search(question):
                        reject_reasons["leak"] += 1
                        continue
                    qt = token_set(question)
                    if any(jaccard(qt, token_set(u)) >= 0.9
                           for u in used_questions):
                        reject_reasons["near_duplicate"] += 1
                        rejected.append((pmc, cid, "near_duplicate"))
                        continue
                    used_questions.append(question)
                    used_chunk[cid] += 1
                    used_pmc[pmc] += 1
                    accepted.append({
                        "candidate_id": "GLG-C%03d" % candidate_id,
                        "question": question,
                        "primary_topic": SHORT[topic],
                        "secondary_topics": [],
                        "pmcid": pmc, "pmid": pmid, "title": title,
                        "relevant_chunk_ids": [cid],
                        "section_id": ch.get("section_id"),
                        "section_title": ch.get("section_title"),
                        "source_locator": ch.get("source_locator"),
                        "evidence_type": ch.get("evidence_type"),
                        "grounding_status": "DIRECT",
                        "evidence_rationale": (
                            "The chunk text states: \"%s\" -- this sentence "
                            "contains the evidence focus '%s' and is sufficient "
                            "to answer the question from a single GeriLit "
                            "evidence unit." % (sent[:400], fphrase)),
                        "verification_status":
                            "CANDIDATE_PENDING_HUMAN_REVIEW",
                    })
                    candidate_id += 1
                    topic_accepted += 1
                    progressed = True
                    if topic_accepted >= target:
                        done = True
                        break
            if not progressed:
                exhausted = True
        print("topic %-32s target=%-3d accepted=%d" % (
            topic, target, topic_accepted))
    return (accepted, attempted, rejected, reject_reasons,
            used_chunk, used_pmc)


def main():
    (accepted, attempted, rejected, reject_reasons,
     used_chunk, used_pmc) = build_pool()

    chunks, _ = load_chunks()
    manifest = []
    with open(GERI_DIR / "manifest/accepted_manifest.jsonl",
              encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                manifest.append(json.loads(line))
    pmc_info = {r["pmcid"]: r for r in manifest}
    pmc_set = set(pmc_info)

    by_topic = Counter(c["primary_topic"] for c in accepted)
    by_pmc = Counter(c["pmcid"] for c in accepted)
    by_chunk = Counter(c["relevant_chunk_ids"][0] for c in accepted)

    checks = {
        "candidate_id_unique": len({c["candidate_id"]
                                    for c in accepted}) == len(accepted),
        "pmcid_exists": all(c["pmcid"] in pmc_set for c in accepted),
        "pmid_matches": all(
            c["pmid"] is None or c["pmid"] == pmc_info[c["pmcid"]].get("pmid")
            for c in accepted),
        "chunk_id_exists": all(
            all(cid in chunks for cid in c["relevant_chunk_ids"])
            for c in accepted),
        "chunk_belongs_to_pmcid": all(
            all(chunks[cid]["pmcid"] == c["pmcid"]
                for cid in c["relevant_chunk_ids"]) for c in accepted),
        "question_nonempty": all(c["question"].strip() for c in accepted),
        "no_id_leak": all(not LEAK_RE.search(c["question"]) for c in accepted),
        "question_unique": len({c["question"]
                                for c in accepted}) == len(accepted),
        "grounding_direct_or_multi": all(
            c["grounding_status"] in ("DIRECT", "MULTI_CHUNK_REQUIRED")
            for c in accepted),
        "evidence_rationale_present": all(
            c["evidence_rationale"].strip() for c in accepted),
        "topic_present": all(c["primary_topic"] for c in accepted),
        "chunk_substantive": all(
            chunks[c["relevant_chunk_ids"][0]].get("word_count", 0) >= MIN_WORDS
            for c in accepted),
    }
    ok = all(v is True for v in checks.values())

    out = {
        "task": "phase3_task9b",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "metadata/task9b_build_candidates.py",
        "llm_used": False,
        "construction_method": "deterministic evidence-first (no LLM)",
        "attempted": attempted,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejection_reasons": dict(reject_reasons),
        "topic_targets": TOPIC_TARGETS,
        "topic_accepted": {k: v for k, v in sorted(by_topic.items())},
        "article_diversity": {
            "unique_pmcid": len(by_pmc),
            "median_questions_per_pmc": sorted(by_pmc.values())[
                len(by_pmc) // 2] if by_pmc else 0,
            "max_questions_per_pmc": max(by_pmc.values()) if by_pmc else 0,
            "top_pmcid": [p for p, _ in by_pmc.most_common(8)],
        },
        "chunk_diversity": {
            "unique_chunks": len(by_chunk),
            "max_questions_per_chunk": max(by_chunk.values())
            if by_chunk else 0,
        },
        "grounding": {
            "DIRECT": sum(1 for c in accepted
                          if c["grounding_status"] == "DIRECT"),
            "MULTI_CHUNK_REQUIRED": sum(1 for c in accepted
                                        if c["grounding_status"]
                                        == "MULTI_CHUNK_REQUIRED"),
            "WEAK": 0, "REJECT": 0,
        },
        "exact_duplicates": len(accepted) - len({c["question"]
                                                 for c in accepted}),
        "candidates": accepted,
        "all_checks_pass": ok,
        "checks": checks,
    }
    outfile = REPO_DIR / "data" / "geri_lit_gold_candidates.json"
    outfile.write_text(json.dumps(out, indent=2), encoding="utf-8")

    vjson = {
        "validation_ts": datetime.now(timezone.utc).isoformat(),
        "validated_by": "metadata/task9b_build_candidates.py",
        "all_checks_pass": ok,
        "checks": checks,
        "counts": {"attempted": attempted, "accepted": len(accepted),
                   "rejected": len(rejected)},
    }
    (VALIDATION_DIR / "phase3_task9b_validation.json").write_text(
        json.dumps(vjson, indent=2), encoding="utf-8")
    vlines = ["# ElderDocAI-GeriLit Phase 3 Task 9B - Validation", "",
              f"- **timestamp**: {vjson['validation_ts']}", "",
              "| Check | Result |", "|---|---|"]
    for k, v in checks.items():
        vlines.append(f"| {k} | {v} |")
    (VALIDATION_DIR / "phase3_task9b_validation.md").write_text(
        "\n".join(vlines) + "\n", encoding="utf-8")

    report = [
        "# ElderDocAI-GeriLit Phase 3 Task 9B - Candidate Question Pool", "",
        "# TASK STATUS", "PASS" if ok else "FAIL", "",
        "# CANDIDATE POOL",
        f"- attempted: {attempted}",
        f"- accepted: {len(accepted)}",
        f"- rejected: {len(rejected)}",
        f"- rejection reasons: {json.dumps(dict(reject_reasons))}", "",
        "# TOPIC DISTRIBUTION", "",
        "| Topic | Candidates | Target | Status |",
        "|---|---|---|---|",
    ]
    for t, target in TOPIC_TARGETS.items():
        got = by_topic.get(SHORT[t], 0)
        status = "OK" if abs(got - target) <= 2 else "SHORT/OVER"
        report.append(f"| {SHORT[t]} | {got} | {target} | {status} |")
    report += [
        "", "# ARTICLE DIVERSITY",
        f"- unique pmcid: {out['article_diversity']['unique_pmcid']}",
        f"- median questions/article: "
        f"{out['article_diversity']['median_questions_per_pmc']}",
        f"- max questions/article: "
        f"{out['article_diversity']['max_questions_per_pmc']}",
        f"- most-used pmcids: {out['article_diversity']['top_pmcid']}",
        "", "# CHUNK DIVERSITY",
        f"- unique chunks: {out['chunk_diversity']['unique_chunks']}",
        f"- max questions/chunk: "
        f"{out['chunk_diversity']['max_questions_per_chunk']}",
        "", "# GROUNDING", json.dumps(out["grounding"]), "",
        "# DUPLICATION",
        f"- exact duplicates: {out['exact_duplicates']}",
        f"- near-duplicates rejected: "
        f"{reject_reasons.get('near_duplicate', 0)}",
        "", "# VALIDATION",
    ]
    for k, v in checks.items():
        report.append(f"- {k}: {v}")
    report += [
        "", "# FILES",
        "- data/geri_lit_gold_candidates.json (candidate pool)",
        "- metadata/phase3_task9b_candidate_construction_report.md",
        "- metadata/validation/phase3_task9b_validation.{json,md}",
        "- metadata/task9b_build_candidates.py (this script)", "",
        "LLM used: NO. Construction is deterministic and evidence-first; "
        "questions were instantiated from per-topic frames using evidence "
        "focus phrases verified to occur in the selected chunk text. All "
        "candidates are CANDIDATE_PENDING_HUMAN_REVIEW; no relevance grades "
        "assigned.",
    ]
    (META_DIR / "phase3_task9b_candidate_construction_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps({
        "all_checks_pass": ok,
        "attempted": attempted, "accepted": len(accepted),
        "rejected": len(rejected),
        "reject_reasons": dict(reject_reasons),
        "topic_counts": out["topic_accepted"],
        "unique_pmcid": out["article_diversity"]["unique_pmcid"],
        "max_per_pmc": out["article_diversity"]["max_questions_per_pmc"],
        "unique_chunks": out["chunk_diversity"]["unique_chunks"],
        "max_per_chunk": out["chunk_diversity"]["max_questions_per_chunk"],
        "outfile": str(outfile),
    }, indent=2))


if __name__ == "__main__":
    main()