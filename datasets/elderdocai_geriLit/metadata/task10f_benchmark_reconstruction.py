#!/usr/bin/env python3
"""ElderDocAI-GeriLit - Phase 3 Task 10F: GeriLit-Gold v1.1 CANDIDATE benchmark
reconstruction (evidence-first, deterministic).

Purpose
-------
Build a SEPARATE candidate benchmark (data/geri_lit_gold_v1_1_candidate.json)
that reconstructs the 134 v1.0 records evidence-first:
    gold evidence -> substantive claim -> specific answerable question.

NOT human review. Every emitted candidate is PENDING_HUMAN_REVIEW. The frozen
v1.0 benchmark, Task 10D/10E artifacts, corpus, and indexes are READ-ONLY.

Method per v1.0 record
----------------------
1. Load the v1.0 gold chunk and the article context (all chunks of the PMC).
2. Compute deterministic alignment diagnostics (lexical overlap + semantic
   similarity using the cached BGE embeddings; same-article competition).
3. Extract the best claim sentence from the gold chunk (quantitative/effect/
   focus-bearing sentences are preferred).
4. Decide:
     RETAINED      - original v1.0 question is specific, focus is answered by
                     the gold claim sentence, alignment is strong.
     RECONSTRUCTED - question rebuilt from the claim sentence (shape-based
                     templates instantiated with evidence-derived content)
                     and gated for answerability from the gold chunk.
                     Chunks may be re-anchored to a better same-article chunk
                     only when the v1.0 gold chunk is not defensible.
   Records that cannot be responsibly reconstructed are EXCLUDED and reported
   with reasons (they do not appear in the candidate file).
5. Every included candidate keeps full provenance and diagnostics.

NO retrieval evaluation (no Recall@K / MRR / nDCG) is computed. No LLM is used.
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

import numpy as np  # noqa: E402

META_DIR = Path(__file__).resolve().parent
GERI_DIR = META_DIR.parent
REPO_DIR = GERI_DIR.parent.parent
if str(GERI_DIR) not in sys.path:
    sys.path.insert(0, str(GERI_DIR))

CAND_FILE = REPO_DIR / "data" / "geri_lit_gold_v1_1_candidate.json"
DIAG_FILE = META_DIR / "task10f_benchmark_reconstruction.json"

GOLD_FILE = REPO_DIR / "data" / "geri_lit_gold.json"
CHUNKS_FILE = GERI_DIR / "chunks" / "chunks.jsonl"
ROW_MAP_FILE = GERI_DIR / "index" / "row_mapping.json"
EMB_FILE = GERI_DIR / "index" / "embeddings.npy"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

GOLD_SHA_EXPECTED = "28ef54faeec3b9a1cb8c1c79c9a478d8ea3618574b2787a82a74c1460f209c62"
FROZEN_EXPECTED = {
    "chunks.jsonl": "62bfdde3c7e77cf56112e79a71bc79bdea70767f6b2625701e392bbb6581c6b3",
    "embeddings.npy": "b0905d2f96903dadc3cf5b4a737f330853656f846955de706f21bd714ce2dacf",
    "faiss_index.bin": "b7016b9dabbab08ee68f875007b6db2b013af4472b1a3c0958af8a566b59f2b3",
    "bm25.pkl": "17f503240f2f9a13abf58c94359884b03b06bec1a5a59f63f921328e2a2c70c9",
    "row_mapping.json": "4d649f81d554c41ec7b63c65dce887466dfa6245745c568315735f1b93a1872b",
}

# ---------------------------------------------------------------------------
# Deterministic gates (documented thresholds)
# ---------------------------------------------------------------------------
RETAIN_LEX = 0.60            # fraction of question content tokens matched in gold
RETAIN_SEM = 0.70            # BGE cosine similarity question vs gold chunk
RECONSTRUCT_MIN_CLAIM = 6.0  # claim-sentence score floor for reconstruction
GATE_LEX = 0.25              # answerability gate: question vs gold chunk
GATE_SEM = 0.45              # answerability gate (semantic)
REANCHOR_FLOOR = 4.0         # claim score floor for a re-anchor candidate chunk

STOPWORDS = set("""
a an and are as at be by for from how in is it of on or the to what when
where which who why with you your this that these those their its not no can
may do does did has have had been being were was also more most such other any
age older adults elderly geriatric care about into than then if but so because
there here all some each both among within between over under per vs versus
""".split())

# Tokens that carry no retrieval intent / pure query scaffolding
GENERIC_WORDS = set("""
what which how why when where who is are was were do does did can may could
approaches strategies interventions methods aspects considerations benefits
factors risks known prevalence outcomes management managing address addresses
support supports help helps improve improves improved reduce reduces reduced
prevent prevents prevented promote promotes maintain maintaining describe
described report reported relating related regards regarding about for in
with among associated relationship role specific common relevant important
older age aged adults adult elderly geriatric the of to and or a an it its
them their they you your this that these those not no also more most other any
""".split())

# discourse connectors that add no content when they open a target phrase
DISCOURSE_WORDS = {
    "moreover", "however", "additionally", "furthermore", "therefore", "thus",
    "hence", "also", "further", "indeed", "notably", "importantly", "similarly",
    "overall", "so", "next", "finally", "firstly", "secondly", "thirdly",
    "first", "second", "third", "yes", "in", "and", "or", "but", "yet", "the",
    "of", "as", "a", "an", "it", "its", "they", "this", "that", "these",
}


def strip_discourse(toks):
    i = 0
    while i < len(toks) and toks[i] in DISCOURSE_WORDS:
        i += 1
    return toks[i:]


def norm_tokens(text):
    return [t for t in re.findall(r"[a-z]+", (text or "").lower())
            if t not in STOPWORDS and len(t) > 1]


def token_match(a, b):
    """Fuzzy content-token match: exact, containment, or shared 5-char prefix."""
    if a == b:
        return True
    if len(a) >= 6 and len(b) >= 6 and a[:5] == b[:5]:
        return True
    if len(a) >= 4 and (a in b or b in a):
        return True
    return False


def content_tokens(text):
    return [t for t in norm_tokens(text) if t not in GENERIC_WORDS]


def fuzzy_overlap(q_toks, gold_toks):
    """Count question content tokens fuzzy-matched in the gold token set."""
    gset = list(set(gold_toks))
    return sum(1 for q in set(q_toks)
               if any(token_match(q, g) for g in gset))


def sha256_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
POP_PATTERNS = [
    r"community[- ]dwelling older adults?",
    r"older adults?",
    r"older people",
    r"older persons?",
    r"older individuals?",
    r"older men",
    r"older women",
    r"older patients?",
    r"nursing[- ]home residents?",
    r"care[- ]home residents?",
    r"long[- ]term care residents?",
    r"elderly",
    r"aging population",
    r"ageing population",
    r"study population",
    r"hospitalized patients?",
    r"patients? with (comorbidities|comorbidity|multimorbidit|heart failure|"
    r"dementia|cognitive impairment|diabetes|stroke)",
]


def extract_population(text, claim=""):
    cand = (claim or "") + " || " + (text or "")[:2000]
    for pat in POP_PATTERNS:
        m = re.search(pat, cand, re.I)
        if m:
            return m.group(0).strip()
    return "older adults"


def split_sentences(text):
    text = re.sub(r"\s+", " ", (text or ""))
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
    return [p.strip() for p in parts if len(p.strip()) > 1]


MARKER_CUES = [
    ("CAUSAL", re.compile(
        r"\b(due to|because of|attribut\w*|result(s|ing)? from|"
        r"lead(s|ing)? to|contribut\w*|aggravat\w*|stem(s|ing)? from|"
        r"driven by|arises? from|explain(ed|s)? by)\b", re.I)),
    ("ASSOCIATION", re.compile(
        r"\b(associated with|correlat\w*|linked to|related to|predictor(s)? of|"
        r"relationship|risk factor(s)? for|increased risk of|"
        r"reduced risk of|higher risk of)\b", re.I)),
    ("RECOMMENDATION", re.compile(
        r"\b(should|recommend\w*|advise\w*|need to|important to|guideline\w*|"
        r"are advised|suggest\w*|may benefit|is recommended)\b", re.I)),
    ("PREVALENCE", re.compile(
        r"\b(prevalence|incidence|proportion|rate(s)? of|estimated|"
        r"reported that|found that)\b", re.I)),
]
QUANT_RE = re.compile(r"\b\d+(?:\.\d+)?(?:\s*(?:%|percent|fold|times))?")
EFFECT_VERBS = {
    "increase", "increases", "increased", "reduce", "reduces", "reduced",
    "improve", "improves", "improved", "prevent", "prevents", "prevented",
    "lower", "higher", "decreased", "decrease", "worsen", "worsened", "risk",
    "benefit", "benefits", "associated", "linked", "related", "contribute",
    "contributes", "contributed", "aggravate", "aggravated", "lead", "leads",
    "effect", "impact",
}
ACTION_VERBS = re.compile(
    r"\b(manage|managing|management|prevent|preventing|prevention|reduce|"
    r"reducing|reduction|improve|improving|promote|promoting|maintain|"
    r"support|supporting|help|helping|address|addressing|treat|treatment|"
    r"intervention|interventions|strategies|approaches|recommend|planning)\b",
    re.I)

TOPIC_NAMES = {
    "C01": "Medication & Polypharmacy",
    "C02": "Dementia & Cognitive Health",
    "C03": "Nutrition",
    "C04": "Physical Activity & Exercise",
    "C05": "Falls & Mobility",
    "C06": "Chronic Disease & Geriatric Conditions",
    "C07": "Mental & Social Wellbeing",
    "C08": "Caregiving",
    "C09": "Preventive Care",
    "C10": "General Healthy Ageing",
}
VALID_TOPICS = set(TOPIC_NAMES)
# ---------------------------------------------------------------------------
# Claim extraction and question building
# ---------------------------------------------------------------------------
def sentence_score(sent, focus_set, is_first):
    toks = norm_tokens(sent)
    n = len(toks)
    if n < 3 or n > 90:
        return 0.0
    s = 0.0
    if QUANT_RE.search(sent):
        s += 4.0
    fm = sum(1 for t in set(toks)
             if any(token_match(t, f) for f in focus_set))
    s += min(fm, 3) * 2.5
    if set(toks) & EFFECT_VERBS:
        s += 2.0
    if 8 <= n <= 60:
        s += 1.5
    if is_first:
        s += 1.0
    low = sent.lower()
    if low.startswith(("table", "figure", "fig.", "supplement", "[", "(")):
        s -= 5.0
    if "|" in sent or low.startswith("rows:"):      # table row content
        s -= 20.0
    if any(ch in sent for ch in ("\u201c", "\u201d", "\u2018", "\u2019",
                                 '"', "'")):
        s -= 4.0                                    # patient quotes / citations
    # methods / design boilerplate is not a claim about the target focus
    if re.search(
        r"(using a .{0,80}(study|approach)|we enrolled|were enrolled|"
        r"were invited to|were recruited|participants? (were|received|had)|"
        r"the aim of|this (study|paper|review|article)( aims?| aimed| was|"
        r"evaluated|examined|investigat)|we conducted|data were (collected|"
        r"obtained)|inclusion criteria|written informed consent|study was "
        r"approved)", low):
        s -= 6.0
    if fm == 0:                                     # claim must touch the focus
        s -= 5.0
    if "http" in low or "doi.org" in low or "creativecommons" in low:
        s -= 3.0
    return s


def detect_shape(sent):
    for shape, rx in MARKER_CUES:
        if rx.search(sent):
            return shape
    if QUANT_RE.search(sent):
        return "PREVALENCE"
    return "DESCRIPTIVE"


def cap_phrase(toks, maxn=8):
    """First <= maxn content tokens joined (deterministic noun-ish phrase cap)."""
    words = strip_discourse(
        [t for t in toks if t not in GENERIC_WORDS])[:maxn]
    return " ".join(words) if words else ""


def strip_citation_bits(text):
    return re.sub(r"\[\s*\d+\s*\]", " ", text)


def clean_phrase(phrase):
    phrase = strip_citation_bits(phrase)
    phrase = re.sub(r"\s+", " ", phrase)
    return phrase.strip(" ,;:.-–—()[]")


def extract_components(claim, focus_set):
    """Return dict: shape, focus (best focus term), population, target phrase,
    has_number, marker_hit. `target` is a claim-specific content phrase that
    makes the reconstructed question distinguish its evidence."""
    shape = detect_shape(claim)
    toks = norm_tokens(claim)
    focus_hits = [t for t in set(toks)
                  if any(token_match(t, f) for f in focus_set)]
    best_focus = max(focus_hits, key=len) if focus_hits else None
    population = extract_population(claim, claim)
    target = ""

    def _distinctive():
        other = [t for t in toks
                 if t not in GENERIC_WORDS
                 and (best_focus is None
                      or not any(token_match(t, f)
                                 for f in (best_focus,)))]
        low = claim.lower()
        # advance cursor past the first effect/causal/statistical marker so the
        # phrase names the OUTCOME, not the clause that introduces it
        m = re.search(
            r"\b(prevalence of|incidence of|proportion of|rate of|reported that|"
            r"reduced risk of|higher risk of|increased risk of|risk of|"
            r"associated with|linked to|compared with|compared to|due to|"
            r"as a result of|result(ing)? in|lead(s|ing)? to|benefit(ed|s)?|"
            r"improved?|increased?|higher|lower|reduced?)\b", low)
        start = m.end() if m else 0
        tail = norm_tokens(claim[start:])
        sel = [t for t in tail
               if t not in GENERIC_WORDS
               and (best_focus is None
                    or not any(token_match(t, f) for f in (best_focus,)))]
        pick = strip_discourse(sel)[:4] if len(sel) >= 2 else other[:4]
        return " ".join(pick)

    if shape in ("CAUSAL", "ASSOCIATION"):
        rx = dict(MARKER_CUES)[shape]
        m = rx.search(claim)
        if m:
            after = clean_phrase(claim[m.end():])
            before = clean_phrase(claim[:m.start()])
            # prefer the segment that does NOT contain the focus term itself
            if best_focus and best_focus in set(norm_tokens(after)):
                target = before
            else:
                target = after
            target = cap_phrase(norm_tokens(target))
        if not target:
            target = _distinctive()
    elif shape == "PREVALENCE":
        target = _distinctive()
        m = QUANT_RE.search(claim)
        if m and not target:
            target = m.group(0).strip()
    else:
        target = _distinctive()
    return {
        "shape": shape,
        "focus": best_focus,
        "population": population,
        "target": target or None,
        "has_number": bool(QUANT_RE.search(claim)),
    }


def build_question(claim, comp):
    shape = comp["shape"]
    focus = comp["focus"] or "the finding"
    pop = comp["population"]
    target = comp.get("target")
    tgt = f" in relation to {target}" if target else ""
    if shape == "CAUSAL":
        if target:
            q = (f"According to the article, how does {focus} relate to "
                 f"{target} in {pop}?")
        else:
            q = (f"According to the article, what is the role of {focus} "
                 f"in {pop}?")
    elif shape == "ASSOCIATION":
        if target:
            q = (f"According to the article, what is the association between "
                 f"{focus} and {target} in {pop}?")
        else:
            q = (f"According to the article, what did the study report about "
                 f"{focus} in {pop}?")
    elif shape == "RECOMMENDATION":
        q = (f"According to the article, what is recommended regarding "
             f"{focus}{tgt} in {pop}?")
    elif shape == "PREVALENCE":
        q = (f"According to the article, what did the study report about "
             f"{focus}{tgt} among {pop}?")
    else:
        q = (f"According to the article, what does the study report about "
             f"{focus}{tgt} in {pop}?")
    return q


def region_of(chunk):
    r = (chunk.get("region") or "").upper()
    if r == "ABSTRACT":
        return "ABSTRACT"
    ct = (chunk.get("content_type") or "").upper()
    if ct == "TABLE":
        return "TABLE"
    if r in ("BODY", "FRONT", "BACK"):
        return "BODY"
    return "OTHER"
# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_gold():
    return json.loads(GOLD_FILE.read_text(encoding="utf-8"))["records"]


def load_chunks():
    by_id, by_pmc = {}, defaultdict(list)
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            by_id[c["chunk_id"]] = c
            by_pmc[c["pmcid"]].append(c)
    return by_id, by_pmc


def load_emb():
    row_mapping = json.loads(ROW_MAP_FILE.read_text(encoding="utf-8"))
    arr = np.load(EMB_FILE, mmap_mode="r")
    idx = {cid: i for i, cid in enumerate(row_mapping)}
    return arr, idx


def embed_texts(model, texts):
    if not texts:
        return None
    return model.encode(
        texts, batch_size=32, convert_to_numpy=True,
        normalize_embeddings=True).astype(np.float32)


def gold_text(chunk_id):
    return (CHUNKS_BY_ID.get(chunk_id) or {}).get("text") or ""


def gold_vec(chunk_id, arr, idx, model, cache):
    if chunk_id in idx:
        return np.asarray(arr[idx[chunk_id]], dtype=np.float32)
    if chunk_id not in cache:
        cache[chunk_id] = embed_texts(model, [gold_text(chunk_id)])[0]
    return cache[chunk_id]
def decide_record(rec, by_pmc, arr, idx, model, embed_cache):
    q = (rec.get("revised_question") or rec.get("question") or "").strip()
    q_toks = norm_tokens(q)
    q_content = content_tokens(q)
    q_cset = set(q_content)
    gold_ids = list(rec.get("gold_relevant_chunk_ids") or [])
    gid0 = gold_ids[0] if gold_ids else None
    gchunk = CHUNKS_BY_ID.get(gid0) or {}
    gtext = (gchunk.get("text") or "").strip()
    g_toks = norm_tokens(gtext)
    pmcid = rec.get("pmcid")

    lex = (fuzzy_overlap(q_content, g_toks) / len(q_cset)
           if q_cset else 0.0)
    vq = vsem = None
    if q and gtext:
        vq = embed_texts(model, [q])[0]
        vg = gold_vec(gid0, arr, idx, model, embed_cache)
        vsem = float(vq @ vg)

    # same-article competition (Task 10E style: content-token intersection)
    comp = 0
    for c in by_pmc.get(pmcid, []):
        if c["chunk_id"] == gid0:
            continue
        if q_cset & set(norm_tokens(c.get("text") or "")):
            comp += 1

    # ----- claim extraction from the gold chunk -----
    sents = split_sentences(gtext)
    scored = [(sentence_score(s, q_cset, i == 0), i, s)
              for i, s in enumerate(sents)]
    scored.sort(key=lambda x: (-x[0], x[1]))
    claim, claim_score = None, 0.0
    if scored and scored[0][0] > 0:
        claim, claim_score = scored[0][2], float(scored[0][0])
    comps = (extract_components(claim, q_cset)
             if claim else {"shape": None, "focus": None,
                            "population": "older adults"})

    focus_in_claim = False
    if comps.get("focus") and claim:
        claim_toks = set(norm_tokens(claim))
        focus_in_claim = any(
            token_match(comps["focus"], t) for t in claim_toks)

    specific = len(q_cset) >= 2
    strong = (lex >= RETAIN_LEX and vsem is not None
              and vsem >= RETAIN_SEM and focus_in_claim and specific)
    action_q = bool(ACTION_VERBS.search(q))
    claim_has_rec = bool(comps.get("shape") == "RECOMMENDATION"
                         or (claim and MARKER_CUES[2][1].search(claim)))
    answered = (not action_q) or claim_has_rec
    base = {
        "original_v1_0_id": rec["final_benchmark_id"],
        "original_candidate_id": rec.get("original_candidate_id"),
        "original_question": q,
        "pmcid": pmcid,
        "pmid": rec.get("pmid"),
        "title": rec.get("title"),
        "topic": rec.get("primary_topic"),
        "secondary_topics": list(rec.get("secondary_topics") or []),
        "evidence_type": rec.get("evidence_type"),
        "section_id": gchunk.get("section_id") or rec.get("section_id"),
        "section_title": gchunk.get("section_title") or rec.get("section_title"),
        "source_locator": gchunk.get("source_locator") or rec.get("source_locator"),
        "source_document": f"{pmcid} ({rec.get('title')})",
        "claim_sentence": claim,
        "claim_score": round(claim_score, 4),
        "claim_shape": comps.get("shape"),
        "focus_phrase": comps.get("focus"),
        "population": comps.get("population"),
        "lex_overlap_ratio": round(lex, 6),
        "sem_sim": round(vsem, 6) if vsem is not None else None,
        "competing_chunks_in_article": comp,
        "gold_region": region_of(gchunk),
        "q_content_token_count": len(q_cset),
        "gold_token_count": len(g_toks),
    }
    status = question = reason = None
    gold_final = gold_ids
    if strong and answered:
        status, question, reason = "RETAINED", q, None
    elif (claim and claim_score >= RECONSTRUCT_MIN_CLAIM
          and comps.get("focus")):
        status, question, reason, gold_final = _try_build(
            claim, comps, gid0, g_toks, arr, idx, model, embed_cache,
            lex, vsem, base)
    else:
        status, question, reason, gold_final = _try_reanchor(
            pmcid, by_pmc, gid0, q_cset, claim_score, arr, idx, model,
            embed_cache, base)
        if status == "EXCLUDE":
            reason = ("no substantive claim in v1.0 gold chunk (score %.2f); " %
                      claim_score) + (reason or "")
    base.update({"reconstruction_status": status,
                 "reconstruction_reason": reason})
    return base, status, question, gold_final


def _try_build(claim, comps, gid0, g_toks, arr, idx, model, cache,
               lex, vsem, base):
    question = build_question(claim, comps)
    nq_cset = set(content_tokens(question))
    nlex = (fuzzy_overlap(list(nq_cset), g_toks) / len(nq_cset)
            if nq_cset else 0.0)
    nvec = embed_texts(model, [question])[0]
    nsem = float(nvec @ gold_vec(gid0, arr, idx, model, cache))
    if nlex >= GATE_LEX and nsem >= GATE_SEM:
        base["lex_overlap_ratio"] = round(nlex, 6)
        base["sem_sim"] = round(nsem, 6)
        return ("RECONSTRUCTED", question,
                "reconstructed evidence-first; original v1.0 question had "
                "weak evidence alignment (lex=%.3f sem=%.3f) or was not "
                "answered by the gold claim" % (lex, vsem or 0.0),
                [gid0])
    return ("EXCLUDE", None,
            "reconstructed question failed answerability gate (lex=%.3f "
            "sem=%.3f)" % (nlex, nsem), None)


def _try_reanchor(pmcid, by_pmc, gid0, q_cset, claim_score, arr, idx,
                  model, cache, base):
    best = None
    for c in by_pmc.get(pmcid, []):
        if c["chunk_id"] == gid0:
            continue
        sidx = [(-sentence_score(s, q_cset, i == 0), i, s)
                for i, s in enumerate(split_sentences(c.get("text") or ""))]
        sidx.sort()
        if sidx and -sidx[0][0] >= REANCHOR_FLOOR:
            if best is None or -sidx[0][0] > best[0]:
                best = (-sidx[0][0], c["chunk_id"], sidx[0][2])
    if not best:
        return "EXCLUDE", None, (
            "no defensible same-article alternative claim"), None
    nid, nclaim = best[1], best[2]
    ncomp = extract_components(nclaim, q_cset)
    nquestion = build_question(nclaim, ncomp)
    ntext = gold_text(nid)
    ntoks = norm_tokens(ntext)
    nqset = set(content_tokens(nquestion))
    nlex = (fuzzy_overlap(list(nqset), ntoks) / len(nqset)
            if nqset else 0.0)
    nvec = embed_texts(model, [nquestion])[0]
    nsem = float(nvec @ gold_vec(nid, arr, idx, model, cache))
    if nlex >= GATE_LEX and nsem >= GATE_SEM:
        nc = CHUNKS_BY_ID.get(nid, {})
        base.update({
            "lex_overlap_ratio": round(nlex, 6),
            "sem_sim": round(nsem, 6),
            "claim_sentence": nclaim,
            "claim_score": round(best[0], 4),
            "claim_shape": ncomp.get("shape"),
            "focus_phrase": ncomp.get("focus"),
            "population": ncomp.get("population"),
            "section_id": nc.get("section_id"),
            "section_title": nc.get("section_title"),
            "source_locator": nc.get("source_locator"),
            "gold_region": region_of(nc),
        })
        return ("RECONSTRUCTED", nquestion,
                "re-anchored to same-article chunk '%s' whose claim supports "
                "a defensible question; original v1.0 gold chunk had no "
                "substantive claim (score %.2f)" % (nid, claim_score), [nid])
    return ("EXCLUDE", None,
            "same-article re-anchor failed answerability gate (lex=%.3f "
            "sem=%.3f)" % (nlex, nsem), None)
    if scored and scored[0][0] > 0:
        claim, claim_score = scored[0][2], float(scored[0][0])
    comps = (extract_components(claim, q_cset)
             if claim else {"shape": None, "focus": None})

    focus_in_claim = False
    if comps.get("focus") and claim:
        claim_toks = set(norm_tokens(claim))
        focus_in_claim = any(
            token_match(comps["focus"], t) for t in claim_toks)

    specific = len(q_cset) >= 2
    strong = (lex >= RETAIN_LEX and vsem is not None
              and vsem >= RETAIN_SEM and focus_in_claim and specific)
    action_q = bool(ACTION_VERBS.search(q))
    claim_has_rec = bool(comps.get("shape") == "RECOMMENDATION"
                         or (claim and MARKER_CUES[2][1].search(claim)))
    answered = (not action_q) or claim_has_rec
# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------
def stats(vals):
    vals = sorted(float(v) for v in vals)
    if not vals:
        return {"n": 0, "min": None, "max": None, "mean": None, "median": None}
    return {"n": len(vals), "min": round(vals[0], 6),
            "max": round(vals[-1], 6),
            "mean": round(sum(vals) / len(vals), 6),
            "median": round(vals[len(vals) // 2], 6)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def detect_near_dups(cands):
    exact, near = [], []
    seen = {}
    for i, c in enumerate(cands):
        q = c["question"].lower()
        if q in seen:
            exact.append((c["candidate_id"], seen[q]))
        else:
            seen[q] = c["candidate_id"]
    ts = [set(content_tokens(c["question"])) for c in cands]
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            if jaccard(ts[i], ts[j]) >= 0.8:
                near.append((cands[i]["candidate_id"],
                             cands[j]["candidate_id"]))
                break
    return exact, near


def template_families(cands):
    fam = Counter()
    for c in cands:
        q = c["question"].lower()
        if q.startswith("according to the article, how does"):
            fam["how_does_FOCUS_relate_to"] += 1
        elif q.startswith("according to the article, what is the association"):
            fam["association_between"] += 1
        elif q.startswith("according to the article, what is the role"):
            fam["role_of"] += 1
        elif q.startswith("according to the article, what is recommended"):
            fam["recommended_regarding"] += 1
        elif q.startswith("according to the article, what did the study report"):
            fam["study_reported_about"] += 1
        elif q.startswith("according to the article, what does the study report"):
            fam["study_reports_about"] += 1
        else:
            fam["retained_v1_0"] += 1
    return dict(fam)
def _disambiguate(q, base):
    """Deterministic fallback: insert the claim's first content token not yet
    present in the question, so colliding questions stay evidence-specific."""
    claim_toks = norm_tokens(base.get("claim_sentence") or "")
    q_toks = set(content_tokens(q))
    extra = None
    for t in claim_toks:
        if t in GENERIC_WORDS or t in q_toks:
            continue
        extra = t
        break
    if not extra:
        return q + " (reported in this study)"
    if " in relation to " in q:
        return q.replace(" in relation to ", " in relation to %s and " % extra, 1)
    if " between " in q and " and " in q:
        qnew = q.rsplit(" and ", 1)
        return qnew[0] + " and " + extra + " " + qnew[1]
    return q[:-1] + f" in relation to {extra}?"


def _make_rationale(base, status, rec):
    claim = base.get("claim_sentence")
    if status == "RETAINED":
        return (rec.get("evidence_rationale")
                or "Retained v1.0 question with strong alignment to the gold "
                "evidence.")
    if claim:
        return ("The gold evidence states: \"%s\". The candidate question was "
                "reconstructed deterministically from this claim so it is "
                "answerable from the selected evidence." % claim[:600])
    return ("Candidate reconstructed from the same-article evidence unit; "
            "answerable from the selected gold chunk.")


def main():
    import hashlib
    h0 = {k: sha256_file(GERI_DIR / p) for k, p in [
        ("chunks.jsonl", "chunks/chunks.jsonl"),
        ("embeddings.npy", "index/embeddings.npy"),
        ("faiss_index.bin", "index/faiss_index.bin"),
        ("bm25.pkl", "index/bm25.pkl"),
        ("row_mapping.json", "index/row_mapping.json")]}
    g0 = sha256_file(GOLD_FILE)
    assert g0 == GOLD_SHA_EXPECTED, f"v1.0 gold SHA changed: {g0}"
    for k, v in h0.items():
        assert v == FROZEN_EXPECTED[k], f"frozen {k} changed: {v}"
    frozen_before = {"gold": g0, "frozen": h0}

    # ---------------- load data ----------------
    from sentence_transformers import SentenceTransformer  # noqa: E402
    global CHUNKS_BY_ID
    CHUNKS_BY_ID, by_pmc = load_chunks()
    arr, idx = load_emb()
    model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    embed_cache = {}

    recs = load_gold()
    assert len(recs) == 134

    # ---------------- per-record decision ----------------
    included, excluded, decisions = [], [], {}
    for rec in recs:
        base, status, question, gold_final = decide_record(
            rec, by_pmc, arr, idx, model, embed_cache)
        decisions[rec["final_benchmark_id"]] = status
        if status in ("RETAINED", "RECONSTRUCTED"):
            included.append((rec, base, status, question, gold_final))
        else:
            excluded.append(base)

    # ---------------- build candidate records (raw IDs) ----------------
    raw = []
    seen_questions = set()
    for idx, (rec, base, status, question, gold_final) in enumerate(
            included, start=1):
        rationale = _make_rationale(base, status, rec)
        q = question.strip()
        if q.lower() in seen_questions:
            q2 = _disambiguate(q, base)
            if q2.lower() in seen_questions:
                # unrecoverable exact-duplicate after re-anchor onto the same
                # evidence -> exclude deterministically, not fail the run
                excluded.append({
                    "original_v1_0_id": base["original_v1_0_id"],
                    "candidate_id": "RAW%03d" % idx,
                    "reconstruction_status": "EXCLUDE_DUPLICATE_UNRESOLVED",
                    "reconstruction_reason": (
                        "question is an exact duplicate of an earlier "
                        "candidate anchored to the same evidence; "
                        "disambiguation could not separate them"),
                })
                decisions[base["original_v1_0_id"]] = \
                    "EXCLUDE_DUPLICATE_UNRESOLVED"
                continue
            q = q2
        seen_questions.add(q.lower())
        raw.append({
            "candidate_id": "RAW%03d" % idx,
            "original_v1_0_id": base["original_v1_0_id"],
            "original_candidate_id": base["original_candidate_id"],
            "original_question": base["original_question"],
            "question": q,
            "pmcid": base["pmcid"],
            "pmid": base.get("pmid"),
            "title": base["title"],
            "topic": base["topic"],
            "secondary_topics": base["secondary_topics"],
            "gold_relevant_chunk_ids": gold_final,
            "source_document": base["source_document"],
            "section_id": base["section_id"],
            "section_title": base["section_title"],
            "source_locator": base["source_locator"],
            "evidence_type": base["evidence_type"],
            "evidence_rationale": rationale,
            "reconstruction_status": status,
            "reconstruction_reason": base["reconstruction_reason"],
            "review_status": "PENDING_HUMAN_REVIEW",
            "diagnostics": {
                k: base[k] for k in (
                    "claim_sentence", "claim_score", "claim_shape",
                    "focus_phrase", "population", "lex_overlap_ratio",
                    "sem_sim", "competing_chunks_in_article", "gold_region",
                    "q_content_token_count", "gold_token_count",
                )
            },
        })

    # ---------------- post-hoc near-duplicate dedup ----------------
    # two candidates that share the SAME gold chunk AND near-identical question
    # text are accidental duplicates of one evidence unit; keep the first
    kept, dropped, dedup_dropped = [], [], []
    for c in raw:
        dup_of = None
        c_toks = set(content_tokens(c["question"]))
        for k in kept:
            if (k["gold_relevant_chunk_ids"] == c["gold_relevant_chunk_ids"]
                    and jaccard(c_toks,
                                set(content_tokens(k["question"]))) >= 0.7):
                dup_of = k["candidate_id"]
                break
        if dup_of:
            dropped.append(c)
            dedup_dropped.append({
                "original_v1_0_id": c["original_v1_0_id"],
                "candidate_id": c["candidate_id"],
                "reconstruction_status": "EXCLUDE_DUPLICATE",
                "reconstruction_reason": (
                    "near-duplicate of %s on the same gold chunk (question "
                    "Jaccard >= 0.7); kept the first occurrence" % dup_of),
            })
        else:
            kept.append(c)

    # ---------------- assign final deterministic IDs ----------------
    cands = []
    for i, c in enumerate(kept, start=1):
        c["candidate_id"] = "GLG11-C%03d" % i
        cands.append(c)
    excluded += dedup_dropped
    for d in dedup_dropped:
        decisions[d["original_v1_0_id"]] = "EXCLUDE_DUPLICATE"
# ---------------- diagnostics ----------------
    qlens = stats([len(norm_tokens(c["question"])) for c in cands])
    qclens = stats([c["diagnostics"]["q_content_token_count"] for c in cands])
    gold_lens = stats([c["diagnostics"]["gold_token_count"] for c in cands])
    lex_all = stats([c["diagnostics"]["lex_overlap_ratio"] for c in cands])
    sem_all = stats([c["diagnostics"]["sem_sim"] for c in cands
                     if c["diagnostics"]["sem_sim"] is not None])
    comp_all = stats([c["diagnostics"]["competing_chunks_in_article"]
                      for c in cands])
    n_gold_per_q = Counter(
        len(c["gold_relevant_chunk_ids"]) for c in cands)
    region_dist = Counter(c["diagnostics"]["gold_region"] for c in cands)
    topic_dist = Counter(c["topic"] for c in cands)
    status_dist = Counter(c["reconstruction_status"] for c in cands)
    n_pmc = len({c["pmcid"] for c in cands})
    n_chunks = len({x for c in cands
                    for x in c["gold_relevant_chunk_ids"]})
    exact_dups, near_dups = detect_near_dups(cands)
    template_fam = template_families(cands)

    # frozen hashes AFTER (must be identical)
    h1 = {k: sha256_file(GERI_DIR / p) for k, p in [
        ("chunks.jsonl", "chunks/chunks.jsonl"),
        ("embeddings.npy", "index/embeddings.npy"),
        ("faiss_index.bin", "index/faiss_index.bin"),
        ("bm25.pkl", "index/bm25.pkl"),
        ("row_mapping.json", "index/row_mapping.json")]}
    g1 = sha256_file(GOLD_FILE)
    frozen_after = {"gold": g1, "frozen": h1}
    assert g1 == GOLD_SHA_EXPECTED and all(
        h1[k] == FROZEN_EXPECTED[k] for k in FROZEN_EXPECTED)

    cand_doc = {
        "dataset": "ElderDocAI-GeriLit-Gold-v1.1-CANDIDATE",
        "status": "PENDING_HUMAN_REVIEW",
        "task": "phase3_task10f",
        "version": "1.1-candidate",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "metadata/task10f_benchmark_reconstruction.py",
        "description": (
            "Evidence-first CANDIDATE reconstruction derived from the 134 "
            "GeriLit-Gold v1.0 records. Records are PENDING_HUMAN_REVIEW; the "
            "v1.0 frozen benchmark is unchanged. NO retrieval evaluation was "
            "run against this candidate set."),
        "candidate_count": len(cands),
        "v1_0_unchanged_sha256": GOLD_SHA_EXPECTED,
        "candidates": cands,
    }
    CAND_FILE.write_text(json.dumps(cand_doc, indent=2), encoding="utf-8")

    diag = {
        "task": "phase3_task10f",
        "generated_at_utc": cand_doc["generated_at_utc"],
        "generated_by": "metadata/task10f_benchmark_reconstruction.py",
        "frozen_before": frozen_before,
        "frozen_after": frozen_after,
        "frozen_unchanged": (
            frozen_before == frozen_after and
            frozen_after["gold"] == GOLD_SHA_EXPECTED),
        "v1_0_processed": 134,
        "decisions": decisions,
        "summary": {
            "candidates": len(cands),
            "retained": status_dist.get("RETAINED", 0),
            "reconstructed": status_dist.get("RECONSTRUCTED", 0),
            "reanchored": sum(1 for c in cands if c["reconstruction_status"]
                              == "RECONSTRUCTED"
                              and "re-anchored" in (c["reconstruction_reason"]
                                                    or "")),
            "excluded": len(excluded),
            "unique_pmcid": n_pmc,
            "unique_gold_chunks": n_chunks,
            "topics": dict(sorted(topic_dist.items())),
            "regions": dict(sorted(region_dist.items())),
            "gold_per_question": dict(sorted(n_gold_per_q.items())),
            "question_token_count": qlens,
            "content_token_count": qclens,
            "gold_token_count": gold_lens,
            "lex_overlap": lex_all,
            "sem_sim": sem_all,
            "competition": comp_all,
            "exact_duplicate_questions": len(exact_dups),
            "near_duplicate_pairs": len(near_dups),
            "template_families": template_fam,
        },
        "exclusions": excluded,
        "candidate_signature_sha256": hashlib.sha256(
            json.dumps(cands, sort_keys=True).encode()).hexdigest(),
    }
    DIAG_FILE.write_text(json.dumps(diag, indent=2), encoding="utf-8")

    print(json.dumps({
        "candidates": len(cands),
        "retained": status_dist.get("RETAINED", 0),
        "reconstructed": status_dist.get("RECONSTRUCTED", 0),
        "excluded": len(excluded),
        "unique_pmcid": n_pmc,
        "unique_gold_chunks": n_chunks,
        "topics": dict(sorted(topic_dist.items())),
        "frozen_unchanged": diag["frozen_unchanged"],
        "v1_0_sha_ok": g1 == GOLD_SHA_EXPECTED,
        "outfile": str(CAND_FILE),
    }, indent=2))


CHUNKS_BY_ID = {}


if __name__ == "__main__":
    main()