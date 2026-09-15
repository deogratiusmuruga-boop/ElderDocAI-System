"""
ElderDocAI - Extended Gold QA Set Builder v2 (KB-anchored, anchor-verified).

Purpose
-------
Raise statistical power for the RQ2/ablation/LLM-baseline evaluations by
expanding the protected 16 gold questions into a larger, KB-anchored,
reproducible set drawn from the system's six public source documents:

  - NIA "Understanding Memory Loss"
  - NIA "Exercise and Physical Activity for Older Adults"
  - NIA "Tips to Take Medicines Safely"
  - NIA "Caregiver's Handbook"
  - USDA/HHS "Dietary Guidelines for Americans 2020-2025"
  - WHO "ICOPE Handbook" (2nd ed.)

Revision (v2) - methodological hardening
----------------------------------------
1. ANCHOR-LEVEL ANSWERABILITY (replaces the old source-only gate)
   A generated question is accepted ONLY when the production hybrid
   retriever (scripts.hybrid_retriever.hybrid_search, top-3) surfaces ALL of
   the following simultaneously:
       * the expected SOURCE DOCUMENT          -> source_correct
       * the expected ANCHOR CHUNK ID         -> chunk_recall
       * the SUPPORTING SPAN from that chunk  -> supporting_span_supported
   These three gates mirror the CURRENT evaluation metrics implemented by
   scripts/evaluate_gold_qa.py and scripts/evaluate_ablation.compute_grounding
   exactly (same normalizer, same 0.85 token-coverage rule). The extended set
   therefore targets the same measurements the evaluator will use.

2. EVIDENCE INTEGRITY
   Every accepted question carries an explicit chain in the output JSON:
       QUESTION -> reference_answer == supporting_span (verbatim)
                -> anchor chunk_id -> source document
   The supporting span must literally exist inside its anchor chunk before
   any LLM call is made (span_in_anchor_chunk = True + char offsets).

3. QUESTION-QUALITY SCREENING (two independent stages)
   * deterministic screen: length, question form, yes/no, answer leakage
     (ordered/verbatim containment), ambiguous anaphora, medical-judgment
     requests, low information, on-topic anchoring.
   * LLM screen (a SEPARATE role from generation): ambiguity, malformation,
     answerability from the span alone, verbatim echo/restatement, scope.
   Generation (formulate_question), screening (quality_screen_llm) and
   evidence verification (verify_anchor) are fully separate functions with
   separate prompts; the same model output never validates itself.

4. DEDUPLICATION (beyond token overlap)
   - exact + Jaccard near-duplicate checks on normalized questions
   - fact-level dedup on supporting-span tokens (same-fact questions)
   - per-anchor-chunk cap so adjacent sentences cannot flood one chunk
   The original 16 questions are READ-ONLY and are used as a protected
   deduplication baseline (GOLD16_PROTECT_JACCARD).

5. DISTRIBUTION
   Capped, sqrt-proportional per-document allocation with a floor for every
   domain. No document may exceed ALLOC_CAP (default 22 of 120 = 18.3%) and
   no single document or pair of documents may dominate the set.

6. REPRODUCIBILITY
   Fixed seed, deterministic document order, full configuration recorded in
   the output metadata (seed, target, allocation, models, sampling,
   retrieval, quality-screen, dedup, source dataset + sha256, timestamp).

Outputs (new files only)
------------------------
  - data/gold_qa_extended.json
  - data/evaluation_results/gold_extension/curation_report.json
  - data/evaluation_results/gold_extension/gold_extension_console.txt

The original data/gold_qa_evaluation.json and all production modules,
datasets, configurations and previous results are NOT modified.
"""
import os
import re
import json
import time
import random
import hashlib
import datetime
import argparse
from collections import defaultdict

# Windows/multiprocessing safety: prevent HF-tokenizers/sentence-transformers
# from spawning a re-entrant child that re-imports this module at startup
# (this previously caused a deadlock hang at CrossEncoder init).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import multiprocessing as _mp  # noqa: E402
try:
    _mp.freeze_support()
except Exception:  # noqa: BLE001
    pass

import ollama  # noqa: E402

# Production retriever (read-only reuse; no configuration is changed here).
# FINAL_RESULTS is the production top-k used by every evaluation script.
from scripts.hybrid_retriever import hybrid_search, FINAL_RESULTS  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNK_FILE = os.path.join(BASE_DIR, "data", "chunks", "knowledge_base_chunks.json")
GOLD16_FILE = os.path.join(BASE_DIR, "data", "gold_qa_evaluation.json")
OUT_JSON = os.path.join(BASE_DIR, "data", "gold_qa_extended.json")
OUT_DIR = os.path.join(BASE_DIR, "data", "evaluation_results", "gold_extension")
CURATION_JSON = os.path.join(OUT_DIR, "curation_report.json")
CONSOLE_TXT = os.path.join(OUT_DIR, "gold_extension_console.txt")

# ---------------------------------------------------------------------------
# Configuration (recorded verbatim in the output metadata)
# ---------------------------------------------------------------------------
TARGET_QUESTIONS = 120
RNG_SEED = 20260907

# Candidate sentence heuristics (unchanged from v1 so the measured pool is
# comparable). MAX_ANSWER_SPAN_LEN additionally caps the answer length.
MIN_SENT_LEN = 80
MAX_SENT_LEN = 500
MAX_ANSWER_SPAN_LEN = 260

# Generation
QUESTION_MODEL = "llama3.2:latest"
QUESTION_SAMPLING = {"temperature": 0, "top_p": 0.1, "top_k": 10}

# Quality screening (LLM stage thresholds; deterministic stage below)
QUALITY_SCREEN_MODEL = "llama3.2:latest"
QUALITY_SCREEN_SAMPLING = {"temperature": 0, "top_p": 0.1, "top_k": 10}
MIN_QUESTION_LEN = 10                 # meaningful alphanumeric characters
MIN_QUESTION_CONTENT_TOKENS = 3       # content tokens after stopword removal
MAX_PER_CHUNK = 2                     # accepted questions per anchor chunk
ANSWER_LEAK_MIN_TOKENS = 4            # min question tokens for leak substring check

# Anchor-level answerability (mirrors scripts/evaluate_gold_qa.py)
RETRIEVAL_TOP_K = int(FINAL_RESULTS)  # 3 == production hybrid_search top-k
SPAN_MIN_TOKEN_COVERAGE = 0.85

# Deduplication
GOLD16_PROTECT_JACCARD = 0.45   # span/fact overlap with original 16 -> skip
SPAN_DEDUP_JACCARD = 0.85       # two facts are "the same" above this
QUESTION_DEDUP_JACCARD = 0.80   # near-identical question above this

# Distribution (capped sqrt-proportional allocation)
ALLOC_FLOOR = 15                # every domain with candidates gets >= floor
ALLOC_CAP = 22                  # no document exceeds this cap
ALLOC_WEIGHT_POWER = 0.5        # weight = pool_size ** power (0.5 == sqrt)
# ---------------------------------------------------------------------------
# Span / doc normalization - MIRRORS scripts/evaluate_gold_qa.py verbatim so
# acceptance is measured with exactly the same math the evaluator uses.
# ---------------------------------------------------------------------------
SOFT_HYPHEN = "\u00ad"
ZERO_WIDTH = "\u200b\u200c\u200d\u200e\u200f\u2060\ufeff"


def normalize_for_span(text):
    """Robust text normalization for supporting-span matching (copied from
    scripts/evaluate_gold_qa.py to guarantee metric parity)."""
    if not text:
        return ""
    t = str(text).casefold()
    # Soft hyphen + any adjacent whitespace: the pieces belong to one word.
    t = re.sub(r"\s*" + SOFT_HYPHEN + r"\s*", "", t)
    # Remaining zero-width / format control chars.
    t = re.sub("[" + ZERO_WIDTH + "]", "", t)
    # PDF line-break hyphenation: hyphen at end-of-line joins the word.
    t = re.sub(r"-\s*\n\s*", "", t)
    # Collapse all other punctuation / spacing to a single ASCII space.
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return t.strip()


SPAN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do",
    "does", "for", "from", "has", "have", "how", "i", "in", "into", "is",
    "it", "its", "may", "of", "on", "or", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "will", "with", "you",
    "your",
}

QUESTION_EXTRA_STOPWORDS = {
    "should", "would", "did", "am", "me", "my", "about", "please", "tell",
    "know", "need", "want", "him", "her", "his", "and", "or", "if", "then",
    "than", "so", "just", "really", "actually", "someone", "somebody",
}


def content_tokens(text):
    """Content-word tokens (stopwords removed) of a normalized string."""
    toks = normalize_for_span(text).split()
    stop = SPAN_STOPWORDS | QUESTION_EXTRA_STOPWORDS
    return {t for t in toks if t not in stop}


def span_token_coverage(span, evidence_text):
    """Fraction of gold-span content tokens present in the evidence text
    (mirrors scripts/evaluate_gold_qa.span_token_coverage)."""
    span_tokens = [
        tok for tok in normalize_for_span(span).split()
        if tok not in SPAN_STOPWORDS
    ]
    evidence_tokens = set(normalize_for_span(evidence_text).split())
    if not span_tokens:
        return 1.0 if span else 0.0
    hits = sum(1 for tok in span_tokens if tok in evidence_tokens)
    return round(hits / len(span_tokens), 4)


def normalize_doc(doc):
    """Document-name normalization used by the evaluators."""
    s = str(doc).lower().replace(".pdf", "").strip()
    return "".join(ch for ch in s if ch.isalnum())


def token_jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0
SKIP_PATTERNS = [
    r"table of contents", r"\bcopyright\b", r"disclaimer\b",
    r"publication may be viewed", r"www\.", r"http", r"usda",
    r"civil rights", r"sovereign immunity", r"non-discrimination",
    r"reasonable accommodation", r"creative commons", r"licen[cs]e",
    r"suggested citation", r"for more information", r"order form",
    r"complaint", r"what.s inside", r"\.\.\.", r"note: this", r"acknowledg",
]

DOMAIN_HINTS = [
    "should", "recommend", "aim for", "at least", "per week", "per day",
    "exercise", "medicine", "medication", "caregiver", "dementia", "memory",
    "icope", "arthrit", "balance", "stron", "bone", "diet", "nutrition",
    "sodium", "protein", "fiber", "older adult", "age", "fall", "walk",
    "strength", "health", "doctor", "symptom", "attention", "aging",
    "support", "screen", "assessment", "activity", "physical",
]

# Question-form heuristics (deterministic quality screen)
YES_NO_STARTS = (
    "is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ",
    "could ", "should ", "will ", "would ", "may ", "might ", "has ",
    "have ", "had ", "am ",
)

ANAPHORA_STARTS = {
    "it", "this", "these", "that", "they", "he", "she", "her", "him",
    "them", "his", "its", "their",
}

MEDICAL_JUDGMENT_PATTERNS = [
    re.compile(r"diagnos"),
    re.compile(r"prescrib"),
    re.compile(r"(which|what) (medication|medicine|drug|pill)"),
    re.compile(r"risk of (dying|death|mortality|falling|complication)"),
    re.compile(r"prognos"),
    re.compile(r"life expectancy"),
    re.compile(r"will (the patient|this patient|my (mother|father|parent|mom|dad|grandmother|grandfather)|he|she|they) (die|worsen|get worse|recover|survive)"),
    re.compile(r"is (the patient|this patient|my (mother|father|mom|dad)) (stable|unstable|ok|okay|fine)"),
    re.compile(r"should (i|we) (take|stop|start|increase|decrease|change|switch)"),
    re.compile(r"do (i|we) (have|need) (a )?(prescription|diagnos)"),
    re.compile(r"how much (should|could|can) (i|we) (take|give)"),
    re.compile(r"is it safe for (me|him|her|the patient)"),
    re.compile(r"am (i|he|she|the patient) (at risk|ok|okay|fine|stable)"),
    re.compile(r"predict"),
]


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"\u201c])", text)
    return [p.strip() for p in parts if p.strip()]


def is_candidate(sent):
    s = sent.strip()
    if len(s) < MIN_SENT_LEN or len(s) > MAX_SENT_LEN:
        return False
    low = s.lower()
    if any(re.search(p, low) for p in SKIP_PATTERNS):
        return False
    if not any(h in low for h in DOMAIN_HINTS):
        return False
    return True


def norm_tokens(text):
    return set(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())


def load_chunks():
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_candidate_pool(chunks):
    """Extract (doc, chunk_id, sentence) candidates passing content heuristics
    (unchanged from v1, so the measured pool size is comparable)."""
    pool = []
    for c in chunks:
        sents = split_sentences(c.get("text") or "")
        for s in sents:
            if is_candidate(s):
                pool.append({
                    "source_document": c.get("source_document"),
                    "chunk_id": c.get("chunk_id"),
                    "sentence": s,
                })
    seen = set()
    uniq = []
    for item in pool:
        sig = tuple(sorted(norm_tokens(item["sentence"])))
        if not sig:
            continue
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(item)
    return uniq


def load_gold16():
    """Read-only access to the original benchmark. Returns (questions,
    span-token-sets, question-token-sets). Never written."""
    with open(GOLD16_FILE, "r", encoding="utf-8") as f:
        d = json.load(f)
    spans = []
    qtoks = []
    for q in d.get("gold_questions", []):
        sp = q.get("supporting_span") or q.get("reference_answer") or ""
        spans.append(content_tokens(sp))
        qtoks.append(content_tokens(q.get("question", "")))
    return d, spans, qtoks
QUESTION_PROMPT = """You are a question-formation assistant for a health
knowledge base used by an elderly-care assistant.

Given the FACT below (a statement from a public health document), write ONE
natural user question that this FACT directly answers.

Constraints:
- The question must be answerable ONLY from the FACT (do not add outside facts).
- Do NOT include the answer in the question.
- Ask like a caregiver or older adult would (plain, friendly, specific).
- Return ONLY JSON: {{"question": "..."}}

FACT:
{fact}
"""


def formulate_question(sentence, retry=2):
    """[GENERATION ROLE] Ask llama3.2 to produce a natural user question for a
    KB fact. The model output here is never used to validate itself: it is
    checked by the deterministic screen, the LLM quality screen and the
    retrieval-based anchor verification below."""
    last_err = None
    for _ in range(retry + 1):
        try:
            prompt = QUESTION_PROMPT.format(fact=sentence)
            resp = ollama.chat(
                model=QUESTION_MODEL,
                options=dict(QUESTION_SAMPLING),
                messages=[
                    {"role": "system",
                     "content": "You produce JSON only."},
                    {"role": "user", "content": prompt},
                ],
            )
            text = resp["message"]["content"].strip()
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not m:
                last_err = "no-json"
                continue
            q = json.loads(m.group(0)).get("question", "").strip()
            if len(q) < MIN_QUESTION_LEN:
                last_err = "question-too-short"
                continue
            return q, None
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(1)
    return None, last_err


def quality_screen_deterministic(q, span, chunk_text):
    """[QUALITY SCREEN - DETERMINISTIC] Fast, transparent rejection rules.
    Returns (accepted: bool, reason: str|None)."""
    r = q.strip()
    if not r:
        return False, "empty-question"
    alnum = re.sub(r"[^A-Za-z0-9]+", "", r)
    if len(alnum) < MIN_QUESTION_LEN:
        return False, "too-short"
    if not r.endswith("?"):
        return False, "not-a-question"
    if re.search(r"\.\.\.|___|TODO|N/A|\{\}|\[.*\]", r):
        return False, "malformed"
    low = r.lower()
    if low.startswith(YES_NO_STARTS):
        return False, "yes-no"
    qt = content_tokens(low)
    if len(qt) < MIN_QUESTION_CONTENT_TOKENS:
        return False, "low-information"
    first = low.split()[0]
    if first in ANAPHORA_STARTS and len(qt) < 8:
        return False, "ambiguous-anaphora"
    if any(p.search(low) for p in MEDICAL_JUDGMENT_PATTERNS):
        return False, "medical-judgment"
    # Answer leakage / restatement: the question must not be an ordered
    # substring of the supporting span (verbatim) nor a near-mirror of it.
    nq = normalize_for_span(low)
    ns = normalize_for_span(span)
    nq_tokens = nq.split()
    if len(nq_tokens) >= ANSWER_LEAK_MIN_TOKENS and ns:
        if nq in ns or ns.startswith(nq):
            return False, "restates-span"
    st = content_tokens(ns)
    ct = content_tokens(chunk_text)
    if qt and st and (len(qt & st) / len(qt)) >= 0.85 and len(st - qt) <= 3:
        return False, "mirrors-span"
    if not (qt & st) and not (qt & ct):
        return False, "not-anchored"
    return True, None
QUALITY_SCREEN_PROMPT = """You are a strict reviewer of gold-evaluation questions
for a health knowledge base used by an elderly-care assistant. The knowledge
base contains public NIA/NIH/WHO caregiver and healthy-aging documents.

You receive:
- QUESTION: a user question proposed for the gold set.
- EVIDENCE SENTENCE: the exact sentence from the knowledge base that is
  supposed to support the answer.

Decide whether the question is acceptable. It MUST be ALL of:
- a natural, well-formed, self-contained user question (no unresolved
  pronouns such as "it", "that", "the patient", "her", "them" that need
  outside context)
- answerable using ONLY the EVIDENCE SENTENCE, without needing outside
  knowledge or an imaginary patient record
- NOT a yes/no question
- NOT asking the assistant to diagnose, prescribe, predict risk or outcome,
  judge medical stability, or make a medical/treatment decision
- NOT restating or echoing the EVIDENCE SENTENCE verbatim or near-verbatim
  (it must be a genuine question a caregiver or older adult would ask)
- NOT answerable by the question itself (no answer leakage)
- appropriate for elderly care (nutrition, exercise, memory, medication
  safety, caregiving, ICOPE care pathways, healthy aging)

Return ONLY JSON: {{"acceptable": true or false, "reason": "one short phrase"}}

QUESTION:
{question}

EVIDENCE SENTENCE:
{span}
"""


def quality_screen_llm(q, span, retry=1):
    """[QUALITY SCREEN - LLM ROLE] Independent reviewer (separate prompt from
    generation; the generated question is not self-validated)."""
    last_err = None
    for _ in range(retry + 1):
        try:
            prompt = QUALITY_SCREEN_PROMPT.format(question=q, span=span)
            resp = ollama.chat(
                model=QUALITY_SCREEN_MODEL,
                options=dict(QUALITY_SCREEN_SAMPLING),
                messages=[
                    {"role": "system",
                     "content": "You produce JSON only."},
                    {"role": "user", "content": prompt},
                ],
            )
            text = resp["message"]["content"].strip()
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not m:
                last_err = "no-json"
                continue
            parsed = json.loads(m.group(0))
            if not isinstance(parsed.get("acceptable"), bool):
                last_err = "malformed-bool"
                continue
            reason = str(parsed.get("reason", ""))[:120]
            return parsed["acceptable"], (None if parsed["acceptable"] else reason)
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(1)
    return False, f"screen-error:{last_err}"


def verify_anchor(question, source_document, chunk_id, span, top_k=None):
    """[EVIDENCE VERIFICATION] Anchor-level acceptance gate using the SAME
    production hybrid retriever and the SAME metric definitions as the current
    evaluation pipeline.

    Returns a dict with:
        source_correct        (expected doc in top-k)
        chunk_recall          (expected anchor chunk id in top-k)
        supporting_span_supported (span exact or token-coverage >= 0.85)
    A question is answerable ONLY when all three flags are True.
    """
    top_k = top_k or RETRIEVAL_TOP_K
    try:
        hits = hybrid_search(question)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}",
                "retrieved_sources": [], "retrieved_chunk_ids": []}
    top = hits[:top_k]
    norm_expected = normalize_doc(source_document)
    norm_sources = {normalize_doc(h.get("source_document")) for h in top}
    retrieved_chunk_ids = {int(h.get("chunk_id"))
                           for h in top if h.get("chunk_id") is not None}
    source_correct = bool(norm_sources) and norm_expected in norm_sources
    chunk_recall = int(chunk_id) in retrieved_chunk_ids
    joined = " ".join(h.get("text") or "" for h in top)
    span_exact = bool(span) and normalize_for_span(span) in normalize_for_span(joined)
    span_cov = span_token_coverage(span, joined)
    span_supported = bool(span) and (span_exact or span_cov >= SPAN_MIN_TOKEN_COVERAGE)
    return {
        "source_correct": source_correct,
        "chunk_recall": chunk_recall,
        "supporting_span_supported": span_supported,
        "supporting_span_exact": span_exact,
        "supporting_span_coverage": span_cov,
        "retrieved_sources": sorted(h.get("source_document") or "" for h in top),
        "retrieved_chunk_ids": sorted(retrieved_chunk_ids),
        "top_k": top_k,
    }


def locate_span(chunk_text, span):
    """Verify the supporting span literally exists in the anchor chunk text.
    Returns (char_start, char_end, found)."""
    if not chunk_text or not span:
        return None, None, False
    raw = chunk_text.find(span)
    if raw != -1:
        return raw, raw + len(span), True
    nchunk = normalize_for_span(chunk_text)
    nspan = normalize_for_span(span)
    pos = nchunk.find(nspan)
    if pos != -1:
        return pos, pos + len(nspan), True
    return None, None, False
_TOPIC_MAP = {
    "understanding-memory-loss.pdf": "memory-loss",
    "exercise-and-older-adults-nia.pdf": "exercise",
    "tips-take-medicines-safely.pdf": "medications",
    "nia_caregivers_handbook.pdf": "caregiving",
    "Dietary_Guidelines_for_Americans_2020-2025.pdf": "nutrition",
    "who_icope_handbook.pdf": "who-health",
}

_CATEGORY_MAP = {
    "understanding-memory-loss.pdf": "memory-loss",
    "exercise-and-older-adults-nia.pdf": "exercise",
    "tips-take-medicines-safely.pdf": "medication_information",
    "nia_caregivers_handbook.pdf": "caregiver_manuals",
    "Dietary_Guidelines_for_Americans_2020-2025.pdf": "healthy_aging",
    "who_icope_handbook.pdf": "hospital_faqs",
}


def _topic_for(doc):
    return _TOPIC_MAP.get(doc, "general")


def _category_for(doc):
    return _CATEGORY_MAP.get(doc, "general")


def allocate_targets(target, doc_counts, cap=ALLOC_CAP, floor=ALLOC_FLOOR,
                     power=ALLOC_WEIGHT_POWER, rng=None):
    """Capped, sqrt-proportional per-document allocation with a floor.

    - Every document with candidates gets at least `floor` slots (no domain
      starves just because its PDF is short).
    - The remaining budget is split proportionally to pool_size ** power
      (power=0.5 -> sqrt), clamped so NO document exceeds `cap`.
    - Leftovers (rounding / caps) go to the largest-weight documents that
      still have room, so the total is exactly `target` when feasible.

    Returns dict {doc: allocated}.
    """
    docs = sorted(d for d, c in doc_counts.items() if c > 0)
    if not docs:
        return {}
    # Adaptive floor: for small targets (validation runs) the fixed 15-floor
    # would exceed the target and round every doc down to 0. Shrink the floor
    # so every doc gets >= 1 slot while the 120-question target path is
    # unaffected (the full floor of 15 applies whenever the target allows).
    n_docs = len(docs)
    if target < n_docs * floor:
        floor = max(1, target // n_docs)
    base = {d: min(floor, doc_counts[d]) for d in docs}
    alloc = dict(base)
    remaining = target - sum(alloc.values())
    if remaining < 0:
        # Target smaller than the floors: give the first `target` docs
        # (by weight) exactly 1 slot each, never 0. This preserves the
        # guarantee that every doc is reachable for small validation runs.
        weights = {d: doc_counts[d] ** power for d in docs}
        order = sorted(docs, key=lambda d: (-weights[d], d))
        alloc = {d: 0 for d in docs}
        for d in order[:target]:
            alloc[d] = 1
        return alloc
    weights = {d: doc_counts[d] ** power for d in docs}
    wsum = sum(weights.values())
    for d in docs:
        share = int(round(remaining * weights[d] / wsum)) if wsum else 0
        alloc[d] = min(cap, doc_counts[d], alloc[d] + share)
    leftover = target - sum(alloc.values())
    order = sorted(docs, key=lambda d: (-weights[d], d))
    while leftover > 0:
        cands = [d for d in order if alloc[d] < min(cap, doc_counts[d])]
        if not cands:
            break
        d = cands[0]
        alloc[d] += 1
        leftover -= 1
    return alloc
def build(target=TARGET_QUESTIONS, max_calls=None, max_drawn=None):
    """Build the anchor-verified extended gold QA set.

    Pipeline per candidate:
      span length -> span-in-chunk -> gold16-protect dedup -> fact dedup
      -> chunk-cap -> [LLM generate] -> deterministic screen -> LLM screen
      -> anchor verification (source + chunk + span) -> question dedup
    Generation, screening and verification are strictly separate functions.
    """
    console = []
    log = console.append
    t0 = time.perf_counter()
    truncated = False

    # --- Load data (read-only) ---
    chunks = load_chunks()
    pool = build_candidate_pool(chunks)
    gold16_doc, gold16_spans, gold16_qtoks = load_gold16()
    chunk_by_key = {(c.get("source_document"), c.get("chunk_id")): c for c in chunks}
    log(f"chunks={len(chunks)} candidates={len(pool)} gold16_questions="
        f"{len(gold16_doc.get('gold_questions', []))}")

    sha = hashlib.sha256()
    with open(CHUNK_FILE, "rb") as f:
        sha.update(f.read())
    chunks_sha = sha.hexdigest()

    # --- Allocation (capped sqrt-proportional, exactly target) ---
    doc_counts = {}
    for item in pool:
        doc_counts[item["source_document"]] = doc_counts.get(item["source_document"], 0) + 1
    allocated = allocate_targets(target, doc_counts)
    log("target allocation: " + json.dumps(allocated, sort_keys=True))
    if sum(allocated.values()) != target:
        log(f"WARNING: allocation sum {sum(allocated.values())} != target {target}")

    # --- Seeded per-document candidate order ---
    rng = random.Random(RNG_SEED)
    by_doc = defaultdict(list)
    for item in pool:
        by_doc[item["source_document"]].append(item)
    cand_by_doc = {}
    for d, items in by_doc.items():
        order = list(items)
        rng.shuffle(order)
        cand_by_doc[d] = iter(order)

    # --- Rejection accounting ---
    REJ = defaultdict(int)
    REJ_D = defaultdict(int)   # deterministic screen reasons
    REJ_L = defaultdict(int)   # LLM screen reasons
    samples = defaultdict(list)

    def _sample(cat, obj, cap=5):
        if len(samples[cat]) < cap:
            samples[cat].append(obj)

    active = []
    accepted_spans = []
    accepted_qtoks = []
    chunk_used = defaultdict(int)
    achieved_by_doc = defaultdict(int)
    calls = 0
    doc_order = sorted(allocated, key=lambda d: (-allocated[d], d))

    # Hard safety bound on total candidate draws across ALL documents.
    # Combined with the per-document `while produced < budget` guard, this
    # guarantees NO execution path can loop without bound even when the LLM
    # call budget is exhausted (previously the loop kept drawing candidates,
    # calling formulate_question -> calls += 1, and continue-ing forever).
    if max_drawn is None:
        max_drawn = max(0, int(round(target * 100)))
    total_drawn = 0
    term_reason = "target_reached"

    def emit(msg):
        print(msg, flush=True)
        log(msg)

    for d in doc_order:
        budget = allocated[d]
        it = cand_by_doc.get(d, iter(()))
        produced = 0
        drawn = 0
        emit(f"== doc {d} budget={budget}")
        while produced < budget:
            # LLM-budget guard: once max_calls is exhausted, stop drawing any
            # further candidates for this and all remaining documents.
            if max_calls is not None and calls >= max_calls:
                term_reason = "llm_budget_exhausted"
                truncated = True
                break
            # Hard draw-limit guard: never draw more than max_drawn candidates
            # across the whole build, regardless of budget settings.
            if total_drawn >= max_drawn:
                term_reason = "draw_limit"
                truncated = True
                break
            item = next(it, None)
            if item is None:
                break
            drawn += 1
            total_drawn += 1
            span = item["sentence"]
            cid = item["chunk_id"]
            chunk = chunk_by_key.get((d, cid)) or {}
            chunk_text = chunk.get("text", "")

            # (0) answer length cap
            if len(span) > MAX_ANSWER_SPAN_LEN:
                REJ["by_span_length"] += 1
                _sample("by_span_length", {"doc": d, "sentence": span[:100]})
                continue

            # (1) evidence integrity: span must literally exist in anchor chunk
            start, end, found = locate_span(chunk_text, span)
            if not found:
                REJ["span_not_in_chunk"] += 1
                _sample("span_not_in_chunk",
                        {"doc": d, "chunk_id": cid, "sentence": span[:100]})
                continue

            span_toks = content_tokens(span)

            # (2) protect the original 16-question benchmark
            if max((token_jaccard(span_toks, g) for g in gold16_spans),
                   default=0.0) >= GOLD16_PROTECT_JACCARD:
                REJ["by_gold16_duplicate"] += 1
                _sample("by_gold16_duplicate", {"doc": d, "sentence": span[:100]})
                continue

            # (3) fact-level dedup vs already-accepted facts
            if max((token_jaccard(span_toks, s) for s in accepted_spans),
                   default=0.0) >= SPAN_DEDUP_JACCARD:
                REJ["by_fact_duplicate"] += 1
                _sample("by_fact_duplicate", {"doc": d, "sentence": span[:100]})
                continue

            # (4) per-anchor-chunk cap (adjacent sentences cannot flood a chunk);
            #     keyed by (source_document, chunk_id) because chunk IDs
            #     restart for each source document.
            if chunk_used[(d, cid)] >= MAX_PER_CHUNK:
                REJ["by_chunk_cap"] += 1
                continue
            # (5) [GENERATION] formulate a question from the span
            if drawn % 5 == 0:
                emit(f"   [{d}] drawn={drawn} produced={produced} "
                     f"span={span[:70]!r}")
            q, err = formulate_question(span)
            calls += 1
            if q is None:
                REJ["llm_generation_failure"] += 1
                _sample("llm_generation_failure",
                        {"doc": d, "sentence": span[:100], "err": err})
                continue

            # (6) [QUALITY - deterministic]
            ok, reason = quality_screen_deterministic(q, span, chunk_text)
            if not ok:
                REJ["by_quality_deterministic"] += 1
                REJ_D[reason] += 1
                _sample("by_quality_deterministic",
                        {"doc": d, "question": q[:120], "reason": reason})
                continue

            # (7) [QUALITY - LLM, independent role]
            emit(f"   [{d}] gen_ok q={q[:70]!r} -> llm_screen")
            ok, reason = quality_screen_llm(q, span)
            calls += 1
            if not ok:
                REJ["by_quality_llm"] += 1
                REJ_L[reason or "unspecified"] += 1
                _sample("by_quality_llm",
                        {"doc": d, "question": q[:120], "reason": reason})
                continue

            # (8) [EVIDENCE VERIFICATION] anchor-level, production retriever
            emit(f"   [{d}] screen_ok -> verify_anchor")
            ver = verify_anchor(q, d, cid, span)
            if ver.get("error"):
                REJ["retrieval_error"] += 1
                _sample("retrieval_error",
                        {"doc": d, "question": q[:120], "err": ver["error"]})
                continue
            if not (ver["source_correct"] and ver["chunk_recall"]
                    and ver["supporting_span_supported"]):
                if not ver["source_correct"]:
                    cat = "by_source"
                elif not ver["chunk_recall"]:
                    cat = "by_anchor_chunk"
                else:
                    cat = "by_span_verification"
                REJ[cat] += 1
                _sample(cat, {
                    "doc": d, "question": q[:120],
                    "retrieved_sources": ver["retrieved_sources"],
                    "retrieved_chunk_ids": ver["retrieved_chunk_ids"]})
                continue
            # (9) question-level dedup vs accepted + protected gold16
            qtoks = content_tokens(q)
            if max((token_jaccard(qtoks, t) for t in accepted_qtoks),
                   default=0.0) >= QUESTION_DEDUP_JACCARD or \
               max((token_jaccard(qtoks, t) for t in gold16_qtoks),
                   default=0.0) >= QUESTION_DEDUP_JACCARD:
                REJ["by_question_duplicate"] += 1
                _sample("by_question_duplicate", {"doc": d, "question": q[:120]})
                continue

            # ---- ACCEPT ----
            record = {
                "id": f"GQX-{len(active) + 1:03d}",
                "question": q,
                "reference_answer": span,
                "topic": _topic_for(d),
                "category": _category_for(d),
                "source_document": d,
                "chunk_ids": [cid],
                "target_keys": [{"source_document": d, "chunk_id": cid}],
                "supporting_span": span,
                "span_char_start": start,
                "span_char_end": end,
                "span_in_anchor_chunk": True,
                "verification": ver,
            }
            active.append(record)
            accepted_spans.append(span_toks)
            accepted_qtoks.append(qtoks)
            chunk_used[(d, cid)] += 1
            achieved_by_doc[d] += 1
            produced += 1
            emit(f"  {record['id']} [{d} c{cid}] {q[:70]}")

            if max_calls is not None and calls >= max_calls:
                term_reason = "llm_budget_exhausted"
                truncated = True
                break
        if truncated:
            break

    elapsed = round(time.perf_counter() - t0, 2)
    status = "ready_for_review" if len(active) >= target else "validation_partial"
    if term_reason == "target_reached" and len(active) < target:
        term_reason = "target_not_reached"
    log(f"accepted={len(active)} calls={calls} drawn={total_drawn} "
        f"term_reason={term_reason} status={status} "
        f"elapsed_s={elapsed}")
    # ---------------- dataset assembly ----------------
    by_doc_ach = {d: achieved_by_doc[d] for d in sorted(achieved_by_doc)}
    n_pass_src = sum(1 for r in active
                     if r["verification"].get("source_correct"))
    n_pass_chunk = sum(1 for r in active
                       if r["verification"].get("chunk_recall"))
    n_pass_span = sum(1 for r in active
                      if r["verification"].get("supporting_span_supported"))
    n = max(len(active), 1)

    dataset = {
        "schema_version": "1.0",
        "evaluation": "gold_qa_elderdocai_knowledge_base_extended_v2",
        "description": (
            "Anchor-verified extended gold QA set from the ElderDocAI "
            "knowledge base's six public NIA/NIH/WHO documents. Questions are "
            "LLM-formulated; reference answers == verbatim supporting spans; "
            "every accepted question passes source+anchor-chunk+span "
            "verification through the production hybrid retriever (top-3)."
        ),
        "metadata": {
            "title": "ElderDocAI Extended Gold QA Set (anchor-verified)",
            "status": status,
            "dataset_final": status == "ready_for_review",
            "seed": RNG_SEED,
            "target_question_count": target,
            "accepted_question_count": len(active),
            "per_domain_allocation_target": dict(allocated),
            "per_domain_allocation_achieved": by_doc_ach,
            "generation": {
                "model": QUESTION_MODEL,
                "params": dict(QUESTION_SAMPLING),
            },
            "quality_screen": {
                "model": QUALITY_SCREEN_MODEL,
                "params": dict(QUALITY_SCREEN_SAMPLING),
                "deterministic_checks": [
                    "meaningful length >= %d" % MIN_QUESTION_LEN,
                    "ends with '?'",
                    "no malformed markers",
                    "not yes/no",
                    ">= %d content tokens" % MIN_QUESTION_CONTENT_TOKENS,
                    "no ambiguous leading pronoun",
                    "no diagnosis/treatment/prediction requests",
                    "no verbatim restatement of span",
                    "no near-mirror of span",
                    "shares content with span or anchor chunk",
                ],
                "llm_checks": [
                    "well-formed self-contained question",
                    "answerable from span alone",
                    "in elderly-care scope",
                    "no medical judgment requests",
                    "no verbatim echo/restatement",
                ],
                "roles_separated": (
                    "generation, quality screening and evidence verification "
                    "are separate functions with separate prompts"
                ),
            },
            "retrieval": {
                "engine": "production scripts.hybrid_retriever.hybrid_search",
                "top_k": RETRIEVAL_TOP_K,
                "acceptance": (
                    "source_correct AND chunk_recall AND "
                    "supporting_span_supported"
                ),
                "span_coverage_threshold": SPAN_MIN_TOKEN_COVERAGE,
            },
            "dedup": {
                "gold16_protect_jaccard": GOLD16_PROTECT_JACCARD,
                "span_dedup_jaccard": SPAN_DEDUP_JACCARD,
                "question_dedup_jaccard": QUESTION_DEDUP_JACCARD,
                "max_per_chunk": MAX_PER_CHUNK,
            },
            "distribution": {
                "floors": ALLOC_FLOOR,
                "caps": ALLOC_CAP,
                "weight_power": ALLOC_WEIGHT_POWER,
            },
            "source_dataset": {
                "chunk_file": "data/chunks/knowledge_base_chunks.json",
                "n_chunks": len(chunks),
                "candidate_pool_size": len(pool),
                "sha256": chunks_sha,
            },
            "creation_timestamp_utc": (
                datetime.datetime.now(datetime.timezone.utc).isoformat()
            ),
            "creator": "scripts/build_gold_extended.py",
        },
        "n_gold_questions": len(active),
        "gold_questions": active,
        "out_of_scope_questions": gold16_doc.get("out_of_scope_questions", []),
    }
    report = {
        "status": status,
        "dataset_final": status == "ready_for_review",
        "config": {
            "seed": RNG_SEED,
            "target": target,
            "generation_model": QUESTION_MODEL,
            "generation_params": dict(QUESTION_SAMPLING),
            "quality_screen_model": QUALITY_SCREEN_MODEL,
            "quality_screen_params": dict(QUALITY_SCREEN_SAMPLING),
            "retrieval_top_k": RETRIEVAL_TOP_K,
            "span_coverage_threshold": SPAN_MIN_TOKEN_COVERAGE,
            "dedup": {
                "gold16_protect_jaccard": GOLD16_PROTECT_JACCARD,
                "span_dedup_jaccard": SPAN_DEDUP_JACCARD,
                "question_dedup_jaccard": QUESTION_DEDUP_JACCARD,
                "max_per_chunk": MAX_PER_CHUNK,
            },
            "distribution": {
                "floors": ALLOC_FLOOR, "caps": ALLOC_CAP,
                "weight_power": ALLOC_WEIGHT_POWER,
            },
            "max_calls": max_calls,
            "max_drawn": max_drawn,
            "truncated_by_max_calls": bool(truncated),
            "termination_reason": term_reason,
        },
        "candidate_pool_size": len(pool),
        "candidate_pool_by_document": dict(doc_counts),
        "target_allocation": dict(allocated),
        "final_accepted_count": len(active),
        "final_distribution_by_document": by_doc_ach,
        "rejected": {
            "by_source": REJ["by_source"],
            "by_anchor_chunk": REJ["by_anchor_chunk"],
            "by_span_verification": REJ["by_span_verification"],
            "by_span_length": REJ["by_span_length"],
            "span_not_in_chunk": REJ["span_not_in_chunk"],
            "by_quality_deterministic": dict(REJ_D),
            "by_quality_llm": dict(REJ_L),
            "by_gold16_duplicate": REJ["by_gold16_duplicate"],
            "by_fact_duplicate": REJ["by_fact_duplicate"],
            "by_question_duplicate": REJ["by_question_duplicate"],
            "by_chunk_cap": REJ["by_chunk_cap"],
            "llm_generation_failure": REJ["llm_generation_failure"],
            "retrieval_error": REJ["retrieval_error"],
        },
        "verification_results": {
            "pct_source_correct": round(100.0 * n_pass_src / n, 2),
            "pct_chunk_recall": round(100.0 * n_pass_chunk / n, 2),
            "pct_supporting_span_supported": round(100.0 * n_pass_span / n, 2),
            "note": "acceptance is mandatory, so these should be 100%",
        },
        "llm_calls_used": calls,
        "candidates_drawn": total_drawn,
        "termination_reason": term_reason,
        "discarded_examples": {k: v for k, v in samples.items()},
        "elapsed_seconds": elapsed,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    with open(CURATION_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(CONSOLE_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(console))

    print(f"\nWROTE {OUT_JSON} ({len(active)} questions, status={status})")
    print(f"WROTE {CURATION_JSON}")
    print(f"WROTE {CONSOLE_TXT}")
    print("target allocation:", json.dumps(allocated, sort_keys=True))
    print("achieved       :", json.dumps(by_doc_ach, sort_keys=True))
    print("rejections     :", json.dumps(dict(REJ), sort_keys=True))
    print(f"source/chunk/span pass = {n_pass_src}/{n_pass_chunk}/{n_pass_span} "
          f"of {len(active)}")
    return active, report


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build the anchor-verified extended gold QA set.")
    ap.add_argument("--target", type=int, default=TARGET_QUESTIONS,
                    help="number of questions to generate (default 120)")
    ap.add_argument("--max-calls", type=int, default=0,
                    help="stop after this many LLM calls (0=unlimited); "
                         "use for quick validation runs")
    ap.add_argument("--max-drawn", type=int, default=0,
                    help="hard cap on candidate draws across all docs "
                         "(0 = target*100 default); use for quick validation")
    args = ap.parse_args(argv)
    build(args.target,
          max_calls=(args.max_calls or None),
          max_drawn=(args.max_drawn or None))


if __name__ == "__main__":
    # A spawned child (any torch/HF re-exec) must NOT re-run the job.
    if _mp.parent_process() is not None:
        import sys as _sys
        _sys.exit(0)
    main()