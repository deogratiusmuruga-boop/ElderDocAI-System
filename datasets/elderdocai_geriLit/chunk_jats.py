#!/usr/bin/env python3
"""ElderDocAI-GeriLit dev-v0.1 - Phase 3 Task 4: JATS-aware, section-aware chunking.

Consumes the frozen Task 2 accepted manifest + Task 3 metadata artifacts and the
raw JATS XML to produce deterministic, provenance-complete RAG-ready chunks.

DATA REPRESENTATION TASK ONLY: no embeddings, no indexes, no retrieval.
Parameters (TARGET/MAX/MIN/OVERLAP) are PROVISIONAL-DEV (see
metadata/chunking_specification.md).
"""
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from html import unescape

BASE = Path(r"C:\Users\chosun\Documents\ElderDocAI-System\datasets\elderdocai_geriLit")
MAN = BASE / "manifest"
META = BASE / "metadata"
RAW = BASE / "raw"
CHUNK_DIR = BASE / "chunks"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR = META / "validation"

VERSION = "ElderDocAI-GeriLit-dev-v0.1"

# ---- PROVISIONAL-DEV chunking parameters (see chunking_specification.md) ----
TARGET_WORDS = 220
MAX_WORDS = 300
MIN_CHUNK_WORDS = 50
MIN_MERGE_WORDS = 100
OVERLAP_SENTENCES = 1

# ---- JATS region / element classification ----
INCLUDE_BACK_TITLE = re.compile(
    r"(data availability|ethic|declaration|limitation|methods|appendix|"
    r"supplement|consent|data sharing|statements|study protocol|"
    r"instrument|questionnaire|coding|definitions)", re.I)
EXCLUDE_TITLE = re.compile(
    r"(acknowledg|funding|financial support|sources of support|grant support|"
    r"conflict of interest|competing interest|author contribution|"
    r"correspondence|publisher|references\b|abbreviations\b|vocabulary)",
    re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])(?=\s+[\"'([]*[A-Z])")


def local(tag):
    return tag.split("}")[-1]


def clean(s):
    if s is None:
        return None
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def strip_doctype(xml):
    return re.sub(r"<!DOCTYPE[^>]*>", "", xml, count=1, flags=re.S)


def fix_entities(xml):
    def repl(m):
        name = m.group(1)
        if name in {"amp", "lt", "gt", "quot", "apos"}:
            return m.group(0)
        u = unescape("&%s;" % name)
        if u != "&%s;" % name:
            return u
        return ""
    return re.sub(r"&([a-zA-Z][a-zA-Z0-9]+);", repl, xml)


def load_xml(path):
    text = fix_entities(strip_doctype(path.read_text(encoding="utf-8",
                                                     errors="replace")))
    try:
        return ET.fromstring(text), []
    except ET.ParseError as e:
        return None, ["xml_parse_error: %s" % e]


def find_one(root, name):
    for n in root.iter():
        if local(n.tag) == name:
            return n
    return None


def text_of(el):
    if el is None:
        return ""
    return clean(" ".join(el.itertext())) or ""


def wc(text):
    return len(text.split())


def split_sentences(text):
    parts = SENT_SPLIT.split(text)
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p.strip()]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def render_caption_text(cap_el):
    parts = []
    for node in cap_el.iter():
        t = local(node.tag)
        if t in ("title", "p"):
            tx = text_of(node)
            if tx:
                parts.append(tx)
    return " ".join(parts)


def render_table(tw):
    """Flatten a JATS <table-wrap> to a deterministic textual representation."""
    cap = None
    for node in tw.iter():
        if local(node.tag) == "caption":
            cap = render_caption_text(node)
            break
    rows = []
    for tr in tw.iter():
        if local(tr.tag) != "tr":
            continue
        cells = []
        for cell in tr:
            t = local(cell.tag)
            if t in ("th", "td"):
                cells.append(text_of(cell))
        if cells:
            rows.append(" | ".join(x for x in cells if x))
    out = []
    if cap:
        out.append("Caption: " + cap)
    if rows:
        out.append("Rows: " + " / ".join(rows))
    return clean(" ".join(out)) if out else ""


def render_fig(fig):
    parts = []
    label = None
    for node in fig.iter():
        t = local(node.tag)
        if t == "label":
            label = text_of(node)
        elif t == "caption":
            cap = render_caption_text(node)
            if cap:
                parts.append(cap)
        elif t == "title":
            tx = text_of(node)
            if tx:
                parts.append(tx)
    if label and label not in " ".join(parts):
        parts.insert(0, "Fig. " + label + ":")
    return clean(" ".join(parts)) if parts else ""


def render_list(lst):
    items = []
    for li in lst.iter():
        if local(li.tag) == "list-item":
            tx = text_of(li)
            if tx:
                items.append(tx)
    return clean(" ".join(items)) if items else ""


def render_def_list(dl):
    items = []
    for di in dl.iter():
        if local(di.tag) == "def-item":
            term = text_of(next((c for c in di if local(c.tag) == "term"), None))
            for c in di:
                if local(c.tag) == "def":
                    items.append("%s: %s" % (term, text_of(c)) if term else text_of(c))
    return clean(" ".join(items)) if items else ""


SPECIAL_TYPES = {
    "table-wrap": ("TABLE", render_table),
    "fig": ("FIGURE_CAPTION", render_fig),
    "boxed-text": ("BOXED_TEXT", lambda el: text_of(el)),
    "disp-quote": ("DISP_QUOTE", lambda el: text_of(el)),
    "disp-formula": ("FORMULA", lambda el: text_of(el)),
    "list": ("LIST", render_list),
    "def-list": ("LIST", render_def_list),
}


def synth_sec_record(sec, sid, chain, region, order):
    label = None
    title = None
    for c in sec:
        t = local(c.tag)
        if t == "label" and label is None:
            label = text_of(c)
        elif t == "title" and title is None:
            title = text_of(c)
    return {
        "section_id": sid,
        "label": label or "",
        "title": title or "",
        "depth": len(chain) + 1,
        "order": order,
        "parent_section_id": chain[-1].get("section_id")
        if chain else None,
        "region": region,
        "section_type": sec.get("sec-type"),
        "_synthesized": True,
    }


def include_sec(region, inv):
    title = inv.get("title") or ""
    if EXCLUDE_TITLE.search(title):
        return False
    if region == "BODY":
        return True
    if region == "BACK":
        return bool(INCLUDE_BACK_TITLE.search(title)) or title == ""
    return True


def walk_region(el, region, chain, inv_index, out, audit, order_counter):
    """Walk a JATS container, appending element dicts to `out` in doc order."""
    for child in el:
        t = local(child.tag)
        if t == "sec":
            sid = child.get("id")
            inv = inv_index.get(sid)
            if inv is None:
                order_counter[0] += 1
                inv = synth_sec_record(child, sid, chain, region,
                                       order_counter[0])
            if not include_sec(region, inv):
                audit["excluded_sections"].append(sid)
                continue
            walk_region(child, region, chain + [inv], inv_index, out, audit,
                        order_counter)
        elif t in ("title", "label", "sec-meta"):
            continue
        elif t == "p":
            txt = text_of(child)
            if not txt:
                audit["empty_source_elements"] += 1
                continue
            out.append({"type": "P", "text": txt, "region": region,
                        "sec": chain[-1] if chain else None})
        elif t in SPECIAL_TYPES:
            ct, renderer = SPECIAL_TYPES[t]
            txt = renderer(child)
            if not txt:
                audit["special_empty_skipped"] += 1
                continue
            out.append({"type": ct, "text": txt, "region": region,
                        "sec": chain[-1] if chain else None})
        # any other element types are ignored at this level (references are
        # handled separately; authors/affiliations excluded by construction)


def group_sentences(sents, max_words, overlap):
    """Greedy sentence grouping with 1-sentence overlap. Returns list of parts."""
    if not sents:
        return []
    if len(sents) == 1:
        return [sents[0]]
    parts = []
    cur = []
    for s in sents:
        if cur and wc(" ".join(cur + [s])) > max_words:
            parts.append(" ".join(cur))
            if overlap and len(cur) >= 1:
                cur = [cur[-1], s]
            else:
                cur = [s]
        else:
            cur.append(s)
    if cur:
        parts.append(" ".join(cur))
    return parts


def scope_key(e):
    return (e.get("region"), (e.get("sec") or {}).get("section_id"))


def resolve_srecord(region, sid, sec_inv):
    """Return a section metadata record for a scope (region, section_id)."""
    if region == "ABSTRACT":
        return {"section_id": "abstract", "label": "", "title": "Abstract",
                "depth": 0, "order": 0, "parent_section_id": None,
                "region": "ABSTRACT", "section_type": None}
    if sid is None:
        return {"section_id": None, "label": "", "title": "", "depth": 0,
                "order": 0, "parent_section_id": None, "region": region,
                "section_type": None}
    inv = sec_inv.get(sid) or {}
    return {"section_id": sid,
            "label": inv.get("label") or "",
            "title": inv.get("title") or "",
            "depth": inv.get("depth", 1),
            "order": int(inv.get("order") or 0),
            "parent_section_id": inv.get("parent_section_id"),
            "region": inv.get("region") or region,
            "section_type": inv.get("section_type")}


def chunk_flags(text, ct, el_refs, split_flag, unsplit_flag, pg_count):
    return {
        "is_empty": False,
        "is_too_short": ct in ("PROSE", "ABSTRACT") and
                        wc(text) < MIN_CHUNK_WORDS,
        "is_too_long": wc(text) > MAX_WORDS,
        "paragraph_split": split_flag,
        "sentence_split": split_flag,
        "unsplit_long_sentence": unsplit_flag,
        "contains_table": ct == "TABLE",
        "contains_figure_caption": ct == "FIGURE_CAPTION",
        "contains_special_content": ct in ("TABLE", "FIGURE_CAPTION",
                                           "BOXED_TEXT", "DISP_QUOTE",
                                           "FORMULA", "LIST"),
        "provenance_complete": True,
        "paragraph_count": pg_count,
    }


class ChunkBuilder:
    def __init__(self, pmc, article_meta, ev_meta, sec_inv):
        self.pmc = pmc
        self.article_meta = article_meta
        self.ev = ev_meta
        self.sec_inv = sec_inv
        self.chunks = []
        self.ci = 0
        self.sec_ci = {}
        self.buffer = []
        self.buffer_scope = None
        self.overlap_parts = 0

    def flush_buffer(self):
        buf, scope = self.buffer, self.buffer_scope
        if buf and scope is not None:
            text = " ".join(e["text"] for e in buf)
            self.emit(scope, "PROSE", text, [e["el_index"] for e in buf],
                      0, False, False)
        self.buffer = []
        self.buffer_scope = None

    def buffer_words(self):
        return sum(wc(x["text"]) for x in self.buffer)

    def add_paragraph(self, e):
        scope = scope_key(e)
        if self.buffer and scope != self.buffer_scope:
            self.flush_buffer()
        if not self.buffer:
            self.buffer_scope = scope
        tw = wc(e["text"])
        if self.buffer and self.buffer_words() + tw > TARGET_WORDS:
            self.flush_buffer()
            self.buffer_scope = scope
        self.buffer.append(e)

    def add_special(self, e, ct):
        self.flush_buffer()
        self.emit(scope_key(e), ct, e["text"], [e["el_index"]], 1,
                  False, False)

    def add_split_paragraph(self, e):
        self.flush_buffer()
        sents = split_sentences(e["text"])
        if len(sents) <= 1:
            self.emit(scope_key(e), "PROSE", e["text"], [e["el_index"]],
                      1, False, True)
            return
        parts = group_sentences(sents, MAX_WORDS, OVERLAP_SENTENCES)
        for i, part in enumerate(parts):
            self.emit(scope_key(e), "PROSE", part, [e["el_index"]],
                      1, True, False)
            if i > 0:
                self.overlap_parts += 1

    def emit(self, scope, ct, text, el_refs, pg_count, split_flag,
             unsplit_flag):
        region, sid = scope
        srec = resolve_srecord(region, sid, self.sec_inv)
        key = (region, sid)
        sci = self.sec_ci.get(key, 0)
        self.sec_ci[key] = sci + 1
        chunk_id = "%s__%s__%03d__%s__%05d" % (
            self.pmc, region, int(srec["order"]), ct, self.ci)
        loc = "%s:%s:%s:%s" % (self.pmc, region, sid or "-",
                               "%d..%d" % (el_refs[0], el_refs[-1])
                               if el_refs else "-")
        chunk = {
            "chunk_id": chunk_id,
            "pmcid": self.pmc,
            "pmid": self.article_meta.get("pmid"),
            "doi": self.article_meta.get("doi"),
            "title": self.article_meta.get("title"),
            "journal": self.article_meta.get("journal"),
            "publication_year": self.article_meta.get("publication_year"),
            "language": self.article_meta.get("language"),
            "article_type": self.article_meta.get("article_type"),
            "license_category": self.article_meta.get("license_category"),
            "topic_ids": self.article_meta.get("topic_ids") or [],
            "corpus_version": VERSION,
            "evidence_type": self.ev.get("evidence_type"),
            "evidence_type_confidence": self.ev.get(
                "evidence_type_confidence"),
            "region": region,
            "section_id": srec["section_id"],
            "section_label": srec["label"],
            "section_title": srec["title"],
            "parent_section_id": srec["parent_section_id"],
            "section_depth": srec["depth"],
            "section_order": srec["order"],
            "content_type": ct,
            "chunk_index": self.ci,
            "section_chunk_index": sci,
            "text": text,
            "word_count": wc(text),
            "character_count": len(text),
            "source_locator": loc,
            "element_indexes": el_refs,
            "flags": chunk_flags(text, ct, el_refs, split_flag,
                                 unsplit_flag, pg_count),
        }
        self.chunks.append(chunk)
        self.ci += 1


def collect_abstract(root, audit):
    elements = []
    ab = find_one(root, "abstract")
    if ab is None:
        audit["no_abstract"] += 1
        return elements
    walk_region(ab, "ABSTRACT", [], {}, elements, audit, [0])
    for e in elements:
        e["region"] = "ABSTRACT"
        e["sec"] = None
    return elements


def process_article(rec, enriched, sec_inventories, ev_map):
    pmc = rec["pmcid"]
    audit = {"excluded_sections": [], "empty_source_elements": 0,
             "special_empty_skipped": 0, "no_abstract": 0,
             "parse_error": False, "warnings": []}
    path = RAW / rec["source_id"] / rec["raw_file"]
    if path is None or not path.exists():
        audit["parse_error"] = True
        audit["warnings"].append("raw_file_missing")
        return [], audit, 0, 0, []
    root, warns = load_xml(path)
    audit["warnings"].extend(warns)
    if root is None:
        audit["parse_error"] = True
        return [], audit, 0, 0, []
    en = enriched.get(pmc, {})
    article = dict(en.get("article") or {})
    man = dict(en.get("manifest") or {})
    article["license_category"] = man.get("license_category")
    article["topic_ids"] = man.get("topic_ids") or []
    inv_index = {s["section_id"]: s for s in sec_inventories.get(pmc, [])
                 if s.get("section_id")}
    elements = []
    elements.extend(collect_abstract(root, audit))
    body = find_one(root, "body")
    if body is not None:
        walk_region(body, "BODY", [], inv_index, elements, audit, [0])
    back = find_one(root, "back")
    if back is not None:
        walk_region(back, "BACK", [], inv_index, elements, audit, [0])
    for i, e in enumerate(elements):
        e["el_index"] = i
    builder = ChunkBuilder(pmc, article, ev_map.get(pmc, {}), inv_index)
    for e in elements:
        ct = e["type"]
        if ct == "P":
            if len(e["text"].split()) > MAX_WORDS:
                builder.add_split_paragraph(e)
            else:
                builder.add_paragraph(e)
        else:
            builder.add_special(e, ct)
    builder.flush_buffer()
    scopes = [{"region": e.get("region"),
               "section_id": (e.get("sec") or {}).get("section_id")}
              for e in elements]
    return builder.chunks, audit, builder.overlap_parts, len(elements), scopes


def pct(v, k, n):
    """k-th inner divider percentile (k in 1..n-1) via statistics.quantiles."""
    v = sorted(v)
    if not v:
        return None
    import statistics
    vals = statistics.quantiles(v, n=n)
    if k < 1 or k >= n:
        return None
    return vals[k - 1]


def build_stats(all_chunks, per_article, audits, overlap_total):
    n_chunks = len(all_chunks)
    words = [c["word_count"] for c in all_chunks]
    chars = [c["character_count"] for c in all_chunks]
    chunks_per_article = sorted(per_article.values())
    ct_dist = Counter(c["content_type"] for c in all_chunks)
    region_dist = Counter(c["region"] for c in all_chunks)
    ev_dist = Counter(c["evidence_type"] or "UNKNOWN" for c in all_chunks)
    atype_dist = Counter(c["article_type"] or "UNKNOWN" for c in all_chunks)
    topic_dist = Counter(t for c in all_chunks for t in (c.get("topic_ids") or []))
    flag_counts = Counter()
    for c in all_chunks:
        for k, v in c["flags"].items():
            if isinstance(v, bool) and v:
                flag_counts[k] += 1
    def dist(v):
        return {"min": min(v) if v else None,
                "p25": pct(v, 1, 4) if v else None,
                "p50": pct(v, 2, 4) if v else None,
                "mean": round(sum(v) / len(v), 2) if v else None,
                "p75": pct(v, 3, 4) if v else None,
                "p90": pct(v, 9, 10) if v else None,
                "p95": pct(v, 19, 20) if v else None,
                "max": max(v) if v else None}
    stats = {
        "corpus_version": VERSION,
        "parameters": {"TARGET_WORDS": TARGET_WORDS, "MAX_WORDS": MAX_WORDS,
                       "MIN_CHUNK_WORDS": MIN_CHUNK_WORDS,
                       "MIN_MERGE_WORDS": MIN_MERGE_WORDS,
                       "OVERLAP_SENTENCES": OVERLAP_SENTENCES},
        "articles": len(per_article),
        "chunks_total": n_chunks,
        "chunks_per_article": {"min": min(chunks_per_article) if chunks_per_article else None,
                               "p25": pct(chunks_per_article, 1, 4),
                               "p50": pct(chunks_per_article, 2, 4),
                               "mean": round(sum(chunks_per_article) / len(chunks_per_article), 2)
                               if chunks_per_article else None,
                               "p75": pct(chunks_per_article, 3, 4),
                               "p95": pct(chunks_per_article, 19, 20),
                               "max": max(chunks_per_article) if chunks_per_article else None},
        "word_count": dist(words),
        "character_count": dist(chars),
        "overlap_sentences_applied": overlap_total,
        "by_region": dict(region_dist),
        "by_content_type": dict(ct_dist),
        "by_evidence_type": dict(ev_dist),
        "by_article_type": dict(atype_dist),
        "by_topic": dict(topic_dist),
        "flag_counts": dict(flag_counts),
    }
    return stats


def load_inputs():
    recs = [json.loads(l) for l in
            (MAN / "accepted_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    enriched = json.loads((META / "jats_enriched_metadata.json").read_text(encoding="utf-8"))
    sec_inv = json.loads((META / "jats_section_inventory.json").read_text(encoding="utf-8"))
    ev = json.loads((META / "evidence_type_enrichment.json").read_text(encoding="utf-8"))
    return recs, enriched, sec_inv, ev


def run_pipeline(recs, enriched, sec_inv, ev):
    all_chunks = []
    per_article = {}
    per_article_elements = {}
    audits = {}
    overlap_total = 0
    n_elements = 0
    for rec in recs:
        chunks, audit, ov, nel, scopes = process_article(
            rec, enriched, sec_inv, ev)
        per_article[rec["pmcid"]] = len(chunks)
        per_article_elements[rec["pmcid"]] = nel
        audits[rec["pmcid"]] = {k: v for k, v in audit.items()
                                if k != "excluded_sections"}
        audits[rec["pmcid"]]["excluded_sections"] = len(audit["excluded_sections"])
        audits[rec["pmcid"]]["element_scopes"] = scopes
        overlap_total += ov
        n_elements += nel
        all_chunks.extend(chunks)
    stats = build_stats(all_chunks, per_article, audits, overlap_total)
    return all_chunks, per_article, per_article_elements, audits, stats


def coverage_audit(all_chunks, per_article_elements):
    by_article = {}
    for c in all_chunks:
        by_article.setdefault(c["pmcid"], []).append(c)
    results = {}
    for pmc in sorted(by_article):
        chunks = by_article[pmc]
        elmap = {}
        for c in chunks:
            for ei in c["element_indexes"]:
                elmap.setdefault(ei, []).append(c["chunk_id"])
        n_el = per_article_elements.get(pmc, 0)
        dropped = [ei for ei in range(n_el) if ei not in elmap]
        multi = {ei: len(set(lst)) for ei, lst in elmap.items()
                 if len(set(lst)) > 1}
        mins = [min(c["element_indexes"]) for c in chunks]
        reorder = any(mins[i] > mins[i + 1] for i in range(len(mins) - 1))
        results[pmc] = {
            "n_elements": n_el,
            "n_chunks": len(chunks),
            "n_elements_covered": n_el - len(dropped),
            "dropped_elements": dropped,
            "multi_chunk_element_counts": multi,
            "reordered_sequence": reorder,
        }
    return results
def render_report(stats, cov, audits):
    L = []
    L.append("# ElderDocAI-GeriLit dev-v0.1 — Phase 3 Task 4 Quality Report")
    L.append("")
    L.append("> **Development chunking parameters only** (see `chunking_specification.md`). "
            "No retrieval/QA tuning was performed; frequencies below are structural "
            "descriptions, not retrieval performance.")
    L.append("")
    L.append("## Input corpus")
    L.append("")
    L.append("- Articles: **%s**" % stats["articles"])
    L.append("- Total chunks: **%s**" % stats["chunks_total"])
    L.append("")
    L.append("## Development chunking parameters")
    L.append("")
    L.append("| Parameter | Value |")
    L.append("|---|---:|")
    for k, v in stats["parameters"].items():
        L.append("| %s | %s |" % (k, v))
    L.append("")
    L.append("## Chunk word-count statistics")
    L.append("")
    wt = stats["word_count"]
    L.append("min %s · P25 %s · median %s · mean %s · P75 %s · P90 %s · P95 %s · max %s"
             % (wt["min"], wt["p25"], wt["p50"], wt["mean"], wt["p75"],
                wt["p90"], wt["p95"], wt["max"]))
    L.append("")
    L.append("## Chunks per article")
    L.append("")
    ca = stats["chunks_per_article"]
    L.append("min %s · P25 %s · median %s · mean %s · P75 %s · P95 %s · max %s"
             % (ca["min"], ca["p25"], ca["p50"], ca["mean"], ca["p75"],
                ca["p95"], ca["max"]))
    L.append("")
    L.append("## Distribution by region")
    L.append("")
    L.append("| Region | Chunks |")
    L.append("|---|---:|")
    for k, v in sorted(stats["by_region"].items(), key=lambda kv: -kv[1]):
        L.append("| %s | %s |" % (k, v))
    L.append("")
    L.append("## Distribution by content type")
    L.append("")
    L.append("| Content type | Chunks |")
    L.append("|---|---:|")
    for k, v in sorted(stats["by_content_type"].items(), key=lambda kv: -kv[1]):
        L.append("| %s | %s |" % (k, v))
    L.append("")
    L.append("## Distribution by evidence type (heuristic metadata)")
    L.append("")
    L.append("| Evidence type | Chunks |")
    L.append("|---|---:|")
    for k, v in sorted(stats["by_evidence_type"].items(), key=lambda kv: -kv[1]):
        L.append("| %s | %s |" % (k, v))
    L.append("")
    L.append("## Quality-flag counts (across all chunks)")
    L.append("")
    L.append("| Flag | Chunks |")
    L.append("|---|---:|")
    for k, v in sorted(stats["flag_counts"].items(), key=lambda kv: -kv[1]):
        L.append("| %s | %s |" % (k, v))
    L.append("")
    L.append("Overlap sentences applied (long-paragraph sentence overlaps): **%s**"
             % stats["overlap_sentences_applied"])
    L.append("")
    L.append("## Coverage audit summary")
    L.append("")
    n_drop = sum(len(v["dropped_elements"]) for v in cov.values())
    n_reorder = sum(1 for v in cov.values() if v["reordered_sequence"])
    n_multi_intended = sum(1 for v in cov.values()
                           if v["multi_chunk_element_counts"])
    L.append("- Articles audited: %d" % len(cov))
    L.append("- Dropped source elements: %d" % n_drop)
    L.append("- Reordered chunk sequences: %d" % n_reorder)
    L.append("- Articles with multi-chunk elements (long-paragraph overlaps only): %d"
             % n_multi_intended)
    L.append("")
    L.append("## Structural audit summary (per-article counts)")
    L.append("")
    parse_err = sum(1 for a in audits.values() if a.get("parse_error"))
    no_abs = sum(1 for a in audits.values() if a.get("no_abstract"))
    empty_el = sum(a.get("empty_source_elements", 0) for a in audits.values())
    sp_skip = sum(a.get("special_empty_skipped", 0) for a in audits.values())
    excl_sec = sum(a.get("excluded_sections", 0) for a in audits.values())
    L.append("- Articles with parse error: %d" % parse_err)
    L.append("- Articles without abstract: %d" % no_abs)
    L.append("- Empty source elements skipped (total): %d" % empty_el)
    L.append("- Special empty elements skipped (total): %d" % sp_skip)
    L.append("- Excluded sections (ack/funding/COI/refs, total): %d" % excl_sec)
    L.append("")
    return "\n".join(L) + "\n"


def main():
    global TARGET_WORDS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET_WORDS)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--sensitivity", default="")
    args = ap.parse_args()
    TARGET_WORDS = args.target

    recs, enriched, sec_inv, ev = load_inputs()
    all_chunks, per_article, per_article_elements, audits, stats = \
        run_pipeline(recs, enriched, sec_inv, ev)

    if args.stats_only:
        print(json.dumps({"target": TARGET_WORDS,
                          "chunks_total": stats["chunks_total"],
                          "articles": stats["articles"],
                          "mean_wc": stats["word_count"]["mean"],
                          "median_wc": stats["word_count"]["p50"]}, indent=2))
        return

    if args.sensitivity:
        out = []
        for t in [int(x) for x in args.sensitivity.split(",")]:
            TARGET_WORDS = t
            _, _, _, _, st = run_pipeline(*(recs, enriched, sec_inv, ev))
            out.append({"target_words": t,
                        "chunks_total": st["chunks_total"],
                        "mean_wc": st["word_count"]["mean"],
                        "median_wc": st["word_count"]["p50"],
                        "max_chunks_per_article": st["chunks_per_article"]["max"]})
        (META / "chunk_sensitivity.json").write_text(
            json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(out, indent=2))
        return

    with open(CHUNK_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    cov = coverage_audit(all_chunks, per_article_elements)
    (META / "chunk_statistics.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    (META / "chunk_quality_audit.json").write_text(
        json.dumps({"flag_counts": stats["flag_counts"],
                    "per_article": audits}, indent=2) + "\n", encoding="utf-8")
    (META / "chunk_provenance_audit.json").write_text(
        json.dumps(cov, indent=2) + "\n", encoding="utf-8")
    (META / "phase3_task4_quality_report.md").write_text(
        render_report(stats, cov, audits), encoding="utf-8")

    h = {n: sha(META / n) for n in
         ("chunk_statistics.json", "chunk_quality_audit.json",
          "chunk_provenance_audit.json", "phase3_task4_quality_report.md")}
    h["chunks.jsonl"] = sha(CHUNK_DIR / "chunks.jsonl")
    print(json.dumps({
        "articles": stats["articles"],
        "chunks_total": stats["chunks_total"],
        "hashes": h,
    }, indent=2))


if __name__ == "__main__":
    main()