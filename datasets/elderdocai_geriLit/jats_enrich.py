#!/usr/bin/env python3
"""ElderDocAI-GeriLit dev-v0.1 - Phase 3 Task 3: JATS metadata & evidence-type
enrichment + corpus quality audit.

Parses the 500 accepted JATS XML files (std-lib only, xml.etree.ElementTree,
namespace-stripped) and writes deterministic enriched metadata, section
inventory, structural audit, and evidence-type enrichment artifacts.

This is METADATA ENRICHMENT ONLY. No chunking / no RAG / no LLM.
"""
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from html import unescape

BASE = Path(r"C:\Users\chosun\Documents\ElderDocAI-System\datasets\elderdocai_geriLit")
MAN = BASE / "manifest"
META = BASE / "metadata"
RAW = BASE / "raw"
VAL = META / "validation"
VAL.mkdir(parents=True, exist_ok=True)

VERSION = "ElderDocAI-GeriLit-dev-v0.1"
ACCEPTED = MAN / "accepted_manifest.jsonl"

ET_CV = [
    "GUIDELINE_OR_CONSENSUS", "SYSTEMATIC_REVIEW", "META_ANALYSIS",
    "RANDOMIZED_CONTROLLED_TRIAL", "OBSERVATIONAL_STUDY",
    "DIAGNOSTIC_OR_VALIDATION_STUDY", "QUALITATIVE_OR_MIXED_METHODS",
    "NARRATIVE_REVIEW", "PROTOCOL", "METHODS_OR_TECHNICAL",
    "EDITORIAL_OR_COMMENTARY", "CASE_REPORT_OR_CASE_SERIES", "OTHER", "UNKNOWN",
]
CONF_CV = ["HIGH", "MEDIUM", "LOW"]


def pat(terms):
    return re.compile(r"(?<![a-z0-9])(" + "|".join(
        re.escape(t) for t in terms) + r")(?![a-z0-9])", re.I)


EV_RULES = [
    ("META_ANALYSIS", pat(["meta-analysis", "meta analysis", "metaanalysis",
                           "meta-analyses", "pooled analysis of"]), 2),
    ("SYSTEMATIC_REVIEW", pat(["systematic review", "systematic literature review",
                               "scoping review protocol"]), 2),
    ("RANDOMIZED_CONTROLLED_TRIAL", pat([
        "randomized controlled trial", "randomised controlled trial",
        "randomized clinical trial", "randomised clinical trial",
        "randomized trial", "randomised trial", "cluster randomized",
        "cluster randomised", "double-blind randomized", "double blind randomized",
        "double-blind randomised", "double blind randomised"]), 2),
    ("GUIDELINE_OR_CONSENSUS", pat([
        "clinical practice guideline", "practice guideline", "clinical guideline",
        "consensus statement", "consensus recommendations",
        "consensus guidance", "evidence-based guideline", "guideline for the",
        "position statement of"]), 2),
    ("DIAGNOSTIC_OR_VALIDATION_STUDY", pat([
        "diagnostic accuracy", "diagnostic test accuracy", "diagnostic test",
        "diagnostic performance", "validation study", "validity study",
        "screening accuracy", "screening performance", "predictive validity",
        "reliability and validity"]), 2),
    ("QUALITATIVE_OR_MIXED_METHODS", pat([
        "qualitative study", "qualitative research", "qualitative investigation",
        "qualitative analysis", "focus group", "grounded theory",
        "phenomenological", "semi-structured interview",
        "semistructured interview", "mixed methods", "mixed-method",
        "in-depth interviews", "thematic analysis"]), 2),
    ("PROTOCOL", pat(["study protocol", "trial protocol", "research protocol",
                      "protocol for a randomized", "protocol for a randomised"]), 2),
    ("CASE_REPORT_OR_CASE_SERIES", pat([
        "case report", "case series", "case study", "report of a case",
        "a case of"]), 2),
    ("OBSERVATIONAL_STUDY", pat([
        "cohort study", "prospective cohort", "retrospective cohort",
        "longitudinal study", "longitudinal cohort", "cross-sectional study",
        "cross sectional survey", "cross-sectional survey", "case-control study",
        "case control study", "case-control analysis", "retrospective study",
        "prospective study", "observational study", "population-based study",
        "secondary analysis of", "follow-up study", "nationwide study",
        "registry-based", "register-based"]), 2),
    ("NARRATIVE_REVIEW", pat(["literature review", "narrative review",
                              "review article", "contemporary review",
                              "state of the art review", "umbrella review",
                              "critical review"]), 2),
]

ATYPE_FALLBACK = {
    "editorial": ("EDITORIAL_OR_COMMENTARY", "LOW"),
    "commentary": ("EDITORIAL_OR_COMMENTARY", "LOW"),
    "letter": ("EDITORIAL_OR_COMMENTARY", "LOW"),
    "discussion": ("EDITORIAL_OR_COMMENTARY", "LOW"),
    "methods-article": ("METHODS_OR_TECHNICAL", "LOW"),
    "protocol": ("PROTOCOL", "MEDIUM"),
    "guideline": ("GUIDELINE_OR_CONSENSUS", "MEDIUM"),
    "review-article": ("NARRATIVE_REVIEW", "LOW"),
    "case-report": ("CASE_REPORT_OR_CASE_SERIES", "MEDIUM"),
    "case-series": ("CASE_REPORT_OR_CASE_SERIES", "MEDIUM"),
    "systematic-review": ("SYSTEMATIC_REVIEW", "MEDIUM"),
}

REGION_TITLE_RULES = [
    ("ACKNOWLEDGMENTS", ["acknowledg"], 1.0),
    ("FUNDING", ["funding", "financial support", "grant support", "sources of support"], 1.0),
    ("AUTHOR_CONTRIBUTIONS", ["author contribution", "authors' contribution", "authors contribution",
                              "contributions of authors", "author statement"], 1.0),
    ("CONFLICTS_OF_INTEREST", ["conflict of interest", "competing interest", "competing statement"], 1.0),
    ("DATA_AVAILABILITY", ["data availability", "availability of data", "data sharing statement",
                           "data availability statement"], 1.0),
    ("ETHICS", ["ethic", "institutional review board", "declaration of interest"], 1.0),
]

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


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_xml(path):
    """Parse JATS XML robustly; return (root, warnings)."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    xml = strip_doctype(raw)
    text = fix_entities(xml)
    try:
        return ET.fromstring(text), []
    except ET.ParseError as e:
        return None, ["xml_parse_error: %s" % e]


def xtext(el, *path, default=None):
    """First text content under a path (local names), cleaned."""
    if el is None:
        return default
    node = el
    for name in path:
        nxt = None
        for child in node:
            if local(child.tag) == name:
                nxt = child
                break
        if nxt is None:
            return default
        node = nxt
    t = "".join(node.itertext())
    return clean(t) if t else default


def all_children(el, name):
    if el is None:
        return []
    return [c for c in el if local(c.tag) == name]


def article_id(root, pub_type):
    for node in root.iter():
        if local(node.tag) == "article-meta":
            for c in node:
                if local(c.tag) == "article-id" and \
                        c.get("pub-id-type") == pub_type:
                    return clean(c.text)
            break
    return None


def find_one(root, name):
    for node in root.iter():
        if local(node.tag) == name:
            return node
    return None


def extract_authors(root):
    """Return (authors, warnings) from JATS contrib-group (contrib-type=author)."""
    authors = []
    warnings = []
    seen = set()
    for node in root.iter():
        if local(node.tag) != "contrib-group":
            continue
        for contrib in node:
            if local(contrib.tag) != "contrib":
                continue
            if contrib.get("contrib-type") not in (None, "author"):
                continue
            surname = given = suffix = prefix = orcid = None
            is_corresp = False
            aff_refs = []
            name_el = None
            for c in contrib:
                lc = local(c.tag)
                if lc == "name":
                    name_el = c
                elif lc == "surname" and surname is None:
                    surname = clean(c.text)
                elif lc == "given-names" and given is None:
                    given = clean(c.text)
                elif lc == "suffix" and suffix is None:
                    suffix = clean(c.text)
                elif lc == "prefix" and prefix is None:
                    prefix = clean(c.text)
                elif lc == "contrib-id" and c.get("contrib-id-type") == "orcid":
                    orcid = clean(c.text)
                elif lc == "xref":
                    rid = c.get("rid")
                    if c.get("ref-type") == "corresp":
                        is_corresp = True
                    if c.get("ref-type") == "aff" and rid:
                        aff_refs.append(rid)
            if name_el is not None:
                sn = given_n = None
                for c in name_el:
                    lc = local(c.tag)
                    if lc == "surname":
                        sn = clean(c.text)
                    elif lc == "given-names":
                        given_n = clean(c.text)
                surname = sn if sn else surname
                given = given_n if given_n else given
            display = " ".join(x for x in
                               ([prefix, given, surname, suffix] if surname else
                                [given, surname, suffix]) if x)
            if not surname and not given and not display:
                collab = clean(" ".join(c.text or "" for c in contrib
                                        if local(c.tag) == "collab"))
                if collab:
                    display = collab
                else:
                    continue
            order = len(authors) + 1
            # dedupe by identity
            key = (surname, given, orcid, display)
            if key in seen:
                continue
            seen.add(key)
            authors.append({
                "order": order,
                "surname": surname, "given_names": given,
                "suffix": suffix, "prefix": prefix,
                "full_name": display or None,
                "orcid": orcid, "is_corresponding": is_corresp,
                "affiliation_refs": aff_refs,
            })
    if not authors:
        warnings.append("no_author_contribs_found")
    return authors, warnings


def extract_affiliations(root):
    affs = []
    for node in root.iter():
        if local(node.tag) != "aff":
            continue
        aff_id = node.get("id")
        parts = []
        for tag in ("institution", "addr-line", "city", "country", "email"):
            for c in node.iter():
                if local(c.tag) == tag:
                    t = clean(c.text)
                    if t:
                        parts.append(t)
        text = clean("".join(node.itertext())) if (node.text or any(
            local(c.tag) in ("institution", "addr-line", "city", "country")
            for c in node)) else None
        affs.append({
            "aff_id": aff_id,
            "institution": xtext(node, "institution"),
            "department": None,
            "city": xtext(node, "city"),
            "country": xtext(node, "country"),
            "complete_text": text or clean(" ".join(parts)),
        })
    return affs


def walk_secs(el, depth, order, parent, region, out):
    """Recursive section walk over <sec> elements."""
    for child in el:
        if local(child.tag) != "sec":
            continue
        sid = child.get("id")
        label = None
        title = None
        sec_type = child.get("sec-type")
        for c in child:
            lc = local(c.tag)
            if lc == "label" and label is None:
                label = clean(c.text)
            elif lc == "title" and title is None:
                title = clean("".join(c.itertext()))
            if title is not None and label is not None:
                break
        if label is None:
            label = ""
        if title is None:
            title = ""
        sec_type = sec_type or None
        out.append({
            "section_id": sid,
            "label": label,
            "title": title,
            "depth": depth,
            "order": order,
            "parent_section_id": parent,
            "region": region,
            "section_type": sec_type,
        })
        order += 1
        order = walk_secs(child, depth + 1, order, sid, region, out)
    return order


def extract_sections(root):
    secs = []
    order = 0
    body = find_one(root, "body")
    if body is not None:
        order = walk_secs(body, 1, order, None, "BODY", secs)
    back = find_one(root, "back")
    if back is not None:
        order = walk_secs(back, 1, order, None, "BACK", secs)
    # classify region for each section by its title text
    for s in secs:
        tt = (s["title"] or "").lower()
        for region, kws, _thr in REGION_TITLE_RULES:
            if any(k in tt for k in kws):
                s["region"] = region
                break
    return secs


def text_of(el):
    return clean(" ".join(el.itertext())) if el is not None else None


def count_nodes(root, name):
    return sum(1 for _ in root.iter() if local(_.tag) == name)


def extract_content_types(root):
    c = {}
    c["has_abstract"] = count_nodes(root, "abstract") > 0
    body = find_one(root, "body")
    c["has_body"] = body is not None and len(text_of(body) or "") > 100
    c["reference_count"] = count_nodes(root, "ref")
    c["table_count"] = count_nodes(root, "table-wrap")
    c["figure_count"] = count_nodes(root, "fig")
    c["caption_count"] = count_nodes(root, "caption")
    c["boxed_text_count"] = count_nodes(root, "boxed-text")
    c["formula_count"] = (count_nodes(root, "disp-formula") +
                          count_nodes(root, "inline-formula"))
    c["supplementary_count"] = count_nodes(root, "supplementary-material")
    c["acknowledgments"] = count_nodes(root, "ack") > 0
    c["funding_statements"] = (count_nodes(root, "funding-group") > 0 or
                               count_nodes(root, "funding-statement") > 0)
    c["conflict_of_interest"] = count_nodes(root, "fn-group") > 0
    c["ethics_statements"] = count_nodes(root, "ethics") > 0
    c["data_availability"] = count_nodes(root, "data-availability") > 0
    return c


def classify_evidence(title, abstract_text, article_type):
    """Rule-based evidence-type classification (transparent, deterministic)."""
    hay = "%s . %s" % ((title or ""), (abstract_text or ""))
    lower = hay.lower()
    for et, rx, _boost in EV_RULES:
        m = rx.search(lower)
        if m:
            seg = 0 if m.start() < 1500 else 1
            conf = "HIGH" if seg == 0 else "MEDIUM"
            return {
                "evidence_type": et,
                "evidence_type_confidence": conf,
                "evidence_type_reason": (
                    "matched explicit design phrase '%s' at offset %d in "
                    "title/abstract" % (m.group(1), m.start())),
                "evidence_type_source": "JATS metadata + title/abstract heuristic",
            }
    if article_type in ATYPE_FALLBACK:
        et, conf = ATYPE_FALLBACK[article_type]
        return {
            "evidence_type": et,
            "evidence_type_confidence": conf,
            "evidence_type_reason": ("fallback from JATS article-type='%s'"
                                     % article_type),
            "evidence_type_source": "JATS article-type fallback",
        }
    return {
        "evidence_type": "UNKNOWN",
        "evidence_type_confidence": None,
        "evidence_type_reason": (
            "no explicit design indicator found in title/abstract and no "
            "JATS article-type fallback"),
        "evidence_type_source": "JATS metadata + title/abstract heuristic",
    }


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def audit_record(rec, build):
    """Build the structural audit entry for one article."""
    errs = build["errors"]
    warns = build["warnings"]
    meta = build["jats"]
    cc = build["crosscheck"]
    issues = []
    if "raw_file_missing" in errs:
        issues.append({"category": "REQUIRED_METADATA_FAILURE",
                       "code": "raw_file_missing"})
    if "xml_parse_failure" in errs:
        issues.append({"category": "REQUIRED_METADATA_FAILURE",
                       "code": "xml_parse_failure"})
    if meta is not None:
        if not meta.get("title"):
            issues.append({"category": "REQUIRED_METADATA_FAILURE",
                           "code": "empty_title"})
        if cc.get("pmcid_match") is False:
            issues.append({"category": "REQUIRED_METADATA_FAILURE",
                           "code": "pmcid_mismatch"})
        if cc.get("pmid_match") is False:
            issues.append({"category": "STRUCTURAL_WARNING",
                           "code": "pmid_mismatch"})
        if cc.get("doi_match") is False:
            issues.append({"category": "STRUCTURAL_WARNING",
                           "code": "doi_mismatch"})
        if cc.get("title_match") is False:
            issues.append({"category": "STRUCTURAL_WARNING",
                           "code": "title_mismatch"})
        if cc.get("year_match") is False:
            issues.append({"category": "STRUCTURAL_WARNING",
                           "code": "year_mismatch"})
        if cc.get("journal_match") is False:
            issues.append({"category": "STRUCTURAL_WARNING",
                           "code": "journal_mismatch"})
        if cc.get("license_permissive_confirmed") is False:
            issues.append({"category": "STRUCTURAL_WARNING",
                           "code": "license_not_confirmed"})
        ct = build["content_types"]
        if not ct.get("has_abstract"):
            issues.append({"category": "OPTIONAL_METADATA_MISSING",
                           "code": "no_abstract"})
        if not ct.get("has_body"):
            issues.append({"category": "OPTIONAL_METADATA_MISSING",
                           "code": "no_body"})
        if ct.get("reference_count", 0) == 0:
            issues.append({"category": "OPTIONAL_METADATA_MISSING",
                           "code": "no_references"})
        if not build["authors"]:
            issues.append({"category": "OPTIONAL_METADATA_MISSING",
                           "code": "no_authors"})
        if not build["affiliations"]:
            issues.append({"category": "OPTIONAL_METADATA_MISSING",
                           "code": "no_affiliations"})
        if not build["sections"]:
            issues.append({"category": "PARSE_WARNING",
                           "code": "no_sections_found"})
        sids = [s["section_id"] for s in build["sections"] if s["section_id"]]
        dup = sorted({x for x in sids if sids.count(x) > 1})[:5]
        if dup:
            issues.append({"category": "PARSE_WARNING",
                           "code": "duplicate_section_ids:" + ",".join(dup)})
    for w in warns:
        issues.append({"category": "PARSE_WARNING", "code": w})
    return {
        "pmcid": rec["pmcid"],
        "xml_parse_success": "xml_parse_failure" not in errs,
        "title_present": bool(meta and meta.get("title")),
        "pmcid_consistent": cc.get("pmcid_match"),
        "pmid_consistent": cc.get("pmid_match"),
        "doi_consistent": cc.get("doi_match"),
        "journal_consistent": cc.get("journal_match"),
        "year_consistent": cc.get("year_match"),
        "body_present": bool(build["content_types"].get("has_body")),
        "abstract_present": bool(build["content_types"].get("has_abstract")),
        "references_present": build["content_types"].get("reference_count", 0) > 0,
        "sections_parsed": len(build["sections"]) > 0,
        "authors_parsed": len(build["authors"]) > 0,
        "tables_parsed": build["content_types"].get("table_count", 0) > 0,
        "figures_parsed": build["content_types"].get("figure_count", 0) > 0,
        "supplementary_detected": build["content_types"].get("supplementary_count", 0) > 0,
        "issues": issues,
        "n_sections": len(build["sections"]),
        "n_authors": len(build["authors"]),
        "n_tables": build["content_types"].get("table_count", 0),
        "n_figures": build["content_types"].get("figure_count", 0),
        "n_references": build["content_types"].get("reference_count", 0),
    }


def build_record(rec):
    """Process one accepted manifest record -> enrichment dict."""
    pmcid = rec["pmcid"]
    source_id = rec["source_id"]
    raw_file = rec.get("raw_file")
    path = RAW / source_id / raw_file if raw_file else None
    out = {
        "corpus_version": rec.get("corpus_version"),
        "pmcid": pmcid, "source_id": source_id,
        "manifest": {k: rec.get(k) for k in
                     ("pmid", "doi", "title", "journal", "publication_year",
                      "article_type", "language", "license",
                      "license_category", "topic_ids", "sentinel",
                      "inclusion_status")},
        "jats": None, "authors": [], "affiliations": [],
        "sections": [], "content_types": {}, "evidence_type": None,
        "crosscheck": {}, "warnings": [], "errors": [],
    }
    if path is None or not path.exists():
        out["errors"].append("raw_file_missing")
        return out
    root, warns = load_xml(path)
    out["warnings"].extend(warns)
    if root is None:
        out["errors"].append("xml_parse_failure")
        return out
    meta = extract_article_meta(root)
    authors, awarns = extract_authors(root)
    affs = extract_affiliations(root)
    secs = extract_sections(root)
    content = extract_content_types(root)
    ev = classify_evidence(meta["title"], meta["abstract"], meta["article_type"])
    out["jats"] = meta
    out["authors"] = authors
    out["affiliations"] = affs
    out["sections"] = secs
    out["content_types"] = content
    out["evidence_type"] = ev
    out["warnings"].extend(awarns)
    out["warnings"].extend(author_aff_warnings(authors, affs))
    out["crosscheck"] = crosscheck(rec, meta)
    return out


def author_aff_warnings(authors, affs):
    w = []
    if authors and affs:
        used = set()
        for a in authors:
            used.update(a["affiliation_refs"] or [])
        aff_ids = {a["aff_id"] for a in affs}
        dangling = [r for r in used if r not in aff_ids]
        if dangling:
            w.append("author_affiliation_refs_not_resolved:%s" %
                     ",".join(sorted(dangling)[:5]))
    return w


def crosscheck(rec, meta):
    cc = {}
    cc["pmcid_match"] = bool(rec.get("pmcid") == meta.get("pmcid"))
    if rec.get("pmid"):
        cc["pmid_match"] = bool(str(rec.get("pmid")) == str(meta.get("pmid")))
    else:
        cc["pmid_match"] = None
    if rec.get("doi"):
        cc["doi_match"] = bool((rec.get("doi") or "").lower() ==
                               (meta.get("doi") or "").lower())
    else:
        cc["doi_match"] = None
    nt = lambda s: re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
    cc["title_match"] = bool(nt(rec.get("title")) == nt(meta.get("title")))
    if rec.get("publication_year") and meta.get("publication_year"):
        cc["year_match"] = bool(rec["publication_year"] == meta["publication_year"])
    else:
        cc["year_match"] = None
    cc["journal_match"] = bool(nt(rec.get("journal")) == nt(meta.get("journal")))
    lic_ok = False
    lurl = (meta.get("license") or {}).get("license_url") or ""
    ltxt = (meta.get("license") or {}).get("license_text") or ""
    lic_raw = (lurl + " " + ltxt).lower()
    if rec.get("license_category") == "PERMISSIVE":
        lic_ok = "creativecommons" in lic_raw or "cc0" in lic_raw
    elif rec.get("license_category") == "UNKNOWN":
        lic_ok = True
    cc["license_permissive_confirmed"] = lic_ok
    return cc


def find_descendant_text(el, name):
    if el is None:
        return None
    for node in el.iter():
        if local(node.tag) == name:
            return text_of(node)
    return None


def extract_article_meta(root):
    meta = {}
    am = None
    for node in root.iter():
        if local(node.tag) == "article-meta":
            am = node
            break
    meta["pmcid"] = article_id(root, "pmcid")
    meta["pmcid_version"] = article_id(root, "pmcid-ver")
    meta["pmid"] = article_id(root, "pmid")
    meta["doi"] = article_id(root, "doi")
    meta["article_type"] = root.get("article-type")
    meta["language"] = root.get("xml:lang") or root.get("lang") or None
    meta["title"] = xtext(am, "title-group", "article-title") or \
        find_descendant_text(root, "article-title")
    meta["journal"] = find_descendant_text(root, "journal-title")
    meta["journal_abbrev"] = None
    jm = find_one(root, "journal-meta")
    if jm is not None:
        for c in jm:
            if local(c.tag) == "journal-id":
                typ = c.get("journal-id-type")
                if typ in ("nlm-ta", "iso-abbrev"):
                    meta["journal_abbrev"] = clean(c.text)
                    break
    # publication year + full date where available
    year = None
    full_date = None
    for node in root.iter():
        if local(node.tag) == "pub-date":
            y = d = mo = None
            for c in node:
                lc = local(c.tag)
                if lc == "year":
                    y = clean(c.text)
                elif lc == "day":
                    d = clean(c.text)
                elif lc == "month":
                    mo = clean(c.text)
            if y and y.isdigit():
                year = int(y)
                if d and mo:
                    full_date = "%s-%s-%s" % (y, mo.zfill(2), d.zfill(2))
                break
        if local(node.tag) == "publication-date":
            iso = node.get("iso-8601-date")
            if iso:
                m = re.match(r"(\d{4})(?:-(\d{2})-(\d{2}))?", iso)
                if m:
                    year = int(m.group(1))
                    if m.group(2):
                        full_date = iso
                break
    meta["publication_year"] = year
    meta["publication_date"] = full_date
    lic_meta = {"license_text": None, "license_url": None}
    for node in root.iter():
        if local(node.tag) == "license":
            lic_meta["license_text"] = clean("".join(node.itertext()))
            href = node.get("{http://www.w3.org/1999/xlink}href") or \
                node.get("xlink:href")
            if href:
                lic_meta["license_url"] = href
            break
    meta["license"] = lic_meta
    meta["copyright_statement"] = None
    for node in root.iter():
        if local(node.tag) == "copyright-statement":
            meta["copyright_statement"] = clean("".join(node.itertext()))
            break
    meta["abstract"] = None
    ab = find_one(root, "abstract")
    if ab is not None:
        meta["abstract"] = text_of(ab)
    return meta
def make_report(recs, enriched, sections, audits, evs):
    n = len(recs)
    meta_present = sum(1 for b in enriched.values() if b["article"])

    def ctr(cond):
        return sum(1 for a in audits.values() if cond(a))
    authors_cnt = sum(1 for a in audits.values() if a["authors_parsed"])
    aff_cnt = sum(1 for b in enriched.values() if b["affiliations"])
    orcid_cnt = sum(1 for b in enriched.values()
                    if any(x.get("orcid") for x in b["authors"]))
    sect_cnt = sum(1 for a in audits.values() if a["sections_parsed"])
    ev_dist = Counter(e.get("evidence_type") for e in evs.values())
    conf_dist = Counter((e.get("evidence_type_confidence") or "n/a")
                        for e in evs.values())
    unk = sum(1 for e in evs.values() if e.get("evidence_type") == "UNKNOWN")
    issue_cat = Counter()
    for a in audits.values():
        for i in a["issues"]:
            issue_cat[i["category"]] += 1
    total_tables = sum(a["n_tables"] for a in audits.values())
    total_figs = sum(a["n_figures"] for a in audits.values())
    total_refs = sum(a["n_references"] for a in audits.values())
    total_secs = sum(a["n_sections"] for a in audits.values())
    with_abstract = ctr(lambda a: a["abstract_present"])
    with_body = ctr(lambda a: a["body_present"])
    with_refs = ctr(lambda a: a["references_present"])
    with_tables = ctr(lambda a: a["tables_parsed"])
    with_figs = ctr(lambda a: a["figures_parsed"])
    with_supp = ctr(lambda a: a["supplementary_detected"])
    title_ok = ctr(lambda a: a["title_present"])
    pmcid_ok = ctr(lambda a: a["pmcid_consistent"] is not False)
    pmid_ok = ctr(lambda a: a["pmid_consistent"] is not False)
    doi_ok = ctr(lambda a: a["doi_consistent"] is not False)
    year_ok = ctr(lambda a: a["year_consistent"] is not False)
    journal_ok = ctr(lambda a: a["journal_consistent"] is not False)
    lic_ok = ctr(lambda a: not any(i["code"] == "license_not_confirmed"
                                   for i in a["issues"]))
    return {
        "report_title": "ElderDocAI-GeriLit dev-v0.1 - Phase 3 Task 3 "
                        "JATS Metadata/Evidence-Type Enrichment & Quality Audit",
        "disclaimer": ("Evidence-type enrichment is a RULE-BASED METADATA "
                       "HEURISTIC, NOT a validated evidence hierarchy. JATS "
                       "article type is structural metadata and must NOT be "
                       "interpreted as evidence quality. No clinical quality, "
                       "risk-of-bias, or effectiveness assessment is made."),
        "objective": ("Enrich the 500 accepted articles with JATS-derived "
                      "metadata, authors, affiliations, section inventory, "
                      "content-type inventory, and transparent evidence-type "
                      "labels, and audit structural quality."),
        "input": {"corpus": VERSION, "accepted_records": n,
                  "raw_files_expected": n},
        "processed": {
            "xml_parse_success": meta_present,
            "xml_parse_failure": n - meta_present,
            "authors_present": authors_cnt,
            "affiliations_present": aff_cnt,
            "orcid_authors_present": orcid_cnt,
            "sections_parsed": sect_cnt,
            "title_present": title_ok,
            "abstract_present": with_abstract,
            "body_present": with_body,
            "references_present": with_refs,
            "tables_present": with_tables,
            "figures_present": with_figs,
            "supplementary_detected": with_supp,
        },
        "totals": {"sections": total_secs, "authors": sum(
            a["n_authors"] for a in audits.values()),
            "tables": total_tables, "figures": total_figs,
            "references": total_refs},
        "consistency_checks": {
            "pmcid_consistent": pmcid_ok, "pmid_consistent": pmid_ok,
            "doi_consistent": doi_ok, "journal_consistent": journal_ok,
            "year_consistent": year_ok, "license_permissive_confirmed": lic_ok,
        },
        "issue_categories": dict(issue_cat),
        "evidence_type_distribution": dict(ev_dist),
        "evidence_type_confidence_distribution": dict(conf_dist),
        "evidence_type_unknown_count": unk,
        "methodology_evidence_type": (
            "Deterministic keyword/phrase rules applied in fixed precedence "
            "order (meta-analysis > systematic review > RCT > guideline/"
            "consensus > diagnostic/validation > qualitative > protocol > "
            "case report > observational > narrative review) over JATS "
            "article-type + title + abstract; fallback to JATS article-type "
            "mappings; otherwise UNKNOWN. Confidence HIGH if the phrase "
            "occurs in the first 1500 chars, MEDIUM if later, LOW for "
            "article-type-only fallbacks."),
        "limitations": [
            "evidence-type labels are heuristic metadata, not evidence ranking",
            "JATS article-type is structural, not a quality signal",
            "authors/affiliations parsed only from JATS; not externally inferred",
            "abstract/body detection is conservative (body requires >100 chars)",
            "conflict-of-interest detection uses fn-group presence only",
        ],
    }


def render_md_report(r):
    L = []
    L.append("# " + r["report_title"])
    L.append("")
    L.append("> " + r["disclaimer"])
    L.append("")
    L.append("## 1. Objective")
    L.append("")
    L.append(r["objective"])
    L.append("")
    L.append("## 2. Input corpus")
    L.append("")
    L.append("- Corpus: `%s`" % r["input"]["corpus"])
    L.append("- Accepted records processed: %d" % r["input"]["accepted_records"])
    L.append("")
    L.append("## 3. Articles processed")
    L.append("")
    L.append("| Metric | Count |")
    L.append("|---|---:|")
    for k, v in r["processed"].items():
        L.append("| %s | %s |" % (k.replace("_", " "), v))
    L.append("")
    L.append("## 4. Totals across articles")
    L.append("")
    L.append("| Element | Total |")
    L.append("|---|---:|")
    for k, v in r["totals"].items():
        L.append("| %s | %s |" % (k, v))
    L.append("")
    L.append("## 5. Manifest/JATS consistency checks")
    L.append("")
    L.append("| Check | Consistent |")
    L.append("|---|---:|")
    for k, v in r["consistency_checks"].items():
        L.append("| %s | %s |" % (k, v))
    L.append("")
    L.append("## 6. Issue categories")
    L.append("")
    L.append("| Category | Count |")
    L.append("|---|---:|")
    for k, v in sorted(r["issue_categories"].items()):
        L.append("| %s | %s |" % (k, v))
    L.append("")
    L.append("## 7. Evidence-type distribution (rule-based heuristic)")
    L.append("")
    L.append("| Evidence type | Articles |")
    L.append("|---|---:|")
    for k, v in sorted(r["evidence_type_distribution"].items(),
                       key=lambda kv: -kv[1]):
        L.append("| %s | %s |" % (k, v))
    L.append("")
    L.append("Unknown count: **%s**" % r["evidence_type_unknown_count"])
    L.append("")
    L.append("## 8. Evidence-type confidence")
    L.append("")
    L.append("| Confidence | Articles |")
    L.append("|---|---:|")
    for k, v in sorted(r["evidence_type_confidence_distribution"].items(),
                       key=lambda kv: -kv[1]):
        L.append("| %s | %s |" % (k, v))
    L.append("")
    L.append("## 9. Evidence-type heuristic methodology")
    L.append("")
    L.append(r["methodology_evidence_type"])
    L.append("")
    L.append("## 10. Limitations")
    L.append("")
    for lim in r["limitations"]:
        L.append("- " + lim)
    L.append("")
    return "\n".join(L) + "\n"


def main():
    recs = [json.loads(l) for l in
            ACCEPTED.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(recs) != 500:
        print("ERROR: accepted manifest has %d records, expected 500"
              % len(recs))
        return
    enriched = OrderedDict()
    sections = OrderedDict()
    audits = OrderedDict()
    evs = OrderedDict()
    for rec in recs:
        b = build_record(rec)
        pmcid = rec["pmcid"]
        enriched[pmcid] = {
            "corpus_version": b["corpus_version"],
            "pmcid": pmcid, "source_id": b["source_id"],
            "manifest": b["manifest"], "article": b["jats"],
            "authors": b["authors"], "affiliations": b["affiliations"],
            "content_types": b["content_types"], "crosscheck": b["crosscheck"],
            "warnings": b["warnings"], "errors": b["errors"],
        }
        sections[pmcid] = b["sections"]
        audits[pmcid] = audit_record(rec, b)
        evs[pmcid] = dict(b["evidence_type"] or {})

    def emit(name, obj):
        p = META / name
        payload = json.dumps(obj, indent=1, ensure_ascii=False) + "\n"
        p.write_text(payload, encoding="utf-8")
        return sha(p)

    h1 = emit("jats_enriched_metadata.json", enriched)
    h2 = emit("jats_section_inventory.json", sections)
    h3 = emit("jats_structural_audit.json", audits)
    h4 = emit("evidence_type_enrichment.json", evs)
    report_json = make_report(recs, enriched, sections, audits, evs)
    h4b = emit("phase3_task3_quality_report.json", report_json)
    (META / "phase3_task3_quality_report.md").write_text(
        render_md_report(report_json), encoding="utf-8")
    h5 = sha(META / "phase3_task3_quality_report.md")
    meta_sha = hashlib.sha256(
        ("%s%s%s%s" % (h1, h2, h3, h4)).encode()).hexdigest()
    print(json.dumps({
        "accepted_processed": len(recs),
        "sha_enriched_metadata": h1,
        "sha_section_inventory": h2,
        "sha_structural_audit": h3,
        "sha_evidence_type": h4,
        "sha_quality_report_json": h4b,
        "sha_quality_report_md": h5,
        "combined_meta_sha256": meta_sha,
    }, indent=2))


if __name__ == "__main__":
    main()
