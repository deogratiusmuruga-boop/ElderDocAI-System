# ElderDocAI-GeriLit dev-v0.1 — Phase 3 Task 4
## JATS-Aware, Section-Aware Chunking Specification

> **Status: Development specification for `ElderDocAI-GeriLit-dev-v0.1` (500 accepted articles).**
> All numeric parameters are **PROVISIONAL DEVELOPMENT PARAMETERS** (Task 4). They were
> selected from corpus structural statistics and engineering considerations, **NOT** tuned
> against Gold96, PubMedQA, TREC-CDS, BioASQ, or any retrieval/QA benchmark.

---

## 1. Chunking objective

Transform each accepted JATS article into globally-unique, provenance-rich, section-aware
text chunks suitable for a future embedding/indexing stage (Phase 3 Task 5+). Chunks must
preserve article identity, section hierarchy, evidence metadata, and exact (normalized)
source text, and must support deterministic re-generation.

## 2. Text extraction rules

- Source: `raw/{source_id}/{raw_file}` JATS XML (byte-identical to Task 2 downloads).
- Robust parse (std-lib `xml.etree.ElementTree`): strip the `<!DOCTYPE …>` declaration, decode
  named entities (HTML5 set via `html.unescape`; unknown named entities dropped), then parse.
- Text of an element = concatenation of `itertext()` **in document order**, with whitespace
  collapsed to single spaces and outer whitespace stripped. This preserves punctuation,
  numbers, units, abbreviations, scientific terms, and inline citation markers. No other
  normalization is applied. Documented normalization: **whitespace collapsing only**.

## 3. Included JATS regions

| Region | Included | Notes |
|---|---|---|
| FRONT / TITLE | Yes (as article metadata on every chunk) | Not a standalone body chunk |
| ABSTRACT | Yes | Dedicated `region=ABSTRACT`, `content_type=ABSTRACT` |
| BODY (all `<sec>`) | Yes | Primary evidence content |
| BACK — substantive sections | Yes (DATA AVAILABILITY, ETHICS/DECLARATIONS, STUDY LIMITATIONS, METHODS, APPENDIX, SUPPLEMENTARY NOTES, STATEMENTS incl. consent/sharing) | Selected by JATS structure + title keyword policy (3a) |
| BACK — non-substantive sections | No | See §4 |

### 3a. Substantive-section policy (JATS-structure based)
A `<sec>` under `<body>` is always included. A `<sec>` under `<back>` (or any `<sec>`/`<notes>`
labeled as an excluded type) is included only when its normalized title matches a substantive
keyword set: `data availability`, `ethic`, `declaration`, `limitation`, `methods`, `appendix`,
`supplementary`, `consent`, `data sharing`, `statements`.

## 4. Excluded JATS regions / elements

- `<ref-list>` (References) — **never** ordinary evidence chunks (§13).
- `<ack>` (Acknowledgments) and sections titled `acknowledg*`.
- `<funding-group>`, `funding*` sections.
- Author information, author contributions, affiliations, copyright/license/publisher metadata.
- `conflict*` / `competing interest` sections (not substantive scientific evidence).
- `<bio>`, correspondence, `<fn-group>` footnotes (preserved only in structural metadata).
- Nested `<table-wrap>`, `<fig>`, `<boxed-text>` are **re-typed** (§11–13), not mixed into prose.

## 5. Section-boundary behavior

- Chunk accumulation **never** crosses a leaf-section boundary. When the next element belongs
  to a different section (or a section ends), the in-progress chunk is closed first.
- `Results` and `Discussion` can never be merged. Parent sections without direct children and
  leaf subsections are handled as distinct chunking scopes.

## 6. Subsection behavior

- The document hierarchy is walked depth-first. The innermost leaf `<sec>` is the chunking
  scope. Each chunk inherits the leaf section metadata; ancestry is recoverable through
  `parent_section_id` (from the Task 3 section inventory).
## 7. Target chunk size
- **TARGET_WORDS = 220** — corpus paragraph median ≈ 97, P75 ≈ 152, P90 ≈ 223 → a typical
  paragraph occupies one chunk; only short consecutive paragraphs are merged.
- **MAX_WORDS = 300** — paragraphs above this are split (§10).
- Both are **PROVISIONAL-DEV** parameters.

## 8. Minimum chunk size
- **MIN_CHUNK_WORDS = 50** — prose chunks below this are flagged `is_too_short` (informational).
  Extremely short but meaningful subsections are still emitted as single chunks; never padded,
  never merged with unrelated sections.
- **MIN_MERGE_WORDS = 100** — consecutive paragraphs in the same section are merged until the
  running chunk reaches ≥ MIN_MERGE_WORDS or TARGET_WORDS or the section ends.

## 9. Overlap strategy
- **OVERLAP_SENTENCES = 1**, applied **only** when a single oversized paragraph is split across
  chunks (§10): the final sentence of chunk _n_ is repeated as the first sentence of chunk _n+1_.
- **Overlap never crosses a major section boundary** and is never applied between distinct
  paragraphs (whole-paragraph boundaries already provide clean separation).

## 10. Handling of long paragraphs
- A paragraph with word_count > MAX_WORDS is split by **deterministic sentence boundaries**
  (regex `[.!?](?=\s+["'(\[]*[A-Z])`), preserving order and applying the §9 one-sentence overlap.
  Flags: `paragraph_split=true`, `sentence_split=true`.
- If a *single sentence* exceeds MAX_WORDS, it is kept as one chunk with `unsplit_long_sentence`.

## 11. Handling of tables
- `content_type=TABLE`. Each `<table-wrap>` becomes its own chunk (never merged into prose).
- Representation: `Caption: <caption>` + `Rows: header1 | header2 / cell… | cell…`, flattened
  deterministically from the underlying XHTML table. Table ID, order, section provenance kept.
- No visual fidelity claimed.

## 12. Handling of figures and captions
- `content_type=FIGURE_CAPTION`. Each `<fig>` caption (`<caption>` + `<label>` + `<title>`)
  becomes its own chunk with `figure_id`, figure order, section provenance. No image processing.

## 13. Handling of boxed text / special content
- `BOXED_TEXT` for `<boxed-text>`; `DISP_QUOTE` for `<disp-quote>`; `FORMULA` for `<disp-formula>`
  (formula text only). Each is its own chunk. If a special element yields no usable text it is
  skipped and recorded (`special_empty_skipped`).

## 14. Handling of references
- References stay out of the ordinary chunk stream; available only via Task 3
  `content_types.reference_count` for provenance/audit. **No reference chunks are produced.**

## 15. Metadata inheritance
Every chunk inherits article metadata (pmcid, pmid, doi, title, journal, publication_year,
language, license_category, article_type, topic_ids, corpus_version) and evidence metadata
(evidence_type, evidence_type_confidence) from the frozen Task 3 artifacts. Section metadata
comes from `jats_section_inventory.json`. Nothing is re-derived independently.

## 16. Provenance representation
- `source_locator = "{PMCID}:{REGION}:{section_id or '-'}:{elementfirst}..{elementlast}"`.
- Each paragraph/special element receives a stable global per-article element index; indices
  are consecutive and deterministic.

## 17. Chunk ID generation
- `chunk_id = "{PMCID}__{region}__{section_order or '000'}__{content_type}__{ci:05d}"` where
  `ci` is the global zero-based chunk counter within the article. PMCID + `ci` → globally
  unique, no UUIDs, deterministic.

## 18. Deterministic ordering
Chunks emitted in document order: article (accepted-manifest order) → region (ABSTRACT then
BODY depth-first) → section `order` → element index → chunk index within section.

## 19. Validation rules
500 PMCIDs, globally-unique chunk IDs, no empty chunks, required fields, evidence metadata
present, valid parent_section_id, no section-boundary crossing, no reference chunks,
provenance complete, deterministic repeated run, reproducible statistics. Full list in
`validate_task4.py`.

## 20. Development-parameter status
| Parameter | Value | Status |
|---|---|---|
| TARGET_WORDS | 220 | PROVISIONAL-DEV |
| MAX_WORDS | 300 | PROVISIONAL-DEV |
| MIN_CHUNK_WORDS | 50 | PROVISIONAL-DEV |
| MIN_MERGE_WORDS | 100 | PROVISIONAL-DEV |
| OVERLAP_SENTENCES | 1 | PROVISIONAL-DEV |

Selection basis: Task-4 structural probe over all 500 articles (19,839 paragraphs; word-count
min 1 / P25 59 / median 97 / mean 121 / P75 152 / P90 223 / P95 288 / P99 490 / max 4236).
**Not** selected by retrieval or benchmark performance.