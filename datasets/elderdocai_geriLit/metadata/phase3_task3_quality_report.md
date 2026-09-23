# ElderDocAI-GeriLit dev-v0.1 - Phase 3 Task 3 JATS Metadata/Evidence-Type Enrichment & Quality Audit

> Evidence-type enrichment is a RULE-BASED METADATA HEURISTIC, NOT a validated evidence hierarchy. JATS article type is structural metadata and must NOT be interpreted as evidence quality. No clinical quality, risk-of-bias, or effectiveness assessment is made.

## 1. Objective

Enrich the 500 accepted articles with JATS-derived metadata, authors, affiliations, section inventory, content-type inventory, and transparent evidence-type labels, and audit structural quality.

## 2. Input corpus

- Corpus: `ElderDocAI-GeriLit-dev-v0.1`
- Accepted records processed: 500

## 3. Articles processed

| Metric | Count |
|---|---:|
| xml parse success | 500 |
| xml parse failure | 0 |
| authors present | 500 |
| affiliations present | 500 |
| orcid authors present | 286 |
| sections parsed | 499 |
| title present | 500 |
| abstract present | 499 |
| body present | 499 |
| references present | 499 |
| tables present | 476 |
| figures present | 373 |
| supplementary detected | 228 |

## 4. Totals across articles

| Element | Total |
|---|---:|
| sections | 9274 |
| authors | 4225 |
| tables | 1649 |
| figures | 1005 |
| references | 28390 |

## 5. Manifest/JATS consistency checks

| Check | Consistent |
|---|---:|
| pmcid_consistent | 500 |
| pmid_consistent | 500 |
| doi_consistent | 500 |
| journal_consistent | 500 |
| year_consistent | 500 |
| license_permissive_confirmed | 500 |

## 6. Issue categories

| Category | Count |
|---|---:|
| OPTIONAL_METADATA_MISSING | 3 |
| PARSE_WARNING | 2 |
| STRUCTURAL_WARNING | 1 |

## 7. Evidence-type distribution (rule-based heuristic)

| Evidence type | Articles |
|---|---:|
| UNKNOWN | 185 |
| OBSERVATIONAL_STUDY | 146 |
| RANDOMIZED_CONTROLLED_TRIAL | 60 |
| NARRATIVE_REVIEW | 37 |
| META_ANALYSIS | 21 |
| QUALITATIVE_OR_MIXED_METHODS | 21 |
| SYSTEMATIC_REVIEW | 15 |
| EDITORIAL_OR_COMMENTARY | 4 |
| DIAGNOSTIC_OR_VALIDATION_STUDY | 3 |
| PROTOCOL | 3 |
| CASE_REPORT_OR_CASE_SERIES | 3 |
| GUIDELINE_OR_CONSENSUS | 1 |
| METHODS_OR_TECHNICAL | 1 |

Unknown count: **185**

## 8. Evidence-type confidence

| Confidence | Articles |
|---|---:|
| HIGH | 275 |
| n/a | 185 |
| LOW | 31 |
| MEDIUM | 9 |

## 9. Evidence-type heuristic methodology

Deterministic keyword/phrase rules applied in fixed precedence order (meta-analysis > systematic review > RCT > guideline/consensus > diagnostic/validation > qualitative > protocol > case report > observational > narrative review) over JATS article-type + title + abstract; fallback to JATS article-type mappings; otherwise UNKNOWN. Confidence HIGH if the phrase occurs in the first 1500 chars, MEDIUM if later, LOW for article-type-only fallbacks.

## 10. Limitations

- evidence-type labels are heuristic metadata, not evidence ranking
- JATS article-type is structural, not a quality signal
- authors/affiliations parsed only from JATS; not externally inferred
- abstract/body detection is conservative (body requires >100 chars)
- conflict-of-interest detection uses fn-group presence only

