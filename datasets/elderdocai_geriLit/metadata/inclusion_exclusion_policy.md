# ElderDocAI-GeriLit — Inclusion / Exclusion Policy (dev-v0.1)

## Inclusion criteria (ALL required for acceptance)
- I1. PMCID present and in the PMC OA Subset (verified via `oa.fcgi` membership/license response).
- I2. Full text available and successfully downloaded (non-empty `.nxml`, contains `<article`, expected PMCID present).
- I3. Metadata integrity: at least one of PMCID/PMID/DOI; a title; a journal or year determination not required for acceptance but recorded as UNKNOWN when absent.
- I4. Relevance anchor: article matched at least one topic query (query membership) and/or contains the topic or aging-relevant terms in its title.
- I5. Language: English (`lang` starting with `en`) or UNKNOWN if not determinable; explicit non-English excluded (E4).
- I5. License permissive (PERMISSIVE) per `license_policy.md` (CC BY / CC0 / equivalent); RESTRICTED/UNKNOWN excluded from the redistributable dev corpus.
- I6. Published 2015-present OR pre-2015 sentinel (sentinel=true with reason).

## Exclusion rules (reason codes)
- E1 `no_full_text` — no usable full-text `.nxml` after download attempt (missing/zero-byte/corrupt/HTML error page).
- E2 `irrelevant_to_elderly_aging` — no topic-query membership and no aging-relevance terms found.
- E3 `non_english` — explicit non-English language detected.
- E4 `license_restricted` — license contains NC/ND or otherwise not PERMISSIVE.
- E5 `license_unknown` — license metadata absent/ambiguous (UNKNOWN).
- E6 `not_oa_subset` — `oa.fcgi` returned no OA package (free-to-read but not OA-Subset).
- E7 `not_peer_reviewed_article` — detected type outside accepted set (editorial, letter, comment, news, erratum/correction, retraction, conference abstract, biography) without substantive evidence.
- E8 `duplicate` — deduplicated against PMCID/PMID/DOI/normalized-title (retained record keeps all identifiers).
- E9 `retracted` — retraction notice detected in article meta/title.
- E10 `metadata_integrity_fail` — required identifiers missing or contradictory.
- E11 `benchmark_protected` — ID appears in the protected-document stoplist.
- E12 `pre2015_non_sentinel` — pre-2015 without a documented sentinel reason.

## Evidence-type preference (curation preference, NOT an evidence-quality ranking)
1. Clinical practice guidelines / consensus statements
2. Systematic reviews
3. Meta-analyses
4. Randomized controlled trials
5. Prospective/retrospective observational studies
6. Evidence syntheses
7. High-quality narrative reviews
8. Other relevant peer-reviewed research

Observational studies are NOT automatically rejected. Article-type hierarchy is a corpus-curation preference, not a clinical evidence ranking.

## Borderline articles
Not silently excluded: borderline cases are recorded in the excluded manifest with the dominant reason code and a note when applicable.