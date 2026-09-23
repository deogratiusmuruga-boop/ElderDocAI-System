# PHASE 3 TASK 8 — FINAL REPORT
## GeriLit Benchmark Acquisition and Relevance-Mapping Audit

## 1. Task status
**COMPLETE — NO VALID QUANTITATIVE BENCHMARK ESTABLISHED.** Standalone
acquisition/mapping audit only. No corpus modification, no retrieval-metric
computation, no ground-truth fabrication, no production integration, no commit/push.

## 2. Repository state before implementation
- Branch `feature/mimic-pmc-migration` @ HEAD `8b0696c` (unchanged).
- `.gitignore` modified (pre-existing raw-ignore rules) + untracked
  `datasets/elderdocai_geriLit/`, `datasets/mimic_demo/`, pre-existing excluded files.
- Frozen Task 6 hashes recorded prior: chunks `62bfdde3…`, embeddings `b0905d2f…`,
  FAISS `b7016b9d…`, BM25 `17f50324…`, row_mapping `4d649f81…`.

## 3. Benchmarks investigated
1. PubMedQA (`pqa_labeled`) — external biomedical QA (yes/no/maybe), MIT.
2. TREC-CDS 2014–2016 — external document-level clinical IR with graded qrels.
3. BioASQ (Task b / 11-series) — expert biomedical semantic QA with golden docs/snippets.

## 4. Acquisition sources
- PubMedQA: official HF datasets-server rows API for `qiaojin/PubMedQA`
  (`huggingface.co/datasets/qiaojin/PubMedQA`; official project GitHub/pubmedqa.github.io).
- TREC-CDS: NIST/TREC (`trec.nist.gov/data/clinical.html`, `trec-cds.org/2016.html`);
  qrels require TREC registration (not acquired).
- BioASQ: `bioasq.org` (participants area registration; not acquired).

## 5. Benchmark versions/dates
- PubMedQA `pqa_labeled`: HF mirror last modified 2024-03-06; fetched 2026-09-18
  (identifier statistics only; dataset content NOT persisted).
- TREC-CDS: track years 2014/2015/2016 (2016 used MIMIC-III admission notes +
  PMC OA snapshot of 1.25M articles, NXML, PMCID document IDs).
- BioASQ: 2026 = 14th challenge; BioASQ-QA corpus published in Scientific Data 10:170 (2023).

## 6. Query counts
- PubMedQA `pqa_labeled`: **1,000 items** fetched (published labeled set; official
  leaderboard test subset = 500 of these).
- TREC-CDS: **30 topics/year** (2014–2016).
- BioASQ: varies per challenge batch (hundreds of expert questions per year; exact
  count requires registration).

## 7. Relevance-judgment availability
- PubMedQA: each item has exactly one relevant document (the abstract identified by
  `pubid` = PubMed PMID). No graded qrels; relevance is implicit single-document.
- TREC-CDS: **graded qrels exist** ("definitely relevant" / "potentially relevant";
  NDCG-based). Requires NIST/TREC materials — not acquired locally.
- BioASQ: golden relevant-article sets + snippets + ideal answers; distributed to
  registered participants — not acquired locally.

## 8. Document-identifier formats
- PubMedQA: `pubid` = PubMed PMID (example: 21645374).
- TREC-CDS: **PMCID only** ("we are only concerned with PMCIDs"; article NXML named
  `<PMCID>.nxml`). Bidirectional PMID↔PMCID normalizable via NCBI.
- BioASQ: golden documents referenced by PubMed IDs/URLs (PMID/PMCID mix; exact
  format per challenge JSON).

## 9. GeriLit PMCID coverage
- GeriLit = 500 PMCIDs, 499 PMIDs present (1 null), frozen & unchanged.

## 10. Query-to-GeriLit coverage
- **PubMedQA: 0 / 1,000 items (0.0000%)** have a PMID present in GeriLit.
- **TREC-CDS:** not computable without NIST qrels; judged-PMCID overlap with the
  frozen 500 unknown — expected low (GeriLit is geriatric-topic-curated; TREC topics
  are general clinical cases).
- **BioASQ:** not computable without registration; expected low for the same reason.
- **Conclusion: no candidate provides a non-trivial compatible query set.**

## 11. Identifier-mapping methodology
- PubMedQA `pubid` (PMID, leading zeros stripped) → `accepted_manifest.pmid` →
  `pmcid` → `chunks.jsonl` chunk IDs. Implemented deterministically in
  `metadata/task8_benchmark_mapping_audit.py`. No content persisted; identifiers +
  statistics only.

## 12. Mapping statistics (PubMedQA, only acquired candidate)
- Items fetched: 1,000 · unique pubids: 1,000 · GeriLit PMIDs: 499
- Mapped into GeriLit: **0** · coverage (items): **0.0%** · coverage (unique pubid): 0.0%
- Unmapped sample: 21645374, 16418930, 9488747, 17208539, 10808977
- Mapped PMCID sample: [] (none)

## 13. Benchmark-suitability assessment
- **PubMedQA**: identifier-compatible in principle (PMID→PMCID), but **0% coverage** →
  unsuitable for GeriLit retrieval evaluation (retrieval-quality metrics would be
  meaningless/empty). Best used later for **QA/grounding** against its own abstracts,
  not for GeriLit retrieval.
- **TREC-CDS**: highest *structural* compatibility (PMCID-domain qrels, graded
  relevance, official IR metrics) — but requires NIST registration and frozen
  500-article overlap is unknown/expected low. Suitable only after (a) acquiring qrels
  and (b) an overlap audit showing sufficient judged documents in GeriLit; otherwise it
  evaluates a *subset*, not the corpus.
- **BioASQ**: registered-access golden PMIDs/PMCIDs; likely low overlap; suited as a
  QA/grounding benchmark, not a GeriLit-retrieval benchmark without an overlap audit.
## 14. Licensing/access limitations
- PubMedQA: MIT, public, ungated — acquired identifiers; **content not stored/redistributed**.
- TREC-CDS: NIST/TREC registration; qrels distribution governed by NIST; not bypassed.
- BioASQ: bioasq.org registration/login; challenge data use conditions per organizers;
  not bypassed.
- No protected-benchmark content was exposed, redistributed, or committed.

## 15. Files created
- `datasets/elderdocai_geriLit/metadata/task8_benchmark_mapping_audit.py`
- `datasets/elderdocai_geriLit/metadata/task8_mapping_results.json` (identifiers+stats)
- `datasets/elderdocai_geriLit/metadata/validation/task8_validation.json`
- `datasets/elderdocai_geriLit/metadata/validation/task8_validation.md`
- `datasets/elderdocai_geriLit/metadata/phase3_task8_benchmark_audit_report.md` (this file)

## 16. Files modified
**None.** No legacy retrieval code, no Task 2–7 frozen artifacts, no corpus/index/model
files modified.

## 17. Frozen-artifact integrity
Verified unchanged (SHA-256) after the audit:
- chunks `62bfdde3c7e77cf56112e79a71bc79bdea70767f6b2625701e392bbb6581c6b3`
- embeddings `b0905d2f96903dadc3cf5b4a737f330853656f846955de706f21bd714ce2dacf`
- FAISS `b7016b9dabbab08ee68f875007b6db2b013af4472b1a3c0958af8a566b59f2b3`
- BM25 `17f503240f2f9a13abf58c94359884b03b06bec1a5a59f63f921328e2a2c70c9`
- row_mapping `4d649f81d554c41ec7b63c65dce887466dfa6245745c568315735f1b93a1872b`

## 18. Git status
- Branch `feature/mimic-pmc-migration`, HEAD `8b0696c` unchanged.
- Staged: none (`git diff --cached` empty). Unstaged: `.gitignore` (pre-existing rules).
- Untracked: `datasets/elderdocai_geriLit/`, `datasets/mimic_demo/`, pre-existing excluded
  files. **Nothing committed or pushed.**

## 19. Limitations
- TREC-CDS and BioASQ coverage could not be computed (registration required; no
  local qrels/goldens). PubMedQA coverage is exact (0%).
- PubMedQA identifiers fetched via public API are a proxy for the official test subset
  (the 1,000-item labeled set contains the 500-item official test).
- GeriLit is geriatric-topic-curated; external general/biomedical benchmarks are
  thematically orthogonal, so near-zero overlap is structurally expected.
- No metric (Recall/Precision/MRR/nDCG) was computed, as required for an audit.

## 20. Recommendation for the subsequent evaluation task
1. **Do not use PubMedQA for GeriLit retrieval metrics** (0% overlap).
2. **Optionally** acquire TREC-CDS qrels via NIST registration (PMCID-domain, graded),
   then run an explicit judged-PMCID↔GeriLit overlap audit before any retrieval
   evaluation; expect low coverage and evaluate with the caveat that only overlapping
   topics/documents are scored.
3. Retain PubMedQA/BioASQ as **QA/grounding** benchmarks over their *own* evidence, not
   over the GeriLit index.
4. Pending any qualifying overlap, GeriLit retrieval evaluation remains
   **functional/deterministic only** (as in Task 7), with Gold96-v2 (GeriLit-grounded,
   built corpus-first after freeze) as the internal grounding benchmark of record.

## 21. Explicit stop condition
Phase 3 Task 8 completed as an acquisition/mapping audit. **No modification of
`rag_chat.py`/API/frontend/reliability/manuscript, no GeriLit production integration, no
final retrieval benchmark, no gold-QA work, no Task 9, and no commit/push were performed.**
Ground truth was not fabricated; no valid quantitative benchmark was established.