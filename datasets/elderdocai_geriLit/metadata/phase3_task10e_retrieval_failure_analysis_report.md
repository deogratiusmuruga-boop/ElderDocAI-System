# PHASE 3 TASK 10E — RETRIEVAL FAILURE & BENCHMARK VALIDITY DIAGNOSTIC

**Status: COMPLETE (PASS)** — read-only diagnostic; no benchmark, corpus, index,
retrieval, or production changes. 21/21 validation checks; deterministic across runs.

---

## 1. Objective

Task 10D measured near-floor retrieval metrics on the frozen GeriLit-Gold
benchmark (mean Recall@5 ≈ 0.02–0.04; only 8/134 questions retrieved their gold
chunk within top-5 by any configuration). This task diagnoses *why*, separating
benchmark-construction effects from retriever behavior — without optimizing either.

## 2. Inputs (read-only)

- `data/geri_lit_gold.json` (frozen, 134 human-verified queries, 46 PMCIDs, 131 gold chunks; SHA-256 `28ef54fa...`)
- `datasets/elderdocai_geriLit/metadata/task10d_retrieval_results.json` (frozen per-query rankings; SHA-256 `503b199e...`)
- `datasets/elderdocai_geriLit/chunks/chunks.jsonl` (17,930 chunks; SHA-256 `62bfdde3...`)
- `datasets/elderdocai_geriLit/geri_lit_retriever.py` (frozen; used only for the semantic diagnostic embedding after retrieval results were already recorded)

## 3. Retrieval failure distribution (per configuration, of 134 queries)

| Config | Top-1 | Top-3 | Top-5 | No-top-5 |
|---|---:|---:|---:|---:|
| Dense | 2 | 2 | 5 | 129 |
| BM25 (sparse) | 1 | 1 | 3 | 131 |
| Hybrid | 1 | 2 | 4 | 130 |
| Hybrid + CrossEncoder | 2 | 4 | 4 | 130 |
| **Any configuration** | — | — | **8** | **126** |

Failure categories (top-5): **A** (gold retrieved by ≥3 configs) = 4 · **B** (1–2 configs) = 4 · **C** (none) = **126 (94.0%)**.

## 4. Diagnostic findings

### 4.1 Lexical alignment (deterministic, stopword-filtered content tokens)

| Group | n | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| Retrieved (top-5, any config) | 8 | **0.6125** | 0.6250 | 0.25 | 1.00 |
| Not retrieved | 126 | **0.3976** | 0.3750 | 0.00 | 1.00 |

Overall lexical-overlap distribution (134 queries): [0,0.1) 5 · [0.1,0.3) 34 ·
[0.3,0.5) 42 · [0.5,0.7) 43 · [0.7,1.0] 10. Only **10/134 (7.5%)** reach
overlap ≥0.7. Questions are very short (mean ≈ 3.5–4.7 content tokens/topic).

### 4.2 Semantic alignment (frozen BGE query-embedding dot product; diagnostic only)

| Group | n | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| Retrieved (top-5, any config) | 8 | **0.7534** | 0.7486 | 0.7005 | 0.8055 |
| Not retrieved | 126 | **0.6548** | 0.6558 | 0.4254 | 0.8400 |

Bins (134): [0,0.5) 3 · [0.5,0.6) 25 · [0.6,0.7) 60 · [0.7,0.8) 44 · [0.8,1) 2.
Only 2/134 reach ≥0.8. Retrieved vs not-retrieved semantic means are
closer (Δ≈0.10) than their lexical gap (Δ≈0.21), consistent with
terminology-mismatch and template-query effects rather than pure embedding failure.

### 4.3 Representative examples (benchmark order; gold text inspected)

**Group 1 — gold at top-1:** GLG-019 (C02: MCI prevalence/factors in nursing homes;
lex 1.00, sem 0.767) · GLG-033 (C03: nutrition & health; lex 1.00, sem 0.743) ·
GLG-058 (C05: mobility; lex 0.75, sem 0.806).

**Group 2 — gold at ranks 2–5:** GLG-075 (C06: heart failure; lex 0.75, sem 0.729) ·
GLG-077 (C06: multimorbidity; **lex 0.00** — query "multimorbidit" vs text
"multimorbidity" — sem 0.794) · GLG-125 (C09: advance care planning; lex 0.40, sem 0.703).

**Group 3 — some configs miss:** GLG-058, GLG-060, GLG-075, GLG-077.

**Group 4 — all four configs fail:** GLG-001 (C01) shows the clearest pattern:
question "What approaches may help manage polypharmacy in older adults?" is
anchored to an abstract whose substantive topic is *C. difficile infection*
(polypharmacy mentioned only as an aggravating factor) — lex 0.25, sem 0.561.

### 4.4 Topic patterns

| Topic | n | R1 | R3 | R5 | none | mean lex | mean q len | mean gold len |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C01 Medication & Polypharmacy | 14 | 0 | 0 | 0 | 14 | 0.274 | 4.29 | 92 |
| C02 Dementia & Cognitive | 17 | 1 | 1 | 1 | 16 | 0.474 | 4.24 | 110 |
| C03 Nutrition | 13 | 1 | 1 | 1 | 12 | 0.440 | 4.08 | 97 |
| C04 Physical Activity | 13 | 0 | 0 | 0 | 13 | 0.427 | 4.15 | 91 |
| C05 Falls & Mobility | 15 | 2 | 2 | 2 | 13 | 0.420 | 3.87 | 86 |
| C06 Chronic Disease | 20 | 0 | 0 | 2 | 18 | 0.336 | 3.55 | 94 |
### 4.5 Corpus / chunk-level patterns

- **Gold chunk regions:** ABSTRACT 41 (31%) · BODY 93 (69%). Content types:
  PROSE 130 · TABLE 4. Evidence types: OBSERVATIONAL 43 · UNKNOWN 35 · RCT 17 ·
  NARRATIVE_REVIEW 15 · META_ANALYSIS 9 · QUALITATIVE 7 · SYSTEMATIC_REVIEW 6 ·
  EDITORIAL 2.
- **Descriptive section titles:** present on 92 gold chunks, absent on 42
  (of which 41 are abstracts — abstracts have no section-title discriminator).
- **Same-article competition:** gold chunks have a median of **15** other chunks
  in the *same article* that share at least one content token with the query
  (min 0, max 61). Intro/background/abstract chunks are heavily competed by
  body chunks that repeat the same terminology.
- **Section profile of gold chunks:** abstract 41, introduction 23, "1.
  introduction" 10, results 9, background 6, discussion 5, plus scattered
  methods/statistical sections.

## 5. Root-cause assessment (confirmed / plausible / unsupported)

### Confirmed by diagnostics
1. **Question template brevity & genericness.** Questions average ≈ 3–5 content
   tokens and follow templates ("What factors are associated with X in older
   adults?" / "What is known about X ...?"). Lexical overlap with gold evidence
   is modest (median ≈ 0.38); only 7.5% exceed 0.7 overlap.
2. **Terms absent from the gold chunk (terminology mismatch).** 5/134 have
   zero content-token overlap yet meaningful semantic scores (e.g., GLG-077
   "multimorbidit" vs "multimorbidity").
3. **Moderate semantic alignment only.** Mean question-gold-chunk semantic
   similarity is ≈ 0.66 (overall), converging to ≈ 0.75 for the few retrieved
   items — high enough that retrieval *can* find them, low enough that 94% are
   not within top-5.
4. **Same-article and cross-article competition is high.** Median 15 competing
   chunks in the same article with overlapping vocabulary; templates make many
   questions collide against alternative chunks.
5. **Anchor-vs-answer mismatch is real.** GLG-001 class shows gold chunks whose
   substantive content does not actually answer the posed question
   (polypharmacy *mention* vs "approaches to manage polypharmacy").

### Plausible
6. **Single-gold-chunk evaluation is brittle.** Each query's gold set is one
   chunk; templates produce many near-identical queries whose single gold chunk
   is one of several plausible evidence pieces — retrieval cannot be expected to
   rank one specific chunk top-5 for nearly identical questions.
7. **Abstract gold chunks (31%) lack section-title discriminators,** leaving
   dense/BM25 to rely purely on body-text similarity where body chunks compete.

### Unsupported by diagnostics
8. **"The retriever is broken."** In Task 10C/10D smoke/functional runs the same
   frozen pipeline returns topically sensible evidence; the diagnostics show
   benchmark-side distributional effects dominate.
9. **"All 134 are poor."** Some gold chunks are genuinely well-discriminated
   (Group 1, lex 0.75–1.0, sem ≥ 0.74) and retrieved at top-1 — this is not a
   uniform corpus-wide failure.

## 6. Integrity

- Frozen artifacts unchanged: chunks `62bfdde3...`, embeddings `b0905d2f...`,
  FAISS `b7016b9d...`, BM25 `17f50324...`, row_mapping `4d649f81...`.
- `data/geri_lit_gold.json` unchanged (`28ef54fa...`); Task 10D result file
  unchanged (`503b199e...`).
- Production code untouched; no relevance labels changed; no revisions made.

## 7. Validation

`metadata/validation/phase3_task10e_validation.{json,md}` — **21/21 checks PASS**
(gold & 10D unchanged, 134 questions, no dup IDs, gold present, all 4 configs
represented, categories cover 134, lexical & semantic diagnostics complete,
representative groups present, all 10 topics, benchmark read-only, frozen
artifacts 5/5, offline/no-download, determinism).

## 8. Reproducibility

The diagnostic was executed twice from clean processes. Canonical content
signatures (JSON minus timestamp) were **identical** across runs:
`6b353005e430e7173383af50e0c7801850dbe5863184278f962a4c8d6dcb1782`. Only the
`generated_at_utc` field differs between files.

## 9. Files created

- `metadata/task10e_failure_analysis.py`
- `metadata/task10e_failure_analysis.json`
- `metadata/task10e_validate.py`
- `metadata/validation/phase3_task10e_validation.json`
- `metadata/validation/phase3_task10e_validation.md`
- `metadata/phase3_task10e_retrieval_failure_analysis_report.md` (this report)

## 10. Files modified

**None.** No benchmark, corpus, index, retriever, router, adapter, or production
file was changed. (Temporary report-data snippets were created and removed;
`__pycache__` cleaned.)

## 11. Findings vs revision policy

The most probable explanation supported by diagnostics is a **combination of
template-brevity, terminology mismatch, single-gold-chunk brittleness, and
same-article competition** — i.e., primarily a benchmark-construction and
evaluation-design effect, not evidence that the retrieval pipeline is defective.
No benchmark item was revised, removed, or relabeled. Any future v1.1 benchmark
revision (or retrieval re-tuning) is a separate, explicitly authorized task.

## 12. Git status

Branch `feature/mimic-pmc-migration` · HEAD `8b0696c` (unchanged) · modified:
`.gitignore`, `scripts/rag_chat.py` (pre-existing) · untracked new: Task 10E
artifacts · untracked pre-existing: GeriLit dir, mimic_demo, candidate/review/
gold JSON, router/adapter/tests. **Nothing staged, committed, or pushed.**
| C07 Mental & Social | 16 | 0 | 0 | 0 | 16 | 0.457 | 4.50 | 209 |
| C08 Caregiving | 9 | 0 | 0 | 0 | 9 | 0.469 | 4.67 | 97 |
| C09 Preventive Care | 11 | 0 | 1 | 1 | 10 | 0.391 | 3.91 | 106 |
| C10 Healthy Ageing | 6 | 1 | 1 | 1 | 5 | 0.500 | 4.00 | 88 |

All 10 topics covered. 5 topics have zero top-5 hits under any configuration.