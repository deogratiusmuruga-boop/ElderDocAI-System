# ElderDocAI-GeriLit — Corpus Specification

## Identity
- **Corpus name:** ElderDocAI-GeriLit
- **Stage:** DEVELOPMENT corpus (this task) — `ElderDocAI-GeriLit-dev-v0.1`
- **Final corpus:** a separate immutable version after curation is finalized (NOT this dev corpus).
- **Source:** PMC Open Access Subset (official NCBI resources only).

## Purpose
Provide real, de-identified, peer-reviewed biomedical full-text evidence for ElderDocAI's
RAG/evidence layer, centered on older-adult care and aging-related assistance. Development
corpus is used for pipeline methodology validation, NOT as final experimental evidence.

## Primary topic domains (9; see `topic_taxonomy.json`)
1. Memory and cognition
2. Medication / medication safety / polypharmacy
3. Exercise / physical activity / mobility / function
4. Nutrition / diet / healthy aging
5. Caregiving / caregiver support
6. Geriatric syndromes
7. Chronic conditions relevant to older-adult care
8. Functional health / healthy aging
9. Preventive and supportive care relevant to older adults

The corpus must remain centered on older-adult care; it must NOT become a generic biomedical corpus.

## Access method (verified)
- **PMC OA Subset + web services** (official NCBI/NLM):
- `esearch` (db=pmc) with `open access[filter]` for OA-Subset discovery;
- `esummary` (db=pmc) for title/journal/years/ids;
- NCBI ID Converter (`idconv`) for PMCID<->PMID;
- PMC OA Web Service (`oa.fcgi`) for authoritative OA-Subset membership + license + tarball URL;
- Full text obtained from official `oa.fcgi` tarball links (NLM/FTP-over-HTTPS).
- No third-party scraping of articles.

## Target
- Development target: ~500 accepted full-text articles.
- Not fewer if strict criteria demand it (curation-quality over count).
- Topic balance target: >=5% of accepted articles per major topic (curation target, not an override).
- Articles may belong to multiple topics without artificial duplication.

## Key parameters (dev)
- Publication window: 2015-present primary; pre-2015 only as sentinel studies (sentinel=true + reason).
- Language: English (or UNKNOWN if not determinable); non-English excluded.
- License preference: PERMISSIVE (CC BY / CC0 / public-domain-equivalent); RESTRICTED/UNKNOWN excluded from a redistributable dev artifact.
- Evidence-type preference order (curation preference, NOT a clinical evidence ranking): guidelines/consensus; systematic reviews; meta-analyses; RCTs; observational studies; evidence syntheses; narrative reviews; other peer-reviewed research.
- Benchmark protection: PubMedQA / TREC-CDS / BioASQ protected-document stoplists via `protected_documents.json` (status PENDING_BENCHMARK_ACQUISITION; no invented IDs).

## Guardrails
- No clinical interpretation; corpus is evidence for information retrieval/Grounding engineering only.
- DOCUMENTED_ACTIVITY_ONLY / NO_DIAGNOSIS / NO_MEDICAL_RISK_PREDICTION / NO_DISEASE_PROGRESSION_INFERENCE apply at the system level.