# ElderDocAI-GeriLit Phase 3 Task 9B - Candidate Question Pool

# TASK STATUS
PASS

# CANDIDATE POOL
- attempted: 4165
- accepted: 134
- rejected: 4031
- rejection reasons: {"no_keyword_in_chunk": 2445, "article_cap": 459, "weak_anchoring": 267, "near_duplicate": 860}

# TOPIC DISTRIBUTION

| Topic | Candidates | Target | Status |
|---|---|---|---|
| C01 | 14 | 14 | OK |
| C02 | 17 | 17 | OK |
| C03 | 13 | 13 | OK |
| C04 | 13 | 13 | OK |
| C05 | 15 | 15 | OK |
| C06 | 20 | 20 | OK |
| C07 | 16 | 16 | OK |
| C08 | 9 | 9 | OK |
| C09 | 11 | 11 | OK |
| C10 | 6 | 6 | OK |

# ARTICLE DIVERSITY
- unique pmcid: 46
- median questions/article: 4
- max questions/article: 4
- most-used pmcids: ['PMC10022263', 'PMC10043750', 'PMC10147734', 'PMC10002240', 'PMC10008549', 'PMC10095830', 'PMC10174559', 'PMC10276510']

# CHUNK DIVERSITY
- unique chunks: 131
- max questions/chunk: 2

# GROUNDING
{"DIRECT": 134, "MULTI_CHUNK_REQUIRED": 0, "WEAK": 0, "REJECT": 0}

# DUPLICATION
- exact duplicates: 0
- near-duplicates rejected: 860

# VALIDATION
- candidate_id_unique: True
- pmcid_exists: True
- pmid_matches: True
- chunk_id_exists: True
- chunk_belongs_to_pmcid: True
- question_nonempty: True
- no_id_leak: True
- question_unique: True
- grounding_direct_or_multi: True
- evidence_rationale_present: True
- topic_present: True
- chunk_substantive: True

# FILES
- data/geri_lit_gold_candidates.json (candidate pool)
- metadata/phase3_task9b_candidate_construction_report.md
- metadata/validation/phase3_task9b_validation.{json,md}
- metadata/task9b_build_candidates.py (this script)

LLM used: NO. Construction is deterministic and evidence-first; questions were instantiated from per-topic frames using evidence focus phrases verified to occur in the selected chunk text. All candidates are CANDIDATE_PENDING_HUMAN_REVIEW; no relevance grades assigned.
