# PHASE 3 TASK 10F - VALIDATION

Generated: 2026-09-22T01:52:38.941949+00:00

Checks: **32/32 PASS**, 0 FAIL

| Check | Description | Result |
|---|---|---|
| T1 | v1.0 benchmark unchanged (28ef54fa...) | PASS |
| T2_chunks_jsonl | frozen chunks.jsonl unchanged | PASS |
| T2_embeddings_npy | frozen embeddings.npy unchanged | PASS |
| T2_faiss_index_bin | frozen faiss_index.bin unchanged | PASS |
| T2_bm25_pkl | frozen bm25.pkl unchanged | PASS |
| T2_row_mapping_json | frozen row_mapping.json unchanged | PASS |
| T2 | all 5 frozen GeriLit artifacts unchanged | PASS |
| T3_task10d_retrieval_results | Task 10D/10E artifact 'task10d_retrieval_results' unchanged | PASS |
| T3_task10d_summary | Task 10D/10E artifact 'task10d_summary' unchanged | PASS |
| T3_task10e_failure_analysis | Task 10D/10E artifact 'task10e_failure_analysis' unchanged | PASS |
| T3_task10e_report_md | Task 10D/10E artifact 'task10e_report_md' unchanged | PASS |
| T4 | candidate file marked PENDING | PASS |
| T5 | candidate file marked v1.1 candidate | PASS |
| T6 | candidate IDs unique | PASS |
| T7 | candidate IDs deterministic GLG11-C001..N | PASS |
| T8 | required fields present (question, pmcid, topic) | PASS |
| T9 | all PMCIDs valid format | PASS |
| T10 | every gold chunk id exists in chunk index | PASS |
| T11 | at least one gold chunk per candidate | PASS |
| T12 | topic labels valid C01-C10 | PASS |
| T13 | review_status is PENDING_HUMAN_REVIEW | PASS |
| T14 | reconstruction_status valid | PASS |
| T15 | evidence rationale present | PASS |
| T16 | provenance present (original ids, pmid, title, locator) | PASS |
| T17 | no accidental duplicate questions | PASS |
| T18 | diagnostics present per candidate | PASS |
| T19 | every candidate maps to an original v1.0 id | PASS |
| T20 | v1.0 -> candidate mapping is injective | PASS |
| T21 | v1.0 records not in candidates are accounted as excluded | PASS |
| T22 | exclusions recorded with reasons | PASS |
| T23 | candidate set reproducible (reconstruction signature) | PASS |
| T24 | reconstruction confirmed frozen_unchanged | PASS |

* Frozen v1.0 benchmark & 5 GeriLit index artifacts unchanged.
* Task 10D/10E result artifacts unchanged vs Task 10F baseline.
* All candidates are PENDING_HUMAN_REVIEW (no human relevance
  judgment was made by this task).
* No retrieval evaluation (Recall@K / MRR / nDCG) run.