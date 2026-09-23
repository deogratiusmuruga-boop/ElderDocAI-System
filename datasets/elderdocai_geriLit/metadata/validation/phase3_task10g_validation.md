# PHASE 3 TASK 10G - VALIDATION

Generated: 2026-09-22T02:20:29.839864+00:00

Checks: **41/41 PASS**, 0 FAIL

| Check | Description | Result |
|---|---|---|
| G1 | final benchmark size = 121 | PASS |
| G2 | candidate set unchanged at 121 (PENDING on input) | PASS |
| G3 | all 121 final records ACCEPTED | PASS |
| G4 | zero PENDING, zero REJECTED | PASS |
| G5 | human_review block consistent (121/121/0/0) | PASS |
| G6 | review artifact records = 121 ACCEPTED | PASS |
| G7 | final benchmark IDs unique | PASS |
| G8 | candidate IDs unique | PASS |
| G9 | deterministic final IDs GLG11-001..121 | PASS |
| G10 | required fields present (question, pmcid, title, topic) | PASS |
| G11 | PMCID format valid | PASS |
| G12 | gold chunk IDs present | PASS |
| G13 | every gold chunk exists in the frozen corpus | PASS |
| G14 | topic labels valid C01-C10 | PASS |
| G15 | provenance preserved (original ids, pmid, locator, evidence) | PASS |
| G16 | evidence rationale present | PASS |
| G17 | reconstruction status valid | PASS |
| G18 | diagnostics preserved (lex/sem/region/claim) | PASS |
| G19 | no duplicate questions | PASS |
| G20 | every candidate appears exactly once in the final records | PASS |
| G21 | final records map back to candidates | PASS |
| G22 | v1.0 -> final mapping injective & traceable | PASS |
| G23 | question/evidence relationships unchanged from candidates | PASS |
| G24 | final benchmark SHA-256 matches finalization record | PASS |
| G25 | candidate benchmark SHA-256 matches finalization record | PASS |
| G26 | review artifact SHA-256 matches finalization record | PASS |
| G27 | v1.0 benchmark unchanged (28ef54fa...) | PASS |
| G28_chunks | frozen chunks unchanged | PASS |
| G28_embeddings | frozen embeddings unchanged | PASS |
| G28_faiss | frozen faiss unchanged | PASS |
| G28_bm25 | frozen bm25 unchanged | PASS |
| G28_row_mapping | frozen row_mapping unchanged | PASS |
| G28 | all 5 frozen GeriLit artifacts unchanged | PASS |
| G29_t10d_results | Task 10D/10E artifact t10d_results unchanged | PASS |
| G29_t10d_summary | Task 10D/10E artifact t10d_summary unchanged | PASS |
| G29_t10e_analysis | Task 10D/10E artifact t10e_analysis unchanged | PASS |
| G29_t10e_report | Task 10D/10E artifact t10e_report unchanged | PASS |
| G29 | Task 10D/10E artifacts unchanged | PASS |
| G30 | Task 10F products unchanged during finalize | PASS |
| G31 | final benchmark deterministic signature matches | PASS |
| G32 | final benchmark status FROZEN, version 1.1 | PASS |

* Frozen GeriLit-Gold v1.1 benchmark: 121 ACCEPTED / 0 rejected / 0 pending.
* v1.0 benchmark, 5 corpus/index artifacts, Task 10D/10E and
  Task 10F products unchanged.
* No retrieval evaluation performed.