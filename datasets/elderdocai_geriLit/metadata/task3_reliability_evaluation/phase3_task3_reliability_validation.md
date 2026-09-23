# PHASE 3 TASK 3 - RELIABILITY RE-EVALUATION VALIDATION

Generated: 2026-09-23T05:12:21.635110+00:00

Checks: **27/27 PASS**, 0 FAIL

| Check | Description | Result |
|---|---|---|
| T1 | v1.1 benchmark hash correct | PASS |
| T2 | v1.1 status/version | PASS |
| T3 | population = 121 unique questions | PASS |
| T4 | all rows have question/topic/pmcid | PASS |
| T5 | all topics valid C01-C10 | PASS |
| T6 | weights unchanged (0.3/0.3/0.2/0.1/0.1) | PASS |
| T7 | thresholds unchanged (0.80/0.65/0.45) | PASS |
| T8 | reliability within [0,1] | PASS |
| T9 | factors within [0,1] | PASS |
| T10 | decisions consistent with fixed thresholds | PASS |
| T11 | decision distribution sums to 121 | PASS |
| T12 | valid decision labels only | PASS |
| T13 | no empty-evidence cases | PASS |
| T14 | deterministic runs (manifest identical) | PASS |
| T15_v1.0 | frozen v1.0 unchanged | PASS |
| T15_v1.1 | frozen v1.1 unchanged | PASS |
| T15_chunks | frozen chunks unchanged | PASS |
| T15_embeddings | frozen embeddings unchanged | PASS |
| T15_faiss | frozen faiss unchanged | PASS |
| T15_bm25 | frozen bm25 unchanged | PASS |
| T15_row_mapping | frozen row_mapping unchanged | PASS |
| T16_t10d_results | Task 10D/10E artifact unchanged | PASS |
| T16_metadata/task10d_summary.json | Task 10D/10E artifact unchanged | PASS |
| T16_t10e_analysis | Task 10D/10E artifact unchanged | PASS |
| T17_t10f_diag | t10f_diag recorded (run-invariant) | PASS |
| T17_t10g_diag | t10g_diag recorded (run-invariant) | PASS |
| T17_t10h_results | t10h_results recorded (run-invariant) | PASS |