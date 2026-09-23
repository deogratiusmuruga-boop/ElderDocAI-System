# PHASE 3 TASK 10H - VALIDATION

Generated: 2026-09-22T04:48:55.617662+00:00

Checks: **29/29 PASS**, 0 FAIL

| Check | Description | Result |
|---|---|---|
| H1 | v1.1 benchmark exists with exact SHA-256 | PASS |
| H2 | v1.1 has 121 records, all ACCEPTED | PASS |
| H3 | no duplicate v1.1 IDs; no missing gold | PASS |
| H4 | v1.0 control unchanged | PASS |
| H5_chunks | frozen chunks unchanged | PASS |
| H5_embeddings | frozen embeddings unchanged | PASS |
| H5_faiss | frozen faiss unchanged | PASS |
| H5_bm25 | frozen bm25 unchanged | PASS |
| H5_row_mapping | frozen row_mapping unchanged | PASS |
| H5 | all 5 frozen GeriLit artifacts unchanged | PASS |
| H6 | configuration K settings match Task 10D | PASS |
| H7 | hybrid weights 0.6/0.4 preserved | PASS |
| H8 | all four configurations present with exact metrics | PASS |
| H9 | ranking lengths respect config K caps | PASS |
| H10 | every v1.1 question evaluated exactly once | PASS |
| H11 | no duplicate evaluation rows | PASS |
| H12 | all gold chunks traceable to the frozen corpus | PASS |
| H13 | all retrieved chunk ids exist in the corpus | PASS |
| H14 | all metrics within [0,1] | PASS |
| H15 | results signature equals summary signature | PASS |
| H16 | canonical signature matches recorded Task 10H runs | PASS |
| H17 | benchmark integrity recorded by evaluation | PASS |
| H18_task10d_results | task10d_results unchanged (Task 10F baseline) | PASS |
| H18_task10d_summary | task10d_summary unchanged (Task 10F baseline) | PASS |
| H18_task10e_analysis | task10e_analysis unchanged (Task 10F baseline) | PASS |
| H19 | Task 10G final benchmark unchanged | PASS |
| H20 | Task 10G review artifact unchanged | PASS |
| H21 | Task 10E diagnostics still cover 134 queries | PASS |
| H22 | v1.0 methodological equivalence rerun matches Task 10D | PASS |

* Evaluation-only task: frozen retriever/indexes/models used;
  no retrieval code, weights, benchmark, or production file
  changed; no model/package downloads.
* Task 10D/10E/10F/10G artifacts unchanged.