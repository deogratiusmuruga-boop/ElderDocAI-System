# Experiment 1 - Validation Report

**Passed:** 63/63

| # | Check | Passed | Detail |
|---|-------|--------|--------|
| 1 | benchmark_sha256_matches | True | 1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72 |
| 2 | benchmark_121_records | True |  |
| 3 | benchmark_all_accepted | True |  |
| 4 | benchmark_ids_unique | True |  |
| 5 | run1_121_records | True | n=121 |
| 6 | run1_ids_unique | True |  |
| 7 | run1_all_benchmark_ids_represented_exactly_once | True |  |
| 8 | field_present_question_id | True | missing in [] |
| 9 | field_present_topic | True | missing in [] |
| 10 | field_present_question | True | missing in [] |
| 11 | field_present_gold_pmcid | True | missing in [] |
| 12 | field_present_gold_chunk_ids | True | missing in [] |
| 13 | field_present_retrieved_evidence_ids | True | missing in [] |
| 14 | field_present_retrieved_evidence_texts | True | missing in [] |
| 15 | field_present_retrieval_backend | True | missing in [] |
| 16 | field_present_evidence_count | True | missing in [] |
| 17 | field_present_initial_reliability | True | missing in [] |
| 18 | field_present_initial_decision | True | missing in [] |
| 19 | field_present_reliability | True | missing in [] |
| 20 | field_present_decision | True | missing in [] |
| 21 | field_present_retrieval_attempts | True | missing in [] |
| 22 | field_present_refinement_attempts | True | missing in [] |
| 23 | field_present_refused | True | missing in [] |
| 24 | field_present_generated | True | missing in [] |
| 25 | field_present_answer | True | missing in [] |
| 26 | field_present_latency_seconds | True | missing in [] |
| 27 | field_present_faithfulness | True | missing in [] |
| 28 | field_present_answer_relevance | True | missing in [] |
| 29 | field_present_evidence_support | True | missing in [] |
| 30 | field_present_hallucination | True | missing in [] |
| 31 | field_present_error | True | missing in [] |
| 32 | gold_chunk_ids_consistent_with_benchmark | True | mismatch [] |
| 33 | aggregate_statistics_recompute | True | mismatches: [] |
| 34 | decision_counts_recompute | True |  |
| 35 | frozen_benchmark_v1_0_present | True | 28ef54faeec3b9a1cb8c1c79c9a478d8ea3618574b2787a82a74c1460f209c62 |
| 36 | frozen_benchmark_v1_1_present | True | 1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72 |
| 37 | frozen_chunks_present | True | 62bfdde3c7e77cf56112e79a71bc79bdea70767f6b2625701e392bbb6581c6b3 |
| 38 | frozen_embeddings_present | True | b0905d2f96903dadc3cf5b4a737f330853656f846955de706f21bd714ce2dacf |
| 39 | frozen_faiss_present | True | b7016b9dabbab08ee68f875007b6db2b013af4472b1a3c0958af8a566b59f2b3 |
| 40 | frozen_bm25_present | True | 17f503240f2f9a13abf58c94359884b03b06bec1a5a59f63f921328e2a2c70c9 |
| 41 | frozen_row_mapping_present | True | 4d649f81d554c41ec7b63c65dce887466dfa6245745c568315735f1b93a1872b |
| 42 | frozen_reliability_config_present | True | e1c94d7b727071c6566ac7129c2e5d48ceb1c124cab85030f4babff38b7e0d02 |
| 43 | frozen_eval_task10d_results_present | True | 503b199ef8917d33df2eb30fadd4dbd47cfa3ae4c7549dc932710f92a0ddff86 |
| 44 | frozen_eval_task10d_summary_present | True | e81230ec4a9502d717177633658842e8c15c292b0dab1cff69b9d0e5758972f4 |
| 45 | frozen_eval_task10e_analysis_present | True | 3a262687fcd775b385ed7d2959a8128db85828b9d095ad4edacef04e0155fe9b |
| 46 | frozen_eval_task3_summary_present | True | 13917f35492ef29eccbe1a1f15fe779c75b54aeaeb50f88259c82410f00e2ab8 |
| 47 | production_generation_options_frozen | True |  |
| 48 | production_llm_frozen | True |  |
| 49 | gate_budgets_frozen | True |  |
| 50 | run2_121_records | True | n=121 |
| 51 | run2_ids_unique | True |  |
| 52 | repro_retrieved_evidence_ids_identical | True | 121/121 match |
| 53 | repro_reliability_identical | True | 121/121 match |
| 54 | repro_decision_identical | True | 121/121 match |
| 55 | repro_refinement_attempts_identical | True | 121/121 match |
| 56 | repro_retrieval_attempts_identical | True | 121/121 match |
| 57 | repro_refused_identical | True | 121/121 match |
| 58 | repro_generated_identical | True | 121/121 match |
| 59 | repro_answer_identical | True | 121/121 match |
| 60 | repro_faithfulness_identical | True | 121/121 match |
| 61 | repro_answer_relevance_identical | True | 121/121 match |
| 62 | repro_evidence_support_identical | True | 121/121 match |
| 63 | repro_hallucination_identical | True | 121/121 match |