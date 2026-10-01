# Experiment 3 - Validation Report

**Passed:** 90/90

| # | Check | Passed | Detail |
|---|-------|--------|--------|
| 1 | benchmark_sha256_matches | True | 1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72 |
| 2 | benchmark_121_records | True |  |
| 3 | benchmark_ids_unique | True |  |
| 4 | current_state_audit_exists | True |  |
| 5 | protocol_exists | True |  |
| 6 | care_state_module_exists | True |  |
| 7 | no_synthea_import_in_estimator | True |  |
| 8 | experiment_no_synthea_data | True |  |
| 9 | generation_options_frozen | True |  |
| 10 | weights_frozen | True |  |
| 11 | thresholds_frozen | True |  |
| 12 | run1_20_pairs | True | n=20 |
| 13 | pair_ids_unique | True |  |
| 14 | pairs_from_benchmark | True |  |
| 15 | field_present_pair_id | True |  |
| 16 | field_present_question_id | True |  |
| 17 | field_present_topic | True |  |
| 18 | field_present_question | True |  |
| 19 | field_present_gold_pmcid | True |  |
| 20 | field_present_gold_chunk_ids | True |  |
| 21 | field_present_state_A | True |  |
| 22 | field_present_state_B | True |  |
| 23 | field_present_state_difference | True |  |
| 24 | field_present_score_A | True |  |
| 25 | field_present_score_B | True |  |
| 26 | field_present_transition_type | True |  |
| 27 | field_present_transition_direction | True |  |
| 28 | field_present_condition_A | True |  |
| 29 | field_present_condition_B | True |  |
| 30 | field_present_adaptation_judgment | True |  |
| 31 | field_present_appropriateness_judgment | True |  |
| 32 | field_present_paired_differences | True |  |
| 33 | condition_A_field_evidence_ids | True |  |
| 34 | condition_A_field_reliability | True |  |
| 35 | condition_A_field_decision | True |  |
| 36 | condition_A_field_refinement_attempts | True |  |
| 37 | condition_A_field_retrieval_attempts | True |  |
| 38 | condition_A_field_refused | True |  |
| 39 | condition_A_field_generated | True |  |
| 40 | condition_A_field_answer | True |  |
| 41 | condition_A_field_faithfulness | True |  |
| 42 | condition_A_field_answer_relevance | True |  |
| 43 | condition_A_field_evidence_support | True |  |
| 44 | condition_A_field_gold_chunk_retrieved | True |  |
| 45 | condition_A_field_error | True |  |
| 46 | condition_B_field_evidence_ids | True |  |
| 47 | condition_B_field_reliability | True |  |
| 48 | condition_B_field_decision | True |  |
| 49 | condition_B_field_refinement_attempts | True |  |
| 50 | condition_B_field_retrieval_attempts | True |  |
| 51 | condition_B_field_refused | True |  |
| 52 | condition_B_field_generated | True |  |
| 53 | condition_B_field_answer | True |  |
| 54 | condition_B_field_faithfulness | True |  |
| 55 | condition_B_field_answer_relevance | True |  |
| 56 | condition_B_field_evidence_support | True |  |
| 57 | condition_B_field_gold_chunk_retrieved | True |  |
| 58 | condition_B_field_error | True |  |
| 59 | states_differ_all_pairs | True |  |
| 60 | evidence_identical_across_conditions | True |  |
| 61 | transition_is_escalation | True |  |
| 62 | delta_recompute_faithfulness_delta | True | reported=0.0375 rec=0.0375 |
| 63 | delta_recompute_answer_relevance_delta | True | reported=0.05 rec=0.05 |
| 64 | delta_recompute_evidence_support_delta | True | reported=-0.00278 rec=-0.00278 |
| 65 | frozen_benchmark_v1_0_present | True | 28ef54faeec3b9a1cb8c1c79c9a478d8ea3618574b2787a82a74c1460f209c62 |
| 66 | frozen_benchmark_v1_1_present | True | 1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72 |
| 67 | frozen_chunks_present | True | 62bfdde3c7e77cf56112e79a71bc79bdea70767f6b2625701e392bbb6581c6b3 |
| 68 | frozen_embeddings_present | True | b0905d2f96903dadc3cf5b4a737f330853656f846955de706f21bd714ce2dacf |
| 69 | frozen_faiss_present | True | b7016b9dabbab08ee68f875007b6db2b013af4472b1a3c0958af8a566b59f2b3 |
| 70 | frozen_bm25_present | True | 17f503240f2f9a13abf58c94359884b03b06bec1a5a59f63f921328e2a2c70c9 |
| 71 | frozen_row_mapping_present | True | 4d649f81d554c41ec7b63c65dce887466dfa6245745c568315735f1b93a1872b |
| 72 | frozen_reliability_config_present | True | e1c94d7b727071c6566ac7129c2e5d48ceb1c124cab85030f4babff38b7e0d02 |
| 73 | frozen_eval_task10d_results_present | True | 503b199ef8917d33df2eb30fadd4dbd47cfa3ae4c7549dc932710f92a0ddff86 |
| 74 | frozen_eval_task10d_summary_present | True | e81230ec4a9502d717177633658842e8c15c292b0dab1cff69b9d0e5758972f4 |
| 75 | frozen_eval_task10e_analysis_present | True | 3a262687fcd775b385ed7d2959a8128db85828b9d095ad4edacef04e0155fe9b |
| 76 | frozen_eval_task3_summary_present | True | 13917f35492ef29eccbe1a1f15fe779c75b54aeaeb50f88259c82410f00e2ab8 |
| 77 | frozen_eval_exp1_per_question_present | True | 838c6d9147eff074a4626ac06bdc3ecf4da59948b731a3badf8b0542f3a01c07 |
| 78 | frozen_eval_exp2_per_question_present | True | 4023cfee215c115e1c358f156947222c886cd02b808b3d2deacd785f167fe730 |
| 79 | run2_20_pairs | True |  |
| 80 | repro_state_inputs_identical | True | 20/20 |
| 81 | repro_questions_identical | True | 20/20 |
| 82 | repro_evidence_A_identical | True | 20/20 |
| 83 | repro_evidence_B_identical | True | 20/20 |
| 84 | repro_reliability_A_identical | True | 20/20 |
| 85 | repro_reliability_B_identical | True | 20/20 |
| 86 | repro_decision_A_identical | True | 20/20 |
| 87 | repro_decision_B_identical | True | 20/20 |
| 88 | repro_answer_A_identical | True | 20/20 |
| 89 | repro_answer_B_identical | True | 20/20 |
| 90 | repro_adaptation_identical | True | 20/20 |