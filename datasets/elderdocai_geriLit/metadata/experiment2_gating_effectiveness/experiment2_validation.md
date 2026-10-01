# Experiment 2 - Validation Report

**Passed:** 89/89

| # | Check | Passed | Detail |
|---|-------|--------|--------|
| 1 | benchmark_sha256_matches | True | 1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72 |
| 2 | benchmark_121_records | True |  |
| 3 | benchmark_all_accepted | True |  |
| 4 | benchmark_ids_unique | True |  |
| 5 | run1_121_records | True | n=121 |
| 6 | run1_ids_unique | True |  |
| 7 | run1_exact_benchmark_id_set | True |  |
| 8 | field_present_question_id | True | missing [] |
| 9 | field_present_topic | True | missing [] |
| 10 | field_present_question | True | missing [] |
| 11 | field_present_gold_pmcid | True | missing [] |
| 12 | field_present_gold_chunk_ids | True | missing [] |
| 13 | field_present_initial_evidence_ids | True | missing [] |
| 14 | field_present_initial_reliability | True | missing [] |
| 15 | field_present_initial_decision | True | missing [] |
| 16 | field_present_gate_on | True | missing [] |
| 17 | field_present_gate_off | True | missing [] |
| 18 | field_present_paired_differences | True | missing [] |
| 19 | gate_on_field_answer | True |  |
| 20 | gate_on_field_decision | True |  |
| 21 | gate_on_field_reliability | True |  |
| 22 | gate_on_field_evidence_ids | True |  |
| 23 | gate_on_field_refinement_attempts | True |  |
| 24 | gate_on_field_retrieval_attempts | True |  |
| 25 | gate_on_field_refused | True |  |
| 26 | gate_on_field_generated | True |  |
| 27 | gate_on_field_faithfulness | True |  |
| 28 | gate_on_field_answer_relevance | True |  |
| 29 | gate_on_field_evidence_support | True |  |
| 30 | gate_on_field_hallucination | True |  |
| 31 | gate_on_field_gold_chunk_retrieved | True |  |
| 32 | gate_on_field_latency_seconds | True |  |
| 33 | gate_on_field_error | True |  |
| 34 | gate_off_field_answer | True |  |
| 35 | gate_off_field_decision | True |  |
| 36 | gate_off_field_reliability | True |  |
| 37 | gate_off_field_evidence_ids | True |  |
| 38 | gate_off_field_refused | True |  |
| 39 | gate_off_field_generated | True |  |
| 40 | gate_off_field_faithfulness | True |  |
| 41 | gate_off_field_answer_relevance | True |  |
| 42 | gate_off_field_evidence_support | True |  |
| 43 | gate_off_field_hallucination | True |  |
| 44 | gate_off_field_gold_chunk_retrieved | True |  |
| 45 | gate_off_field_latency_seconds | True |  |
| 46 | gate_off_field_error | True |  |
| 47 | gate_off_uses_shared_initial_evidence | True | mismatch [] |
| 48 | gate_off_no_refinement_no_rere_retrieve | True | mismatch [] |
| 49 | gate_budgets_frozen | True |  |
| 50 | generation_options_frozen | True |  |
| 51 | weights_frozen | True |  |
| 52 | thresholds_frozen | True |  |
| 53 | evaluator_prompts_imported | True |  |
| 54 | deterministic_judge_options | True |  |
| 55 | stat_recompute_faithfulness | True | reported=-0.02686 recomputed=-0.02686 |
| 56 | stat_recompute_answer_relevance | True | reported=0.004132 recomputed=0.004132 |
| 57 | stat_recompute_evidence_support | True | reported=-0.001467 recomputed=-0.001467 |
| 58 | frozen_benchmark_v1_0_present | True | 28ef54faeec3b9a1cb8c1c79c9a478d8ea3618574b2787a82a74c1460f209c62 |
| 59 | frozen_benchmark_v1_1_present | True | 1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72 |
| 60 | frozen_chunks_present | True | 62bfdde3c7e77cf56112e79a71bc79bdea70767f6b2625701e392bbb6581c6b3 |
| 61 | frozen_embeddings_present | True | b0905d2f96903dadc3cf5b4a737f330853656f846955de706f21bd714ce2dacf |
| 62 | frozen_faiss_present | True | b7016b9dabbab08ee68f875007b6db2b013af4472b1a3c0958af8a566b59f2b3 |
| 63 | frozen_bm25_present | True | 17f503240f2f9a13abf58c94359884b03b06bec1a5a59f63f921328e2a2c70c9 |
| 64 | frozen_row_mapping_present | True | 4d649f81d554c41ec7b63c65dce887466dfa6245745c568315735f1b93a1872b |
| 65 | frozen_reliability_config_present | True | e1c94d7b727071c6566ac7129c2e5d48ceb1c124cab85030f4babff38b7e0d02 |
| 66 | frozen_eval_task10d_results_present | True | 503b199ef8917d33df2eb30fadd4dbd47cfa3ae4c7549dc932710f92a0ddff86 |
| 67 | frozen_eval_task10d_summary_present | True | e81230ec4a9502d717177633658842e8c15c292b0dab1cff69b9d0e5758972f4 |
| 68 | frozen_eval_task10e_analysis_present | True | 3a262687fcd775b385ed7d2959a8128db85828b9d095ad4edacef04e0155fe9b |
| 69 | frozen_eval_task3_summary_present | True | 13917f35492ef29eccbe1a1f15fe779c75b54aeaeb50f88259c82410f00e2ab8 |
| 70 | frozen_eval_exp1_per_question_present | True | 838c6d9147eff074a4626ac06bdc3ecf4da59948b731a3badf8b0542f3a01c07 |
| 71 | run2_121_records | True | n=121 |
| 72 | repro_initial_evidence_ids_identical | True | 121/121 |
| 73 | repro_initial_decision_identical | True | 121/121 |
| 74 | repro_on_decision_identical | True | 121/121 |
| 75 | repro_on_refinement_identical | True | 121/121 |
| 76 | repro_on_retrieval_identical | True | 121/121 |
| 77 | repro_on_refused_identical | True | 121/121 |
| 78 | repro_on_evidence_ids_identical | True | 121/121 |
| 79 | repro_off_evidence_ids_identical | True | 121/121 |
| 80 | repro_on_answer_identical | True | 121/121 |
| 81 | repro_off_answer_identical | True | 121/121 |
| 82 | repro_on_faithfulness_identical | True | 121/121 |
| 83 | repro_off_faithfulness_identical | True | 121/121 |
| 84 | repro_on_support_identical | True | 121/121 |
| 85 | repro_off_support_identical | True | 121/121 |
| 86 | repro_off_refused_identical | True | 121/121 |
| 87 | judge_relevance_answers_identical_on_mismatches | True | on_bad=1 off_bad=2 (0.25-step borderline judge scores on identical generated answers) |
| 88 | judge_relevance_mismatch_scope_small | True | on_bad=1 off_bad=2 |
| 89 | system_fields_reproducible | True |  |