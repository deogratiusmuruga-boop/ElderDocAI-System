# ElderDocAI Final RQ Evidence Matrix

> Planning / documentation artifact. Only experimentally verified information is included. No results are invented; no results were modified by this document.

| RQ | Research question / purpose | Experiment | Dataset | Primary metrics | Current result | Evidence artifact | Status |
|---|---|---|---|---|---|---|---|
| RQ1 | Retrieval / evidence grounding | Primary Gold-QA evaluation | **Gold96** (`data/gold_qa_extended.json`) | Source retrieval accuracy; evidence/chunk recall; supporting-span support; generation failure rate | Source acc **96/96 = 100%** · chunk recall **96/96 = 100%** · span support **96/96 = 100%** · gen failures **0** | `data/evaluation_results/gold_qa_extended_results.json` | COMPLETE |
| RQ2 | Answer quality / reliability | Primary Gold-QA evaluation | **Gold96** | Reliability mean; relevance mean; faithfulness mean; ACCEPT/REFINE; generation failures | Rel **0.9059** · relv **0.9401** · faith **0.9193** · ACCEPT **88/96 = 91.7%** · REFINE **8/96 = 8.3%** · gen failures **0** | `data/evaluation_results/gold_qa_extended_results.json` | COMPLETE |
| RQ2 (comparison) | 16 vs 96 descriptive comparison | Matched Gold-QA evaluation | **Gold16** (`data/gold_qa_evaluation.json`) | Reliability; relevance; faithfulness; ACCEPT/REFINE; span support | Rel **0.9394** · relv **0.9062** · faith **0.9219** · ACCEPT **16/16** · REFINE **0** · span support **14/16 = 87.5%** | `data/evaluation_results/gold_qa_original16_results.json` | COMPLETE |
| RQ3 | Ablation (component contribution) | Ablation study, 7 conditions (A0–A6) | **Gold96** | Per-condition source accuracy, Recall@3, faithfulness, relevance, reliability | A0 100/1.0/0.9089/0.8974/0.9059 · A1 100/1.0/0.9141/0.8958/0.9059 · A2 100/1.0/0.9193/0.9115/0.9059 · A3 98.96/0.9167/0.8906/0.9453/0.899 · A4 98.96/0.9062/0.9036/0.8984/0.9025 · A5 100/1.0/0.9349/0.9219/0.9059 · A6 100/1.0/0.9141/0.8958/0.9059 | `data/evaluation_results/ablation_gold96/` (results.json, report.md, figures) | COMPLETE |
| RQ4 | Dynamic care-state behavior | RQ4/RQ5 dynamic care-state evaluation | **Synthea-derived** longitudinal corpus (`datasets/synthea/elderdocai/processed/`), 9,723 windows / 178 patients — **NOT** Gold-QA | Care-state distribution, transitions, escalation/de-escalation, longitudinal persistence, care-state score | Care-state score mean **0.6202** (n=3,367 scored, median 0.5851) · transition rate **46.19%** of transitionable windows · state distribution: NO_DATA 65.37%, LOW 13.07%, MOD 12.93%, HIGH 8.62% | `data/evaluation_results/rq4_rq5_care_state_assistance_results.json` (`rq4`), `/report.md`, figures `rq4_fig*.png` | COMPLETE (distinct dataset) |
| RQ5 | Adaptive assistance behavior | RQ4/RQ5 dynamic care-state evaluation | **Synthea-derived** longitudinal corpus (9,723 windows / 178 patients) — **NOT** Gold-QA | Assistance-mode/priority distribution, state→assistance mapping, rule-consistency | Rule-consistency: mode agreement **9,723/9,723 = 100%**, priority agreement **100%**, 0 mismatches | `data/evaluation_results/rq4_rq5_care_state_assistance_results.json` (`rq5`), `/report.md`, figures `rq5_fig*.png` | COMPLETE (distinct dataset) |
| LLM baseline | External LLM generation quality vs reference | LLM baseline generation, 5 models × FULL/PLAIN | **Gold96** | Faithfulness, relevance, reliability (shared), generation latency, prompt-equivalence | llama3.2 ref FULL 0.9089/0.8974; mistral FULL 0.9323/0.9661 · qwen3 FULL 0.9271/0.9714 · phi3:mini FULL 0.9167/0.9427 · gemma2 FULL 0.8947/0.9342 | `data/evaluation_results/llm_baselines_gold96/` (results.json, report.md, figures) | COMPLETE |

**RQ4/RQ5 dataset distinction (explicit):** RQ4 (dynamic care-state) and RQ5 (adaptive assistance) are evaluated on the **Synthea-derived longitudinal corpus** (9,723 windows, 178 patients), NOT on the Gold-QA benchmark. Their metrics are descriptive statistics over the simulated patient timeline plus an internal rule-consistency check. They therefore do **not** share the Gold-QA retrieval/answer metrics and must be presented as a separate evidence basis from RQ1–RQ3 and the LLM baselines.
## Remaining manuscript gaps

1. **COMPLETE** — RQ1/RQ2 (Gold96 primary, Gold16 matched), RQ3 ablation (Gold96), LLM baselines (Gold96). All have verified JSON + Markdown + figures.
2. **DOCUMENTATION CLEANUP** — The **JSON `data_source` metadata strings** inside `ablation_gold96/ablation_results.json` and `llm_baselines_gold96/llm_baselines_results.json` still say **"16 gold QA questions"** because the experiments ran before the dynamic count override was added to the generators. The values in those JSON files are frozen and must not be edited (per freeze rules). The **Markdown reports** have been corrected to say 96, and the generator code now emits the correct count, so a future regeneration would be accurate.
3. **FIGURE REGENERATION NEEDED** — None. All ablation and baseline figure titles are generic (no N in label) or already dynamic (`{n_questions}`). No stale-16 figure exists.
4. **RESULT VERIFICATION NEEDED** — Gold96 vs Gold16 statistical comparison (bootstrap CIs, Mann–Whitney, effect sizes, power analysis) has been computed read-only but is **not yet written into a manuscript table**; the numbers must be transcribed carefully into the final paper.
5. **MANUSCRIPT WRITING NEEDED** —
   - RQ4/RQ5 section must clearly separate the Synthea-corpus evidence from the Gold-QA evidence (the matrix above records the datasets).
   - LLM baseline interpretation: note the reliability/retrieval are shared across conditions by design (identical evidence), so baseline differences reflect generation (and judge) only.
   - Report the Gold96 ablation's A3/A4 recall@3/source-accuracy degradations as the notable retrieval differences, and A5's small faithfulness/relevance uplift.
   - Report the Gold16 vs Gold96 reliability difference (0.9394 vs 0.9059) as a benchmark-recalibration effect, not a system regression.
   - Include seed/sampling/model/judge config, offline HF_HUB_OFFLINE reproducibility note, and the frozen hashes for all datasets/results.

## Integrity

- All frozen artifacts untouched (see Task 5 verification).
- Legacy Gold16 `ablation/` and `llm_baselines/` results untouched.
- Nothing committed or pushed.