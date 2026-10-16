# ElderDocAI-System

An **evidence-grounded, rule-bounded** RAG (Retrieval-Augmented Generation) assistant for elderly care. The system retrieves from a caregiver knowledge base, grounds LLM answers with verifiable evidence, gates output through a reliability/adaptive-decision layer, and is served to users through a React frontend.

This repository also contains the **GeriLit-Gold v1.1** benchmark and the frozen Phase-3 evaluation artifacts (Experiments 1–4) reported in the paper, plus the full corpus build recipe, human-calibration package, and evaluation code. Release tag `v1.1` pins the frozen benchmark and all experiment artifacts.

---

## 1. Project summary

- **Hybrid retrieval** — dense vectors (FAISS, BGE `BAAI/bge-base-en-v1.5`) + BM25 lexical search + CrossEncoder (`ms-marco-MiniLM-L-6-v2`) reranking → top-3 evidence.
- **Evidence-grounded generation** — LLM answers forced to cite the retrieved SOURCE only (temperature 0).
- **Reliability gating** — accept / refine / re-retrieve / reject policy with configurable scores & thresholds (`config/reliability_config.json`: weights 0.3/0.3/0.2/0.1/0.1; thresholds ACCEPT ≥ 0.80, REFINE ≥ 0.65, RE-RETRIEVE ≥ 0.45; budgets MAX_REFINE = 1, MAX_RETRIEVE = 1).
- **Adaptive care context** — maps longitudinal records into care-state windows, transitions, and adaptive assistance plans (`scripts/care_state.py`, deterministic, non-clinical).
- **Rule-bounded safety** — deterministic no-diagnosis / no-risk-prediction guardrails in generated plans.
- **Frozen evaluation** — four experiments on GeriLit-Gold v1.1 (n = 121 questions, 45 PMCIDs) with two judge runs each, reproducibility manifests, and human calibration.

> **Note:** `SCI_PAPER_DRAFT.md` contains the research-manuscript draft that previously lived here. The standalone final-stack Results manuscript is `datasets/elderdocai_geriLit/metadata/paper_assembly/c_manuscript/PHASE3_GERILIT_MANUSCRIPT.md`. This README is the project-level reference for the codebase.

---

## 2. Folder map

```
data/
  geri_lit_gold.json               GeriLit-Gold v1.0 (superseded)
  geri_lit_gold_v1_1.json          FROZEN GeriLit-Gold v1.1 benchmark (121 questions, SHA-256 recorded below)
  geri_lit_gold_v1_1_candidate.json / _review.json   candidate + human-review files used to build v1.1
  evaluation_results/              prior Gold-QA / ablation / baseline result sets

datasets/elderdocai_geriLit/       CORPUS BUILD RECIPE (ElderDocAI-GeriLit-dev-v0.1)
  discover_candidates.py           PMC OA Subset discovery (esearch/esummary/oa.fcgi)
  select_candidates.py             inclusion/exclusion selection
  download_fulltext.py             full-text download from official oa.fcgi tarballs
  chunk_jats.py                    JATS chunking
  build_bm25_index.py / build_embeddings.py / build_faiss_index.py   index builders
  assemble_validate.py, jats_enrich.py, geri_lit_retriever.py
  chunks/chunks.jsonl              frozen chunk store
  index/                           frozen embeddings.npy, faiss_index.bin, bm25.pkl, row_mapping.json
  manifest/                        candidate / accepted / excluded manifests, download status, selected
  metadata/
    corpus_specification.md, inclusion_exclusion_policy.md, license_policy.md, chunking_specification.md
    articles_metadata.json, corpus_statistics.json, topic_taxonomy.json, audit JSONs
    experiment1_generation_evaluation/    EXP-1 OUTPUTS (2 judge runs + protocol + manifests + reports)
    experiment2_gating_effectiveness/     EXP-2 OUTPUTS
    experiment3_care_state_adaptation/    EXP-3 OUTPUTS
    experiment4_vanilla_baseline/         EXP-4 OUTPUTS
    task3_reliability_evaluation/         reliability-evaluation artifacts
    paper_assembly/                       HUMAN CALIBRATION + paper assembly
      a1_human_calibration/               annotation instructions/sheet, human labels, agreement stats,
                                          matched-evidence calibration, unsupported_28_manual_classification.{csv,md}
      a2_judge_reliability/               run-to-run judge agreement (Exps 1-4)
      a3_confidence_intervals/            interval estimates
      b4_v11_failure_diagnostics/         failure diagnostics
      c_manuscript/                       PHASE3_GERILIT_MANUSCRIPT.md + figures/tables

scripts/                      RAG pipeline + evaluation/build/reliability scripts
  rag_chat.py                 gated generation (GENERATION_SYSTEM_PROMPT), judge prompts via evaluate_gold_qa.py
  build_grounded_prompt.py    grounded prompt builder
  evaluate_*.py               evaluation scripts (faithfulness / relevance / gold-QA / ablation / baselines / RQ4-RQ5)
  retrieval_router.py, hybrid_retrieval.py, reranker.py, reliability_evaluation.py, care_state.py, ...
evaluation/evaluation_queries.json   query set
api/                          FastAPI routers (main, profile, medication, appointment, reminder, deps)
models/                       SQLAlchemy models
config/                       runtime configuration (reliability thresholds/weights)
frontend/                     React UI
tests/                        retrieval / BM25 / API tests
```

**Benchmark** = `data/geri_lit_gold_v1_1.json` — frozen, SHA-256 `1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72` (121/121 human-reviewed and accepted, 45 PMCIDs, 116 gold chunks).

---

## 3. Recorded experiment environment

The frozen GeriLit-Gold v1.1 evaluation (recorded in the Phase-3 reproducibility section and the per-run manifests) used:

- **Python 3.12** (as recorded for the evaluation; the bundled `venv/` in this repo is Python 3.14.6 — see `requirements.txt` note below)
- **Ollama** serving `llama3.2:latest` (3.2B, Q4_K_M), used for **both generation and judging**
- **Sampler** (generation and judge): `temperature 0`, `top_p 0.1`, `top_k 10`
- **Retrieval**: BGE `BAAI/bge-base-en-v1.5` dense top-5 + BM25 top-5, 0.6/0.4 hybrid fusion, CrossEncoder `cross-encoder/ms-marco-MiniLM-L-6-v2` rerank → top-3
- **Libraries (recorded)**: sentence-transformers 5.6, faiss-cpu 1.14, rank-bm25 0.2, scikit-learn 1.9, torch 2.13, fastapi 0.140, uvicorn 0.51, sqlalchemy 2.0, pydantic 2.13

**`requirements.txt`** (repo root) is a `pip freeze` snapshot of the bundled `venv/` (Python 3.14.6) — 54 pinned packages including `ollama==0.6.2` (Python client library), `sentence-transformers==5.6.0`, `faiss-cpu==1.14.3`, `rank-bm25==0.2.2`, `scikit-learn==1.9.0`, `torch==2.13.0`, `fastapi==0.140.0`, `uvicorn==0.51.0`, `SQLAlchemy==2.0.51`, `pydantic==2.13.4`.

---

## 4. Rerun steps — Experiments 1–4 (GeriLit-Gold v1.1)

Prerequisites: Ollama running with `llama3.2:latest` pulled; frozen index present (`datasets/elderdocai_geriLit/index/*`, `chunks/chunks.jsonl`); Python environment with `requirements.txt` installed; the scripts set `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` for offline model use.

Each evaluation script forces `ELDERDOCAI_RETRIEVAL_BACKEND=geri_lit` and the frozen reliability config itself, writes **only** under its own directory, and verifies the SHA-256 of the frozen artifacts before/after every run (any change → STOP). Run each experiment **twice** (`--run-id 1`, `--run-id 2`) and then validate the two runs.

```bash
# Experiment 1 — generation quality (n=121)
python datasets/elderdocai_geriLit/metadata/experiment1_generation_evaluation/experiment1_generation_evaluation.py --run-id 1
python datasets/elderdocai_geriLit/metadata/experiment1_generation_evaluation/experiment1_generation_evaluation.py --run-id 2
python datasets/elderdocai_geriLit/metadata/experiment1_generation_evaluation/experiment1_validate.py
python datasets/elderdocai_geriLit/metadata/experiment1_generation_evaluation/experiment1_finalize.py --run-id 1
python datasets/elderdocai_geriLit/metadata/experiment1_generation_evaluation/experiment1_finalize.py --run-id 2

# Experiment 2 — reliability-gating effectiveness (paired gate ON vs OFF, n=121)
python datasets/elderdocai_geriLit/metadata/experiment2_gating_effectiveness/experiment2_gating_evaluation.py --run-id 1
python datasets/elderdocai_geriLit/metadata/experiment2_gating_effectiveness/experiment2_gating_evaluation.py --run-id 2
python datasets/elderdocai_geriLit/metadata/experiment2_gating_effectiveness/experiment2_validate.py
python datasets/elderdocai_geriLit/metadata/experiment2_gating_effectiveness/experiment2_finalize.py --run-id 1
python datasets/elderdocai_geriLit/metadata/experiment2_gating_effectiveness/experiment2_finalize.py --run-id 2

# Experiment 3 — dynamic care-state and adaptive assistance (paired states, n=20)
python datasets/elderdocai_geriLit/metadata/experiment3_care_state_adaptation/experiment3_evaluation.py --run-id 1
python datasets/elderdocai_geriLit/metadata/experiment3_care_state_adaptation/experiment3_evaluation.py --run-id 2
python datasets/elderdocai_geriLit/metadata/experiment3_care_state_adaptation/experiment3_validate.py
python datasets/elderdocai_geriLit/metadata/experiment3_care_state_adaptation/experiment3_finalize.py --run-id 1
python datasets/elderdocai_geriLit/metadata/experiment3_care_state_adaptation/experiment3_finalize.py --run-id 2

# Experiment 4 — vanilla-RAG baseline (paired FULL vs VANILLA, n=121)
python datasets/elderdocai_geriLit/metadata/experiment4_vanilla_baseline/experiment4_evaluation.py --run-id 1
python datasets/elderdocai_geriLit/metadata/experiment4_vanilla_baseline/experiment4_evaluation.py --run-id 2
python datasets/elderdocai_geriLit/metadata/experiment4_vanilla_baseline/experiment4_validate.py
python datasets/elderdocai_geriLit/metadata/experiment4_vanilla_baseline/experiment4_finalize.py --run-id 1
python datasets/elderdocai_geriLit/metadata/experiment4_vanilla_baseline/experiment4_finalize.py --run-id 2
```

`--limit N` optionally restricts each evaluation to the first N questions for diagnostics (default: all). Frozen per-question outputs are `experiment{N}_per_question_run{1,2}.json`; reproducibility signatures are in `experiment{N}_reproducibility.json`; protocol and integrity rules are in `experiment{N}_protocol.md`.

### Assembly / paper deliverables (read-only, reproduce with)

```bash
cd datasets/elderdocai_geriLit/metadata/paper_assembly
python a2_judge_reliability/a2_judge_reliability.py       # judge run-to-run reliability
python a3_confidence_intervals/a3_confidence_intervals.py # confidence intervals
python a1_human_calibration/a1_select_sample.py           # deterministic calibration sample
python a1_human_calibration/a1_compute_agreement.py       # human-vs-judge agreement
python b4_v11_failure_diagnostics/b4_v11_failure_diagnostics.py
python c_manuscript/c7_c8_narrative_tables.py
python c_manuscript/c7_c8_figures.py
```

---

## 5. License policy (redistribution)

Per `datasets/elderdocai_geriLit/metadata/license_policy.md`, only **permissively licensed** articles are accepted into the redistributable corpus:

- **PERMISSIVE** = Creative Commons **CC BY**, **CC BY-SA**, **CC0/public domain**, or clearly permissive equivalents (**no `nc`/`nd` modifiers**).
- **RESTRICTED** (CC BY-NC / CC BY-ND / CC BY-NC-ND) and **UNKNOWN** (absent/ambiguous license) are **excluded** from the redistributable artifact and recorded in the excluded manifest.
- License string and category are recorded per article (`metadata/articles_metadata.json` and the manifests). The development corpus is 500/500 PERMISSIVE, 100% English (recorded in `corpus_statistics.json`).
- PMC Open Access membership is verified via `oa.fcgi`; full text comes from official NLM tarballs (no third-party scraping).

The GeriLit-Gold benchmark itself and all evaluation artifacts in this repository are original research outputs; only the underlying corpus articles carry the CC terms above.

---

## 6. Quickstart (application)

### Backend

Create/use the venv and install `requirements.txt`:

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt   # Windows
python create_tables.py                        # one-time SQLite table creation
uvicorn api.main:app --reload                  # default port 8000
```

### Frontend

```bash
cd frontend
npm start
```

Point the frontend at the backend via `frontend/.env` (`REACT_APP_API_URL=http://localhost:8000`); defaults fall back to `http://127.0.0.1:8000`.

---

## 7. Tests

Scripts under `tests/`, plus `test_retrieval.py` and `test_bm25.py`; PASS/FAIL-style evaluation scripts run under `scripts/` (`evaluate_retrieval.py`, `evaluate_latency.py`, `evaluate_carebuddy.py`, …). The Phase-3 validators passed at checkpoint `fe20c5a` (recorded in the Phase-3 reproducibility section).

---

## 8. Status

Internal research prototype with frozen, paper-reported evaluation artifacts. Manuscript drafts tracked in `SCI_PAPER_DRAFT.md` and `datasets/elderdocai_geriLit/metadata/paper_assembly/c_manuscript/PHASE3_GERILIT_MANUSCRIPT.md`.

