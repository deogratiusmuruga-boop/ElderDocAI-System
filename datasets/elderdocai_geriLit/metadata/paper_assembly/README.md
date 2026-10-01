# ElderDocAI Phase 3 — Paper Assembly Status

Root: `datasets/elderdocai_geriLit/metadata/paper_assembly/`

**All experiments remain frozen and read-only. Nothing in this folder regenerates
or modifies experiment artifacts; every script reads frozen JSON and writes only
under this assembly.**

## Status summary (this session)

| Task | Status | Location |
|------|--------|----------|
| A2 — LLM judge run-to-run reliability | ✅ DONE | `a2_judge_reliability/` |
| A3 — Confidence intervals | ✅ DONE | `a3_confidence_intervals/` |
| A1 — Human calibration package | ✅ PACKAGE READY, **BLOCKED on researcher labels** | `a1_human_calibration/` |
| C7/C8 — Narrative tables + 6 figures | ✅ DONE | `c_manuscript/` |
| B4/B5 — Failure diagnostics + examples | ✅ DONE | `b4_v11_failure_diagnostics/` |
| E — Journal manuscript | ✅ DRAFT ASSEMBLED | `c_manuscript/PHASE3_GERILIT_MANUSCRIPT.md` |

## Headline frozen results (verbatim)

- Exp 1: faith 0.7541 (CI [0.6835, 0.8248]), relv 0.6715, support 0.8528;
  unsupported 23.1% (CI [16.0, 31.7]), contradiction 18.2%.
- Exp 2: no statistic-supported gate effect (faith t p = 0.0685).
- Exp 3: state-response 35% (7/20, CI [15.4, 59.2]); appropriateness 0/20;
  evidence-identical 100%.
- Exp 4: FULL > VANILLA on all 3 metrics, all deltas supported (t p ≤ 0.0025,
  Wilcoxon p ≤ 0.0028); evidence identity held 121/121.
- Judge reliability: 90.0–100% exact match across runs; kappa 0.82–1.0
  (Exp 3 appropriateness kappa = 0.0 documented as kappa paradox).

## How to reproduce each deliverable

```bash
# A2 (reads frozen per-question run1/run2 of Exps 1-4)
python a2_judge_reliability/a2_judge_reliability.py

# A3 (reads frozen summaries + scipy)
python a3_confidence_intervals/a3_confidence_intervals.py

# A1 (deterministic selection + blocked agreement script)
python a1_human_calibration/a1_select_sample.py
python a1_human_calibration/a1_compute_agreement.py   # reports BLOCKED until labels

# C7/C8 (narrative tables + figure data, then figures)
python c_manuscript/c7_c8_narrative_tables.py
python c_manuscript/c7_c8_figures.py

# B4/B5 (failure diagnostics)
python b4_v11_failure_diagnostics/b4_v11_failure_diagnostics.py
```

## Researcher action required (unblocks A1 → manuscript §7)

1. Open `a1_human_calibration/a1_annotation_sheet.csv` (20 rows).
2. Read `a1_human_calibration/a1_annotation_instructions.md` (rubric).
3. Fill the four `human_*` columns per row; save as
   `a1_human_calibration/a1_human_labels.csv`.
4. Run `a1_compute_agreement.py` → writes `a1_human_judge_agreement.{json,md}`.
5. Manuscript §7 then receives the agreement statistics.

## Differences vs. repo-root `SCI_PAPER_DRAFT.md`

The repo-root scaffold is the methods-focused draft centered on the
Synthea-derived deterministic layer (178 patients, 9,723 windows).
`c_manuscript/PHASE3_GERILIT_MANUSCRIPT.md` is the standalone final-stack
Results manuscript (GeriLit-Gold v1.1, frozen experiments). The scaffold has
not been modified.