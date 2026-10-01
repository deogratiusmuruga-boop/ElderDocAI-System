# Experiment 3 - Current-Stack State Audit

**Date:** 2026-09-22
**Checkpoint:** `fe20c5ae6351ba2ee3fe3aae8d73d722078b9fbc`
**Repository:** `C:\Users\chosun\Documents\ElderDocAI-System`
**Status:** AUDIT COMPLETE - EXPERIMENT BLOCKED (see section 8)

**Revision (same day):** following review of this audit, the researcher
authorized **Option 1**: build a real (non-synthetic) dynamic care-state
mechanism into the system, then execute Experiment 3 on it. That mechanism
(`scripts/care_state.py`) is now implemented and integrated; the section 8
"BLOCKED" conclusion applies to the pre-change implementation only. See
`experiment3_protocol.md` for the mechanism description and Experiment 3
design.
This document records a read-only audit of how the CURRENT ElderDocAI
implementation represents and uses dynamic care state. No files were modified.

---

## 1. Current care-state representation

The runtime care-state mechanism is:

- **Source file:** `datasets/synthea/elderdocai/processed/adaptive_context.json`
  (9,723 precomputed records), loaded at import time in
  `scripts/rag_chat.py` (`ADAPTIVE_CONTEXT_FILE`).
- **Record shape** (verified by inspection of the actual JSON):
  `patient_id` (e.g. `8855fb38-21b3-1cab-1e78-84154dad9252` - a UUID),
  `window_start`, `window_end`, `year`, `context_status` (ACTIVE /
  INITIAL / DATA_GAP), `care_state`, `transition`, `changed_dimensions`,
  `adaptive_assistance`, `patient_care_state`, `patient_profile`.
- **`care_state`** contains: `state` in
  {`LOW_ACTIVITY`, `MODERATE_ACTIVITY`, `HIGH_ACTIVITY`, `NO_DATA`, `STABLE`},
  `overall_score` (e.g. 0.4496), `has_documented_activity`, `event_summary`,
  `dimensions[]`, `interpretation`.
- **`transition`** contains: `type`, `direction`, `magnitude`, `score_delta`,
  `previous_state`, `current_state`, `supporting_evidence`.
- **`adaptive_assistance`** contains: `mode`, `priority`, `reason_codes`,
  `reasons`.

The representation IS explicit and precomputed; it is NOT computed at
request time from live signals.

## 2. Available state dimensions

`care_state.dimensions` is a list of
`{name, level, score}` entries, e.g.:
- Care Complexity (MODERATE)
- Clinical Activity (MODERATE)
- Condition Burden (MODERATE)
- Encounter Intensity (MODERATE)
- Medication Burden (MODERATE)
- Observation Intensity (MODERATE)
- (further dimensions per record)

Plus the derived overall state, transition fields, and the adaptive
assistance mode/priority.

## 3. How state is supplied to the system

- `api/main.py` `POST /ask` accepts `question`, `user_id`, `user_profile`
  (patient_id, age, location, chronic_conditions, medications,
  preferred_language, speech_speed), `conversation_history`.
- `scripts/carebuddy_service.py::answer_question` passes the profile to
  `scripts/rag_chat.py::generate_answer`.
- `generate_answer` calls `get_adaptive_context(patient_id)` which performs a
  LOOKUP into the in-memory index built from the precomputed
  `adaptive_context.json` (Synthea-derived records).
- The care-state condition therefore changes ONLY by changing `patient_id` to
  a UUID present in the Synthea-derived dataset. There is NO request-time
  care-state field, NO check-in input, NO state computed from live data.

**Important:** the DB (`SQLite carebuddy.db`) contains only
`user_profiles` (id, age, location, chronic_conditions, medications,
preferred_language, speech_speed), `medications`, `appointments`. There are
NO care-state columns and NO care-state / check-in tables.
## 4. How state influences assistance

- `prepare_adaptive_context(context)` (rag_chat.py) extracts `care_state`,
  `transition`, `adaptive_assistance` and returns a structured dict.
- `prepare_assistance_plan(...)` (rag_chat.py) loads the matching assisted plan
  from `assistance_plans.json` (same precomputed dataset).
- Both are injected into `scripts/build_grounded_prompt.py::build_grounded_prompt`
  as ADAPTIVE CONTEXT text sections (care state, transition, assistance mode /
  priority / reasons / strategy / actions / safety).
- **Scope of influence:** the care state changes ONLY the prompt text given to
  the LLM. It does NOT change retrieval, reliability evaluation, the decision
  gate, or evidence handling (retrieval receives only the query).

## 5. Where adaptation occurs in the current pipeline

| Stage | File | Role |
|---|---|---|
| Precomputed state loading | `scripts/rag_chat.py` (module import) | loads `adaptive_context.json`, `assistance_plans.json`, `assistance_decisions.json` |
| State lookup | `scripts/rag_chat.py::get_adaptive_context(patient_id, context_date)` | patient-id -> latest/exact window record |
| Assistance plan lookup | `scripts/rag_chat.py::get_assistance_plan(...)` | (patient_id, window_start, window_end) -> plan |
| Prompt injection | `scripts/build_grounded_prompt.py::build_grounded_prompt` | renders adaptive context + plan into the LLM prompt |
| API exposure | `scripts/carebuddy_service.py::prepare_care_context` | returns `care_context` in the `/ask` response |
| /ask endpoint | `api/main.py` | `question` + `user_profile` -> `answer_question` |

## 6. What current data can legitimately support the experiment

- **Nothing in the current implementation can support a state-contrast
  experiment without using the precomputed Synthea-derived dataset.**
- The only runtime input that changes the care-state condition is
  `user_profile.patient_id` mapping to the 178 UUID patients in
  `datasets/synthea/elderdocai/processed/adaptive_context.json`.
- The task explicitly forbids resurrecting / importing / reusing /
  reproducing historical Synthea-based RQ4/RQ5 experiments and synthetic
  patient data (Experiment 3 CRITICAL note). The precomputed dataset is
  exactly that historical Synthea-derived artifact.
- There is no endpoint, schema field, DB column, or code path that accepts a
  care state from the caller at request time.
## 7. Limitations / missing state signals

- No `/check-in/{user_id}/today` endpoint exists in `api/`.
- No check-in / state / conversation-derived state mechanism.
- No live or externally supplied state input to `/ask`.
- State is precomputed and offline; it cannot be varied per request without:
  (a) reusing the forbidden Synthea-derived records, or
  (b) modifying production code to accept a state input (manufacturing an
  experimental capability), or
  (c) fabricating a state injection path that the current application does
  not have.
- All three options are prohibited by the Experiment 3 constraints
  ("Do NOT resume Synthea", "Do NOT modify the application to manufacture an
  experimental capability", "Do not invent a care-state mechanism", "Do not
  use synthetic patient data").

## 8. Audit conclusion - EXPERIMENT BLOCKED

Per Experiment 3 Section 1:

> "If the current implementation does not contain a sufficiently explicit
> dynamic care-state mechanism, STOP after the audit and report the gap."

The current implementation DOES contain an explicit (precomputed)
representation, but it is **entirely Synthea-derived synthetic data**, and
the request-time API/DB surface exposes **no legitimate care-state input**.

Constructing the required paired state contrast (same question, same
evidence, different state) would necessarily reuse the historical
Synthea-based dataset, which the task explicitly prohibits, or require
production modification, which the task also prohibits.

**Therefore Experiment 3 cannot be executed on the current implementation
under the stated constraints. No experimental outputs were produced.**