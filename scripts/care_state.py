#!/usr/bin/env python3
"""ElderDocAI - Real Dynamic Care-State Estimator (non-synthetic).

Computes a care-state record from REAL, CURRENT system inputs only:

    - user profile          (age, chronic conditions, medications)
    - active medications    (SQLite ``medications`` rows)
    - upcoming appointments (SQLite ``appointments`` rows)
    - conversation history  (per-request interaction signal)

Design rules
------------
* NO synthetic / Synthea data is used anywhere in this module.
* PURE and DETERMINISTIC: identical inputs always yield identical outputs.
  No randomness, no wall-clock dependence other than the ``now`` parameter
  (which callers must pass explicitly for reproducibility).
* The returned adaptive-context record matches the shape consumed by
  ``prepare_adaptive_context`` / ``prepare_assistance_plan`` /
  ``build_grounded_prompt`` so the existing production rendering chain can
  consume it without modification.
* All thresholds and weights are documented constants below so the decision
  rule is fully transparent and testable.

Inputs
------
profile (dict or object, optional)   : age, chronic_conditions, medications
medications (list[dict], optional)   : DB rows for the user
appointments (list[dict], optional)  : DB rows for the user
conversation_history (list, optional): previous conversation messages
previous_state (str, optional)       : previously computed state, for
                                       transition responsiveness
now (str, optional)                  : ISO date (YYYY-MM-DD) used only to
                                       window upcoming appointments

Output
------
A dict with keys consumed downstream:
    patient_id, window_start, window_end, context_status, care_state,
    transition, changed_dimensions, adaptive_assistance,
    assistance_plan_record
"""
from __future__ import annotations

import datetime
import statistics
from typing import Any, Dict, List, Optional
# ---------------------------------------------------------------- constants

# State labels (alphabet consistent with the runtime state vocabulary).
NO_DATA = "NO_DATA"
STABLE = "STABLE"
LOW_ACTIVITY = "LOW_ACTIVITY"
MODERATE_ACTIVITY = "MODERATE_ACTIVITY"
HIGH_ACTIVITY = "HIGH_ACTIVITY"

# Dimension level labels.
LEVEL_LOW = "LOW"
LEVEL_MODERATE = "MODERATE"
LEVEL_HIGH = "HIGH"

# Dimension weights (sum = 1.0).
WEIGHTS = {
    "Medication Burden": 0.25,
    "Condition Burden": 0.25,
    "Encounter Intensity": 0.20,
    "Care Complexity": 0.20,
    "Interaction Signal": 0.10,
}

# State score bands (documented thresholds, tested in test_care_state.py).
STABLE_MAX_SCORE = 0.35      # STABLE requires all-dimension LOW as well
LOW_MAX_SCORE = 0.40
MODERATE_MAX_SCORE = 0.60
HIGH_MIN_SCORE = 0.60

# Per-dimension thresholds.
MEDS_HIGH = 5                # polypharmacy reference (>=5)
MEDS_MODERATE = 3
COND_HIGH = 5
COND_MODERATE = 3
APPOINT_HIGH = 2             # upcoming within horizon
APPOINT_MODERATE = 1
APPOINT_HORIZON_DAYS = 30
HISTORY_HIGH = 5
HISTORY_MODERATE = 2

# Assistance mode labels (deterministic state -> mode mapping).
ASSISTANCE_BY_STATE = {
    NO_DATA: "DEFAULT_SUPPORT",
    STABLE: "MAINTENANCE_REINFORCEMENT",
    LOW_ACTIVITY: "ACTIVITY_MAINTENANCE",
    MODERATE_ACTIVITY: "MONITORING_AND_GUIDANCE",
    HIGH_ACTIVITY: "ESCALATED_SUPPORT",
}
PRIORITY_BY_STATE = {
    NO_DATA: "LOW",
    STABLE: "LOW",
    LOW_ACTIVITY: "LOW",
    MODERATE_ACTIVITY: "MEDIUM",
    HIGH_ACTIVITY: "HIGH",
}


# ---------------------------------------------------------------- helpers

def _as_dict_or_none(value) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    try:
        if hasattr(value, "__dict__"):
            return dict(vars(value))
    except Exception:  # noqa: BLE001
        pass
    return None


def _get(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_list(value) -> list:
    if not value:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [value]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def _iso_today(now) -> str:
    if now:
        return str(now)
    return datetime.date.today().isoformat()


def _count_upcoming(appointments, now, horizon_days=APPOINT_HORIZON_DAYS):
    """Count appointments whose date is within [today, today+horizon]."""
    if not appointments:
        return 0
    today = _iso_today(now)
    try:
        ref = datetime.date.fromisoformat(today[:10])
    except ValueError:
        return len(appointments)
    limit = ref + datetime.timedelta(days=horizon_days)
    count = 0
    for item in appointments:
        date_str = _get(item, "appointment_date", "") or ""
        if not date_str:
            continue
        try:
            d = datetime.date.fromisoformat(str(date_str)[:10])
        except ValueError:
            continue
        if ref <= d <= limit:
            count += 1
    return count


def _dimension_level(score: float) -> str:
    if score < 0.4:
        return LEVEL_LOW
    if score < 0.6:
        return LEVEL_MODERATE
    return LEVEL_HIGH
# ---------------------------------------------------------------- dimensions

def _medication_burden(profile, medications) -> Dict[str, Any]:
    prof_meds = _as_list(_get(profile, "medications"))
    db_meds = [m for m in (medications or []) if m]
    # Merge profile-listed names and DB rows (dedupe lowercased names).
    names = set()
    for name in prof_meds:
        n = str(name).strip().lower()
        if n:
            names.add(n)
    for med in db_meds:
        n = (str(_get(med, "medicine_name", "") or "")).strip().lower()
        if n:
            names.add(n)
    count = len(names)
    if count >= MEDS_HIGH:
        score = 0.85
    elif count >= MEDS_MODERATE:
        score = 0.55
    elif count >= 1:
        score = 0.30
    else:
        score = 0.10
    return {"name": "Medication Burden", "level": _dimension_level(score),
            "score": round(score, 4), "count": count}


def _condition_burden(profile) -> Dict[str, Any]:
    conditions = _as_list(_get(profile, "chronic_conditions"))
    count = len(conditions)
    if count >= COND_HIGH:
        score = 0.85
    elif count >= COND_MODERATE:
        score = 0.55
    elif count >= 1:
        score = 0.30
    else:
        score = 0.10
    return {"name": "Condition Burden", "level": _dimension_level(score),
            "score": round(score, 4), "count": count}


def _encounter_intensity(appointments, now) -> Dict[str, Any]:
    count = _count_upcoming(appointments, now)
    if count >= APPOINT_HIGH:
        score = 0.80
    elif count >= APPOINT_MODERATE:
        score = 0.50
    else:
        score = 0.10
    return {"name": "Encounter Intensity", "level": _dimension_level(score),
            "score": round(score, 4), "count": count}


def _care_complexity(profile) -> Dict[str, Any]:
    age = _get(profile, "age")
    conditions = len(_as_list(_get(profile, "chronic_conditions")))
    medications = len(_as_list(_get(profile, "medications")))
    base = (conditions + medications) / 8.0                      # [0, ~1+]
    age_boost = 0.0
    try:
        a = float(age)
        if a >= 85:
            age_boost = 0.25
        elif a >= 75:
            age_boost = 0.15
        elif a >= 65:
            age_boost = 0.05
    except (TypeError, ValueError):
        pass
    score = _clamp(base + age_boost)
    return {"name": "Care Complexity", "level": _dimension_level(score),
            "score": round(score, 4), "count": conditions + medications}


def _interaction_signal(conversation_history) -> Dict[str, Any]:
    messages = [m for m in (conversation_history or []) if m]
    count = len(messages)
    if count > HISTORY_HIGH:
        score = 0.75
    elif count >= HISTORY_MODERATE:
        score = 0.50
    elif count >= 1:
        score = 0.25
    else:
        score = 0.15
    return {"name": "Interaction Signal", "level": _dimension_level(score),
            "score": round(score, 4), "count": count}
# ---------------------------------------------------------------- core

def _classify_state(overall_score: float, dimensions) -> str:
    """Map the weighted overall score to a state label.

    STABLE additionally requires that every dimension is LOW (no active
    MODERATE/HIGH signal) so that a low score with a single elevated
    dimension is still called LOW_ACTIVITY, not STABLE.
    """
    if overall_score < STABLE_MAX_SCORE:
        if all(d["level"] == LEVEL_LOW for d in dimensions):
            return STABLE
        return LOW_ACTIVITY
    if overall_score < LOW_MAX_SCORE:
        return LOW_ACTIVITY
    if overall_score < MODERATE_MAX_SCORE:
        return MODERATE_ACTIVITY
    return HIGH_ACTIVITY


def _build_transition(previous_state, current_state, prev_score, cur_score):
    """Deterministic transition record from prior vs. current state."""
    score_delta = (cur_score if cur_score is not None else 0.0) - (
        prev_score if prev_score is not None else 0.0)
    if not previous_state or previous_state == NO_DATA:
        ttype = "INITIAL"
        direction = "NONE"
    elif current_state == previous_state:
        ttype = "CONTINUATION"
        direction = "STABLE"
    elif (current_state == HIGH_ACTIVITY
          or (current_state == MODERATE_ACTIVITY
              and previous_state == LOW_ACTIVITY)
          or (current_state == MODERATE_ACTIVITY
              and previous_state == STABLE)
          or (current_state == LOW_ACTIVITY
              and previous_state == STABLE)):
        ttype = "ESCALATION"
        direction = "UP"
    else:
        # any score decrease across a state boundary is a de-escalation
        ttype = "DE_ESCALATION"
        direction = "DOWN"
    if direction == "STABLE" or direction == "NONE":
        magnitude = 0.0
    else:
        magnitude = round(abs(score_delta) * 10.0, 4) if score_delta else 0.0
    return {
        "type": ttype,
        "direction": direction,
        "magnitude": magnitude,
        "score_delta": round(score_delta, 4),
        "previous_state": previous_state,
        "current_state": current_state,
        "supporting_evidence": {
            "basis": "DETERMINISTIC_RULE_BASED_REAL_SIGNALS"
                     if previous_state else "INITIAL_OBSERVATION",
        },
    }


def _assistance_plan_record(state, priority, state_score):
    """Deterministic assistance plan matching prepare_assistance_plan shape."""
    actions = {
        NO_DATA: [
            {"action": "GATHER_PROFILE", "reason": "insufficient real data"},
        ],
        STABLE: [
            {"action": "REINFORCE_MAINTENANCE",
             "reason": "state stable and all dimensions low"},
            {"action": "ENCOURAGE_SAFE_ROUTINE",
             "reason": "low care need"},
        ],
        LOW_ACTIVITY: [
            {"action": "MAINTAIN_ACTIVITY",
             "reason": "low care need"},
        ],
        MODERATE_ACTIVITY: [
            {"action": "MONITOR_SIGNALS",
             "reason": "moderate care need"},
            {"action": "PROVIDE_GUIDANCE",
             "reason": "moderate activity"},
        ],
        HIGH_ACTIVITY: [
            {"action": "ESCALATE_SUPPORT",
             "reason": "high care need"},
            {"action": "RECOMMEND_FOLLOW_UP",
             "reason": "elevated burden"},
        ],
    }
    safety = [
        "These actions adapt response structure and pacing only; they are "
        "not clinical advice, diagnoses, or risk predictions.",
        "Never turn a care-state label into a medical claim.",
        "Remain grounded in retrieved evidence.",
    ]
    return {
        "assistance_strategy": state.lower(),
        "priority": priority,
        "actions": actions.get(state, []),
        "safety_constraints": safety,
    }
def compute_care_state(
    profile=None,
    medications=None,
    appointments=None,
    conversation_history=None,
    previous_state=None,
    previous_score=None,
    now=None,
    patient_id=None,
):
    """Compute a deterministic care-state record from real system inputs.

    Returns a dict with the same keys the existing production chain consumes
    (prepare_adaptive_context / prepare_assistance_plan / build_grounded_prompt).
    """
    profile = _as_dict_or_none(profile)

    dims = [
        _medication_burden(profile, medications),
        _condition_burden(profile),
        _encounter_intensity(appointments, now),
        _care_complexity(profile),
        _interaction_signal(conversation_history),
    ]

    raw_score = sum(WEIGHTS[d["name"]] * d["score"] for d in dims)
    overall_score = round(_clamp(raw_score), 4)

    # NO_DATA only when no real inputs exist at all (no profile fields,
    # no DB medications/appointments, no conversation history).
    has_real_signal = bool(
        (profile and any(
            _get(profile, f, None) not in (None, "", [], ())
            for f in ("age", "chronic_conditions", "medications")))
        or (medications and any(bool(m) for m in medications))
        or (appointments and any(bool(a) for a in appointments))
        or (conversation_history and any(bool(m) for m in conversation_history)))
    if not has_real_signal:
        state = NO_DATA
        overall_score = None
    else:
        state = _classify_state(overall_score, dims)

    transition = _build_transition(
        previous_state, state, previous_score, overall_score)

    priority = PRIORITY_BY_STATE[state]
    mode = ASSISTANCE_BY_STATE[state]
    plan_record = _assistance_plan_record(state, priority, overall_score)

    window_start = _iso_today(now)
    window_end = window_start  # single observation window for real signals

    care_state = {
        "state": state,
        "overall_score": overall_score,
        "has_documented_activity": state != NO_DATA,
        "event_summary": {
            d["name"]: d["count"] for d in dims
        },
        "dimensions": dims,
        "interpretation": (
            "Real care signals are combined deterministically into a "
            "single care-state label. It is not a diagnosis and does not "
            "predict medical risk."),
    }

    return {
        "patient_id": patient_id,
        "window_start": window_start,
        "window_end": window_end,
        "context_status": "ACTIVE",
        "care_state": care_state,
        "transition": transition,
        "changed_dimensions": [d["name"] for d in dims],
        "adaptive_assistance": {
            "mode": mode,
            "priority": priority,
            "reason_codes": ["REAL_SIGNAL_" + state],
            "reasons": [
                "Deterministic rule-based estimation from real, current "
                "system signals."],
        },
        "assistance_plan_record": plan_record,
    }


if __name__ == "__main__":
    import json
    rec = compute_care_state(
        profile={"age": 68, "chronic_conditions": [], "medications": []},
        now="2026-09-22", patient_id="demo-low")
    print(json.dumps(rec, indent=2, default=str))