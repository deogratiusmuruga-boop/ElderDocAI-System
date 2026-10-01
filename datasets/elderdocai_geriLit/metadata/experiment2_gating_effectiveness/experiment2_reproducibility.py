#!/usr/bin/env python3
"""Experiment 2 - reproducibility comparison (run1 vs run2).

Compares substantive paired fields, computes run signatures, writes
experiment2_reproducibility.json. Latency/timestamps non-substantive.
"""
import hashlib
import json
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

SUBSTANTIVE_FIELDS = [
    "initial_evidence_ids", "initial_decision",
    ("on", "decision"), ("on", "reliability"),
    ("on", "refinement_attempts"), ("on", "retrieval_attempts"),
    ("on", "refused"), ("on", "generated"), ("on", "evidence_ids"),
    ("on", "answer"), ("on", "faithfulness"), ("on", "answer_relevance"),
    ("on", "evidence_support"),
    ("off", "evidence_ids"), ("off", "refused"), ("off", "generated"),
    ("off", "answer"), ("off", "faithfulness"), ("off", "answer_relevance"),
    ("off", "evidence_support"),
    "paired_differences",
]


def load(name):
    p = OUT_DIR / name
    if not p.exists():
        print("MISSING FILE: %s" % p)
        sys.exit(2)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _get(row, key):
    # map ('on', f) -> row['gate_on'][f], ('off', f) -> row['gate_off'][f]
    if isinstance(key, tuple):
        branch = {"on": "gate_on", "off": "gate_off"}.get(key[0])
        if not branch:
            return None
        return row.get(branch, {}).get(key[1])
    return row.get(key)


def signature(rows, fields):
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda x: x["question_id"]):
        payload = {}
        for f in fields:
            if isinstance(f, tuple):
                payload[".".join(f)] = _get(r, f)
            else:
                payload[f] = _get(r, f)
        h.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def main():
    rows1 = load("experiment2_per_question_run1.json")
    rows2 = load("experiment2_per_question_run2.json")
    m1 = {r["question_id"]: r for r in rows1}
    m2 = {r["question_id"]: r for r in rows2}

    if set(m1) != set(m2):
        print("QUESTION ID SETS DIFFER")
        sys.exit(1)

    sig1 = signature(rows1, SUBSTANTIVE_FIELDS)
    sig2 = signature(rows2, SUBSTANTIVE_FIELDS)

    field_match = {}
    mismatched = []
    for f in SUBSTANTIVE_FIELDS:
        label = ".".join(f) if isinstance(f, tuple) else f
        bad = [q for q in m1 if _get(m1[q], f) != _get(m2[q], f)]
        field_match[label] = {
            "identical": len(bad) == 0,
            "matching": len(m1) - len(bad),
            "total": len(m1),
            "mismatched_question_ids": sorted(bad),
        }
        if bad:
            mismatched.extend(bad)

    # Relevance judge scores are produced by the same llama3.2 judge with the
    # same deterministic options, but the judge is a separate measurement
    # instrument: boundary scores (e.g. 0.75 vs 1.0) can vary across runs on
    # byte-identical answers under Ollama JSON mode. This is judge
    # reproducibility, distinct from system-under-test determinism.
    rel_on_bad = [q for q in m1
                  if m1[q]["gate_on"]["answer_relevance"]
                  != m2[q]["gate_on"]["answer_relevance"]]
    rel_off_bad = [q for q in m1
                   if m1[q]["gate_off"]["answer_relevance"]
                   != m2[q]["gate_off"]["answer_relevance"]]
    sys_fields = [f for f in SUBSTANTIVE_FIELDS
                  if not (isinstance(f, tuple) and f[0] in ("on", "off")
                          and f[1] == "answer_relevance")
                  and f != "paired_differences"]

    out = {
        "run1_file": "experiment2_per_question_run1.json",
        "run2_file": "experiment2_per_question_run2.json",
        "run1_signature": sig1,
        "run2_signature": sig2,
        "signatures_identical": sig1 == sig2,
        "field_comparison": field_match,
        "mismatched_question_ids": sorted(set(mismatched)),
        "deterministic": sig1 == sig2 and not mismatched,
        "deterministic_system": signature(rows1, sys_fields)
                                == signature(rows2, sys_fields),
        "judge_relevance_nondeterminism": {
            "on_mismatched": sorted(rel_on_bad),
            "off_mismatched": sorted(rel_off_bad),
            "on_answers_identical": all(
                m1[q]["gate_on"]["answer"] == m2[q]["gate_on"]["answer"]
                for q in rel_on_bad),
            "off_answers_identical": all(
                m1[q]["gate_off"]["answer"] == m2[q]["gate_off"]["answer"]
                for q in rel_off_bad),
            "note": ("LLM answer-relevance judge: same judge model+options "
                     "re-run; boundary score (0.25 grid) varied on "
                     "byte-identical generated answers under Ollama JSON "
                     "mode. This is judge-instrument reproducibility, "
                     "neither a gate nor a generation effect."),
        },
        "note": ("Substantive system fields compared for determinism; "
                 "answer-relevance judge scores separated as a measurement "
                 "instrument. Latency non-substantive."),
    }
    with open(OUT_DIR / "experiment2_reproducibility.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("run1 signature:", sig1)
    print("run2 signature:", sig2)
    print("signatures identical:", sig1 == sig2)
    print("system-fields deterministic:",
          out["deterministic_system"])
    print("judge relevance nondeterminism on:", rel_on_bad,
          "off:", rel_off_bad)
    for label, info in field_match.items():
        print("  %-32s identical=%s (%d/%d)"
              % (label, info["identical"], info["matching"],
                 info["total"]))
    print("deterministic (incl. judge):", out["deterministic"])


if __name__ == "__main__":
    main()