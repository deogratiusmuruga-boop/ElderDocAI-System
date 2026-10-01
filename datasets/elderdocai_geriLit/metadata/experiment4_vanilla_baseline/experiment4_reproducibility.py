#!/usr/bin/env python3
"""Experiment 4 - reproducibility comparison (run1 vs run2).

Compares substantive paired fields and writes experiment4_reproducibility.json.
Judge-instrument instability (if observed) is reported separately.
"""
import hashlib
import json
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

SUBSTANTIVE_FIELDS = [
    "evidence_identity_ok",
    ("full", "evidence_ids"), ("full", "reliability"), ("full", "decision"),
    ("full", "refinement_attempts"), ("full", "retrieval_attempts"),
    ("full", "refused"), ("full", "generated"), ("full", "answer"),
    ("full", "faithfulness"), ("full", "answer_relevance"),
    ("full", "evidence_support"),
    ("vanilla", "evidence_ids"), ("vanilla", "generated"),
    ("vanilla", "answer"), ("vanilla", "faithfulness"),
    ("vanilla", "answer_relevance"), ("vanilla", "evidence_support"),
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
    if isinstance(key, tuple):
        branch = {"full": "full", "vanilla": "vanilla"}.get(key[0])
        if not branch:
            return None
        return row.get(branch, {}).get(key[1])
    return row.get(key)


def signature(rows, fields):
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda x: x["pair_id"]):
        payload = {}
        for f in fields:
            if isinstance(f, tuple):
                payload[".".join(f)] = _get(r, f)
            else:
                payload[f] = _get(r, f)
        h.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def main():
    rows1 = load("experiment4_per_question_run1.json")
    rows2 = load("experiment4_per_question_run2.json")
    m1 = {r["pair_id"]: r for r in rows1}
    m2 = {r["pair_id"]: r for r in rows2}

    if set(m1) != set(m2):
        print("PAIR ID SETS DIFFER")
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
            "mismatched_pair_ids": sorted(bad),
        }
        if bad:
            mismatched.extend(bad)

    sys_fields = [f for f in SUBSTANTIVE_FIELDS
                  if not (isinstance(f, tuple) and f[1] in (
                      "faithfulness", "answer_relevance",
                      "evidence_support"))]

    out = {
        "run1_file": "experiment4_per_question_run1.json",
        "run2_file": "experiment4_per_question_run2.json",
        "run1_signature": sig1,
        "run2_signature": sig2,
        "signatures_identical": sig1 == sig2,
        "field_comparison": field_match,
        "mismatched_pair_ids": sorted(set(mismatched)),
        "deterministic": sig1 == sig2 and not mismatched,
        "deterministic_system": signature(rows1, sys_fields)
                                == signature(rows2, sys_fields),
        "note": ("System fields (evidence, decisions, answers) compared for "
                 "determinism; LLM judge scores separated as measurement "
                 "instrument. Latency non-substantive."),
    }
    with open(OUT_DIR / "experiment4_reproducibility.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("run1 signature:", sig1)
    print("run2 signature:", sig2)
    print("signatures identical:", sig1 == sig2)
    print("system-fields deterministic:", out["deterministic_system"])
    for label, info in field_match.items():
        print("  %-32s identical=%s (%d/%d)"
              % (label, info["identical"], info["matching"],
                 info["total"]))
    print("deterministic (incl. judge):", out["deterministic"])


if __name__ == "__main__":
    main()