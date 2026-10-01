#!/usr/bin/env python3
"""Experiment 1 - reproducibility comparison (run1 vs run2).

Compares substantive per-question fields, computes run signatures, and writes
experiment1_reproducibility.json. Latency and timestamps are non-substantive.
"""
import hashlib
import json
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

SUBSTANTIVE_FIELDS = [
    "retrieved_evidence_ids", "reliability", "decision",
    "refinement_attempts", "retrieval_attempts", "refused",
    "generated", "answer", "faithfulness", "answer_relevance",
    "evidence_support", "hallucination",
]


def load(name):
    p = OUT_DIR / name
    if not p.exists():
        print("MISSING FILE: %s" % p)
        sys.exit(2)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def signature(rows, fields):
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda x: x["question_id"]):
        payload = {f: r.get(f) for f in fields}
        h.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def main():
    rows1 = load("experiment1_per_question_run1.json")
    rows2 = load("experiment1_per_question_run2.json")
    m1 = {r["question_id"]: r for r in rows1}
    m2 = {r["question_id"]: r for r in rows2}

    if set(m1) != set(m2):
        print("QUESTION ID SETS DIFFER")
        sys.exit(1)

    sig1 = signature(rows1, SUBSTANTIVE_FIELDS)
    sig2 = signature(rows2, SUBSTANTIVE_FIELDS)

    field_match = {}
    mismatched_questions = []
    for field in SUBSTANTIVE_FIELDS:
        bad = [q for q in m1 if m1[q].get(field) != m2[q].get(field)]
        field_match[field] = {
            "identical": len(bad) == 0,
            "matching": len(m1) - len(bad),
            "total": len(m1),
            "mismatched_question_ids": sorted(bad),
        }
        if bad:
            mismatched_questions.extend(bad)

    out = {
        "run1_file": "experiment1_per_question_run1.json",
        "run2_file": "experiment1_per_question_run2.json",
        "run1_signature": sig1,
        "run2_signature": sig2,
        "signatures_identical": sig1 == sig2,
        "field_comparison": field_match,
        "mismatched_question_ids": sorted(set(mismatched_questions)),
        "deterministic": (
            sig1 == sig2 and not mismatched_questions),
        "note": (
            "Substantive fields compared. Latency and timestamps excluded. "
            "LLM generation configured temperature=0 / top_p=0.1 / top_k=10; "
            "judge uses the same deterministic sampling options."),
    }
    with open(OUT_DIR / "experiment1_reproducibility.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("run1 signature:", sig1)
    print("run2 signature:", sig2)
    print("identical:", sig1 == sig2)
    for field, info in field_match.items():
        print("  %-28s identical=%s (%d/%d)"
              % (field, info["identical"], info["matching"],
                 info["total"]))
    print("deterministic:", out["deterministic"])


if __name__ == "__main__":
    main()