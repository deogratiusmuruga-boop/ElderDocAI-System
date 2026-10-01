#!/usr/bin/env python3
"""A1 - Deterministic, pre-registered sample selection for human calibration.

Selects a fixed, reproducible subset of Experiment 1 questions (n=20) for
human annotation, stratified on the LLM judge's faithfulness score so that
the calibration span covers the score range the judge uses in production
(high / partial / low).

Pre-registered choices (documented before labels are collected):
  * seed = 42 (fixed; makes the draw fully reproducible from the frozen
    dataset)
  * strata on faithfulness: high = 1.0, partial = 0.75, low <= 0.5
  * allocation: proportional, with a minimum of 4 per stratum so that
    partial and low cover the failure modes the calibration is meant to
    catch: high=11, partial=4, low=5 (total 20)
  * ordering: ascending question_id (trivial, no substantive effect)

Reads only frozen artifacts; writes only under paper_assembly/a1_human_calibration/.
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent
META = Path(__file__).resolve().parent.parent.parent

EXP1_JSON = (META / "experiment1_generation_evaluation"
             / "experiment1_per_question_run1.json")
SEED = 42
ALLOCATION = {"high": 11, "partial": 4, "low": 5}


def stratify(rows):
    out = {"high": [], "partial": [], "low": []}
    for r in rows:
        f = r["faithfulness"]
        if f == 1.0:
            out["high"].append(r)
        elif f == 0.75:
            out["partial"].append(r)
        else:  # 0.0 / 0.25
            out["low"].append(r)
    return out


def main():
    with open(EXP1_JSON, "r", encoding="utf-8") as f:
        rows = json.load(f)

    strata = stratify(rows)
    rng = random.Random(SEED)
    selected = []
    for name in ["high", "partial", "low"]:
        pool = sorted(strata[name], key=lambda r: r["question_id"])
        pick = rng.sample(pool, ALLOCATION[name])
        selected.extend(pick)

    selected.sort(key=lambda r: r["question_id"])
    manifest = {
        "pre_registered": {
            "seed": SEED,
            "n": len(selected),
            "strata_definition": "faithfulness: high==1.0, partial==0.75, "
                                 "low<=0.5",
            "allocation": ALLOCATION,
            "population": len(rows),
        },
        "items": [],
    }
    for r in selected:
        manifest["items"].append({
            "question_id": r["question_id"],
            "topic": r.get("topic"),
            "stratum": ("high" if r["faithfulness"] == 1.0
                        else "partial" if r["faithfulness"] == 0.75
                        else "low"),
            # Judge scores are recorded here for the later human-vs-judge
            # agreement computation but are intentionally NOT exposed on the
            # annotation sheet (the human annotator is blinded to them).
            "judge_faithfulness": r["faithfulness"],
            "judge_answer_relevance": r["answer_relevance"],
            "judge_evidence_support": r["evidence_support"],
            "judge_hallucination_flag": bool(r["hallucination"]),
        })

    (OUT / "a1_sample_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- blinded annotation sheet (no judge values) ----
    cols = ["question_id", "stratum", "topic", "question", "answer",
            "evidence_abridged",
            "human_faithfulness", "human_answer_relevance",
            "human_evidence_support", "human_unsupported_claim", "notes"]
    import csv
    with open(OUT / "a1_annotation_sheet.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in selected:
            ev = r.get("retrieved_evidence_texts") or []
            abridged = " || ".join(
                (e[:400] + ("..." if len(e) > 400 else "")) for e in ev)
            w.writerow({
                "question_id": r["question_id"],
                "stratum": ("high" if r["faithfulness"] == 1.0
                            else "partial" if r["faithfulness"] == 0.75
                            else "low"),
                "topic": r.get("topic"),
                "question": r["question"],
                "answer": r["answer"],
                "evidence_abridged": abridged,
                "human_faithfulness": "",
                "human_answer_relevance": "",
                "human_evidence_support": "",
                "human_unsupported_claim": "",
                "notes": "",
            })

    print("wrote a1_sample_manifest.json (%d items, seed=%d)"
          % (len(selected), SEED))
    print("  strata: high=%d partial=%d low=%d"
          % (sum(1 for s in selected
                 if s["faithfulness"] == 1.0),
             sum(1 for s in selected if s["faithfulness"] == 0.75),
             sum(1 for s in selected if s["faithfulness"] <= 0.5)))
    print("  annotation sheet: a1_annotation_sheet.csv (blinded)")


if __name__ == "__main__":
    main()