#!/usr/bin/env python3
"""ElderDocAI - Task 3 reliability evaluation validator (READ-ONLY).

Verifies: benchmark identity, population completeness, reliability bounds,
factor bounds, decision-threshold consistency, determinism across runs, and
frozen-artifact integrity (authoritative SHA-256 where a Task 10F baseline
exists; run-invariance for newer 10F/10G/10H outputs).
"""
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
META_DIR = Path(__file__).resolve().parent
GERI_DIR = META_DIR.parent.parent
REPO_DIR = GERI_DIR.parent.parent
for _p in (str(GERI_DIR), str(REPO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.reliability_config import load_reliability_config  # noqa: E402
from scripts.adaptive_decision_controller import (  # noqa: E402
    make_reliability_decision)

GOLD_11 = REPO_DIR / "data" / "geri_lit_gold_v1_1.json"
V11_SHA = "1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72"
V10_SHA = "28ef54faeec3b9a1cb8c1c79c9a478d8ea3618574b2787a82a74c1460f209c62"
SUMM = META_DIR / "task3_summary.json"
PERQ = META_DIR / "task3_per_question.json"
HASH = META_DIR / "task3_frozen_hashes.json"
BASELINE = GERI_DIR / "metadata/task10f_frozen_hashes.json"

checks = []


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def add(cid, desc, expected, actual, passed):
    checks.append({"check_id": cid, "description": desc,
                   "expected": expected, "actual": actual,
                   "result": "PASS" if passed else "FAIL"})


def main():
    summ = json.loads(SUMM.read_text(encoding="utf-8"))
    perq = json.loads(PERQ.read_text(encoding="utf-8"))
    rows = perq["rows"]
    cfg = load_reliability_config()
    wei = cfg["reliability_weights"]
    thr = cfg["decision_thresholds"]

    add("T1", "v1.1 benchmark hash correct", V11_SHA, sha256(GOLD_11),
        sha256(GOLD_11) == V11_SHA)
    add("T2", "v1.1 status/version", ("FROZEN", "1.1"),
        (summ["benchmark"]["status"], summ["benchmark"]["version"]),
        summ["benchmark"]["status"] == "FROZEN" and
        summ["benchmark"]["version"] == "1.1")
    add("T3", "population = 121 unique questions", 121,
        (len(rows), len({r["final_benchmark_id"] for r in rows})),
        len(rows) == 121 and
        len({r["final_benchmark_id"] for r in rows}) == 121)
    add("T4", "all rows have question/topic/pmcid", "121/121",
        sum(1 for r in rows if r["question"] and r["topic"] and r["pmcid"]),
        all(r["question"] and r["topic"] and r["pmcid"] for r in rows))
    add("T5", "all topics valid C01-C10", 10, len({r["topic"] for r in rows}),
        all(r["topic"] in {f"C{i:02d}" for i in range(1, 11)} for r in rows))
    add("T6", "weights unchanged (0.3/0.3/0.2/0.1/0.1)",
        {"authority": 0.3, "relevance": 0.3, "support": 0.2,
         "coverage": 0.1, "consistency": 0.1}, wei,
        wei == {"authority": 0.3, "relevance": 0.3, "support": 0.2,
                "coverage": 0.1, "consistency": 0.1})
    add("T7", "thresholds unchanged (0.80/0.65/0.45)",
        {"accept": 0.8, "refine": 0.65, "re_retrieve": 0.45}, thr,
        thr == {"accept": 0.8, "refine": 0.65, "re_retrieve": 0.45})
    add("T8", "reliability within [0,1]",
        "121 in [0,1]",
        sum(1 for r in rows if 0.0 <= r["overall_reliability"] <= 1.0),
        all(0.0 <= r["overall_reliability"] <= 1.0 for r in rows))
    add("T9", "factors within [0,1]",
        "121x5 in [0,1]",
        sum(1 for r in rows for k in
            ("authority", "relevance", "support", "coverage", "consistency")
            if 0.0 <= r[k] <= 1.0),
        all(0.0 <= r[k] <= 1.0 for r in rows for k in
            ("authority", "relevance", "support", "coverage", "consistency")))
    cons_ok = all(
        make_reliability_decision(
            {"overall_reliability": r["overall_reliability"]})["decision"]
        == r["decision"] for r in rows)
    add("T10", "decisions consistent with fixed thresholds",
        "121/121",
        sum(1 for r in rows if
            make_reliability_decision(
                {"overall_reliability":
                 r["overall_reliability"]})["decision"] == r["decision"]),
        cons_ok)
    dist = Counter(r["decision"] for r in rows)
    add("T11", "decision distribution sums to 121",
        121, sum(dist.values()), sum(dist.values()) == 121)
    add("T12", "valid decision labels only",
        sorted({"ACCEPT", "REFINE", "RE-RETRIEVE", "REJECT"}),
        sorted(set(dist)),
        set(dist) <= {"ACCEPT", "REFINE", "RE-RETRIEVE", "REJECT"})
    add("T13", "no empty-evidence cases", 0,
        summ["evidence_behaviour"]["empty_evidence"],
        summ["evidence_behaviour"]["empty_evidence"] == 0)
    add("T14", "deterministic runs (manifest identical)",
        True, summ.get("deterministic"),
        summ.get("deterministic") is True and len(
            {r["signature"] for r in summ["run_manifest"].values()}) == 1)
# ------------------------------------------------ frozen integrity
    exp = {
        "v1.0": V10_SHA,
        "v1.1": V11_SHA,
        "chunks": "62bfdde3c7e77cf56112e79a71bc79bdea70767f6b2625701e392bbb6581c6b3",
        "embeddings": "b0905d2f96903dadc3cf5b4a737f330853656f846955de706f21bd714ce2dacf",
        "faiss": "b7016b9dabbab08ee68f875007b6db2b013af4472b1a3c0958af8a566b59f2b3",
        "bm25": "17f503240f2f9a13abf58c94359884b03b06bec1a5a59f63f921328e2a2c70c9",
        "row_mapping": "4d649f81d554c41ec7b63c65dce887466dfa6245745c568315735f1b93a1872b",
    }
    man = json.loads(HASH.read_text(encoding="utf-8"))["frozen_artifacts"]
    for k, e in exp.items():
        add("T15_" + k, "frozen " + k + " unchanged", e, man.get(k),
            man.get(k) == e)
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))["baseline"]
    for key, mkey in [
        ("metadata/task10d_retrieval_results.json", "t10d_results"),
        ("metadata/task10d_summary.json", None),
        ("metadata/task10e_failure_analysis.json", "t10e_analysis"),
    ]:
        cur = man.get(mkey) if mkey else sha256(GERI_DIR / key)
        bl = baseline[key]["sha256"]
        add("T16_" + (mkey or key), "Task 10D/10E artifact unchanged", bl,
            cur, cur == bl)
    for name in ("t10f_diag", "t10g_diag", "t10h_results"):
        add("T17_" + name, name + " recorded (run-invariant)", "present",
            man.get(name), bool(man.get(name)))

    failed = [c for c in checks if c["result"] == "FAIL"]
    for c in failed:
        print("FAILED:", c["check_id"], c["description"], c["actual"])

    out = {
        "task": "phase3_task3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validated_by": "task3_validate.py",
        "n_questions": len(rows),
        "checks": checks,
        "n_passed": sum(1 for c in checks if c["result"] == "PASS"),
        "n_failed": len(failed),
    }
    (META_DIR / "phase3_task3_reliability_validation.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    lines = ["# PHASE 3 TASK 3 - RELIABILITY RE-EVALUATION VALIDATION", "",
             "Generated: " + out["generated_at_utc"], "",
             "Checks: **" + str(out["n_passed"]) + "/" + str(len(checks))
             + " PASS**, " + str(out["n_failed"]) + " FAIL", "",
             "| Check | Description | Result |", "|---|---|---|"]
    for c in checks:
        lines.append("| " + c["check_id"] + " | " + c["description"] + " | "
                     + c["result"] + " |")
    (META_DIR / "phase3_task3_reliability_validation.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps({"n_passed": out["n_passed"], "n_failed": out["n_failed"],
                      "total": len(checks)}, indent=2))


if __name__ == "__main__":
    main()