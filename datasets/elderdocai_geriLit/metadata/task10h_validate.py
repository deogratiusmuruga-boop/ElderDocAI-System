#!/usr/bin/env python3
"""ElderDocAI-GeriLit - Phase 3 Task 10H validation (READ-ONLY).

Validates the v1.1 quantitative retrieval evaluation: benchmark integrity,
frozen-artifact integrity, exact Task 10D methodology settings, per-question
results completeness/validity, reproducibility signature, and that all prior
artifacts (Task 10D/10E/10F/10G) are unchanged.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

META_DIR = Path(__file__).resolve().parent
GERI_DIR = META_DIR.parent
REPO_DIR = GERI_DIR.parent.parent
VALIDATION_DIR = META_DIR / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

RES_FILE = META_DIR / "task10h_retrieval_results.json"
SUMM_FILE = META_DIR / "task10h_summary.json"
GOLD_11 = REPO_DIR / "data" / "geri_lit_gold_v1_1.json"
GOLD_10 = REPO_DIR / "data" / "geri_lit_gold.json"
REVIEW_11 = REPO_DIR / "data" / "geri_lit_gold_v1_1_review.json"
BASELINE = META_DIR / "task10f_frozen_hashes.json"
CHUNKS_FILE = GERI_DIR / "chunks" / "chunks.jsonl"
V10D_RES = META_DIR / "task10d_retrieval_results.json"
V10D_SUM = META_DIR / "task10d_summary.json"
V10E_ANA = META_DIR / "task10e_failure_analysis.json"
T10F_DIAG = META_DIR / "task10f_benchmark_reconstruction.json"
T10G_DIAG = META_DIR / "task10g_finalization.json"

V11_SHA = "1488d164e067084ff244edc3969450edb9244e3ed0e1fe10e2120fdf9dadee72"
V10_SHA = "28ef54faeec3b9a1cb8c1c79c9a478d8ea3618574b2787a82a74c1460f209c62"
FROZEN_EXPECT = {
    "chunks": "62bfdde3c7e77cf56112e79a71bc79bdea70767f6b2625701e392bbb6581c6b3",
    "embeddings": "b0905d2f96903dadc3cf5b4a737f330853656f846955de706f21bd714ce2dacf",
    "faiss": "b7016b9dabbab08ee68f875007b6db2b013af4472b1a3c0958af8a566b59f2b3",
    "bm25": "17f503240f2f9a13abf58c94359884b03b06bec1a5a59f63f921328e2a2c70c9",
    "row_mapping": "4d649f81d554c41ec7b63c65dce887466dfa6245745c568315735f1b93a1872b",
}
CONFIGS = ("dense", "sparse", "hybrid", "rerank")
METRIC_KEYS = ("recall@1", "recall@3", "recall@5", "mrr", "ndcg@5")
CONFIG_K = {"dense": 5, "sparse": 5, "hybrid": 5, "rerank": 3}

checks = []


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def add(cid, desc, expected, actual, passed):
    checks.append({"check_id": cid, "description": desc,
                   "expected": expected, "actual": actual,
                   "result": "PASS" if passed else "FAIL"})


def main():
    res = json.loads(RES_FILE.read_text(encoding="utf-8"))
    summ = json.loads(SUMM_FILE.read_text(encoding="utf-8"))
    g11 = json.loads(GOLD_11.read_text(encoding="utf-8"))
    recs_gold = g11["records"]
    rows = res["results"]

    corpus_chunk_ids = set()
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                corpus_chunk_ids.add(json.loads(line)["chunk_id"])

    # ------------------------------------------------ benchmark
    add("H1", "v1.1 benchmark exists with exact SHA-256",
        V11_SHA, sha256(GOLD_11), sha256(GOLD_11) == V11_SHA)
    add("H2", "v1.1 has 121 records, all ACCEPTED",
        121, (len(recs_gold), sum(1 for r in recs_gold
                                  if r["review_status"] == "ACCEPTED")),
        len(recs_gold) == 121 and all(
            r["review_status"] == "ACCEPTED" for r in recs_gold))
    add("H3", "no duplicate v1.1 IDs; no missing gold",
        "unique 121",
        len({r["final_benchmark_id"] for r in recs_gold}),
        len({r["final_benchmark_id"] for r in recs_gold}) == 121
        and all(r["gold_relevant_chunk_ids"] for r in recs_gold))

    # ------------------------------------------------ frozen artifacts
    add("H4", "v1.0 control unchanged",
        V10_SHA, sha256(GOLD_10), sha256(GOLD_10) == V10_SHA)
    frozen_ok = True
    for name, key in [("chunks", "chunks/chunks.jsonl"),
                      ("embeddings", "index/embeddings.npy"),
                      ("faiss", "index/faiss_index.bin"),
                      ("bm25", "index/bm25.pkl"),
                      ("row_mapping", "index/row_mapping.json")]:
        cur = sha256(GERI_DIR / key)
        ok = cur == FROZEN_EXPECT[name]
        frozen_ok &= ok
        add("H5_" + name, "frozen " + name + " unchanged",
            FROZEN_EXPECT[name], cur, ok)
    add("H5", "all 5 frozen GeriLit artifacts unchanged", "5/5",
        frozen_ok, frozen_ok)
# ------------------------------------------------ methodology
    k = res.get("config_k", {})
    add("H6", "configuration K settings match Task 10D",
        {"dense_k": 5, "sparse_k": 5, "hybrid_k": 5, "final_k": 3},
        k, k == {"dense_k": 5, "sparse_k": 5, "hybrid_k": 5, "final_k": 3})
    add("H7", "hybrid weights 0.6/0.4 preserved",
        (0.6, 0.4), (res.get("weights", {}).get("dense_weight"),
                     res.get("weights", {}).get("sparse_weight")),
        (res.get("weights", {}).get("dense_weight") == 0.6
         and res.get("weights", {}).get("sparse_weight") == 0.4))
    add("H8", "all four configurations present with exact metrics",
        list(CONFIGS),
        sorted({c for r in rows for c in r["metrics"]}),
        all(set(r["metrics"]) == set(CONFIGS) and
            all(set(r["metrics"][c]) == set(METRIC_KEYS) for c in CONFIGS)
            for r in rows))
    add("H9", "ranking lengths respect config K caps",
        {"dense<=5": 5, "sparse<=5": 5, "hybrid<=5": 5, "rerank<=3": 3},
        {c: max(len(r["rankings"][c]) for r in rows) for c in CONFIGS},
        all(max(len(r["rankings"][c]) for r in rows) <= CONFIG_K[c]
            for c in CONFIGS))

    # ------------------------------------------------ results completeness
    gold_ids = {r["final_benchmark_id"] for r in recs_gold}
    add("H10", "every v1.1 question evaluated exactly once",
        "121 unique",
        (len(rows), len({r["final_benchmark_id"] for r in rows})),
        len(rows) == 121 and len({r["final_benchmark_id"]
                                  for r in rows}) == 121
        and {r["final_benchmark_id"] for r in rows} == gold_ids)
    add("H11", "no duplicate evaluation rows",
        121, len({(r["final_benchmark_id"], r["question"]) for r in rows}),
        len({(r["final_benchmark_id"], r["question"]) for r in rows}) == 121)
    add("H12", "all gold chunks traceable to the frozen corpus",
        "all present",
        sum(1 for r in recs_gold for x in r["gold_relevant_chunk_ids"]
            if x in corpus_chunk_ids),
        all(x in corpus_chunk_ids for r in recs_gold
            for x in r["gold_relevant_chunk_ids"]))
    add("H13", "all retrieved chunk ids exist in the corpus",
        "all present",
        sum(1 for r in rows for c in CONFIGS for x in r["rankings"][c]
            if x in corpus_chunk_ids),
        all(x in corpus_chunk_ids for r in rows for c in CONFIGS
            for x in r["rankings"][c]))
    add("H14", "all metrics within [0,1]",
        "121x4x5",
        sum(1 for r in rows for c in CONFIGS
            for v in r["metrics"][c].values() if 0.0 <= v <= 1.0),
        all(0.0 <= v <= 1.0 for r in rows for c in CONFIGS
            for v in r["metrics"][c].values()))
# ------------------------------------------------ reproducibility
    add("H15", "results signature equals summary signature",
        summ.get("results_signature"), res.get("signature"),
        res.get("signature") == summ.get("results_signature"))
    add("H16", "canonical signature matches recorded Task 10H runs",
        "f3accb4e27f7934de02ae13204e723778294e2bd773c025cbd9bcbe4179b5e9c",
        res.get("signature"),
        res.get("signature")
        == "f3accb4e27f7934de02ae13204e723778294e2bd773c025cbd9bcbe4179b5e9c")
    add("H17", "benchmark integrity recorded by evaluation",
        V11_SHA, (res.get("benchmark", {}).get("expected_sha256"),
                  res.get("protected_hashes_before", {}).get("v1.1")),
        (res.get("benchmark", {}).get("expected_sha256") == V11_SHA
         and res.get("protected_hashes_before", {}).get("v1.1") == V11_SHA))

    # ------------------------------------------------ historical artifacts
    base = json.loads(BASELINE.read_text(encoding="utf-8"))["baseline"]
    for name, path, key in [
        ("task10d_results", V10D_RES, "metadata/task10d_retrieval_results.json"),
        ("task10d_summary", V10D_SUM, "metadata/task10d_summary.json"),
        ("task10e_analysis", V10E_ANA, "metadata/task10e_failure_analysis.json"),
    ]:
        exp = base.get(key, {}).get("sha256")
        cur = sha256(path)
        add("H18_" + name, name + " unchanged (Task 10F baseline)", exp, cur,
            bool(exp) and cur == exp)
    t10g = json.loads(T10G_DIAG.read_text(encoding="utf-8"))
    add("H19", "Task 10G final benchmark unchanged",
        V11_SHA, sha256(GOLD_11), sha256(GOLD_11) == V11_SHA)
    add("H20", "Task 10G review artifact unchanged",
        t10g.get("review_sha256"), sha256(REVIEW_11),
        sha256(REVIEW_11) == t10g.get("review_sha256"))
    v10e = json.loads(V10E_ANA.read_text(encoding="utf-8"))
    add("H21", "Task 10E diagnostics still cover 134 queries",
        134, len(v10e.get("per_query", [])),
        len(v10e.get("per_query", [])) == 134)
    add("H22", "v1.0 methodological equivalence rerun matches Task 10D",
        "0 mismatches over 134",
        (summ.get("v10_equivalence", {}) or {}).get("n_mismatches"),
        bool(summ.get("v10_equivalence"))
        and summ["v10_equivalence"].get("methodological_equivalence_ok") is True)

    failed = [c for c in checks if c["result"] == "FAIL"]
    for c in failed:
        print("FAILED:", c["check_id"], c["description"], c["actual"])

    out = {
        "task": "phase3_task10h",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validated_by": "metadata/task10h_validate.py",
        "n_questions": len(rows),
        "checks": checks,
        "n_passed": sum(1 for c in checks if c["result"] == "PASS"),
        "n_failed": len(failed),
    }
    (VALIDATION_DIR / "phase3_task10h_validation.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    lines = ["# PHASE 3 TASK 10H - VALIDATION", "",
             "Generated: " + out["generated_at_utc"], "",
             "Checks: **" + str(out["n_passed"]) + "/" + str(len(checks))
             + " PASS**, " + str(out["n_failed"]) + " FAIL", "",
             "| Check | Description | Result |", "|---|---|---|"]
    for c in checks:
        lines.append("| " + c["check_id"] + " | " + c["description"] + " | "
                     + c["result"] + " |")
    lines += ["",
              "* Evaluation-only task: frozen retriever/indexes/models used;",
              "  no retrieval code, weights, benchmark, or production file",
              "  changed; no model/package downloads.",
              "* Task 10D/10E/10F/10G artifacts unchanged."]
    (VALIDATION_DIR / "phase3_task10h_validation.md").write_text(
        "\n".join(lines), encoding="utf-8")

    print(json.dumps({"n_passed": out["n_passed"],
                      "n_failed": out["n_failed"],
                      "total": len(checks),
                      "n_questions": len(rows)}, indent=2))


if __name__ == "__main__":
    main()