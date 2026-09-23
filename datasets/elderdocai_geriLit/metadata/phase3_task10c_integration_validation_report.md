# PHASE 3 TASK 10C — INTEGRATION VALIDATION & REPRODUCIBILITY AUDIT — REPORT

## 1. Task status
**PASS** — controlled validation of the Task 10B integration. 22/22 validator
checks, 11/11 unit tests, 7/7 API regression, live backend determinism and
equivalence verified; no artifact/source/API/frontend change; nothing committed.

## 2. Unit test results
- Command: `python -m unittest tests.test_retrieval_router` (pytest not
  installed in this environment; Task 10B used the same `unittest` runner).
- Result: **Ran 11 tests ... OK** (11/11 PASS). No test modified.

## 3. API regression results
- Command: `python -m unittest tests.test_api`
- Result: **Ran 7 tests ... OK** (7/7 PASS). API behavior unchanged.

## 4. Default legacy verification
- Env `ELDERDOCAI_RETRIEVAL_BACKEND` unset, `HF_HUB_OFFLINE=1`.
- `retrieve(query)` == direct `hybrid_search(query)` → **identical=True** (3 chunks).
- Legacy FAISS/BM25/CrossEncoder pipeline executes; evidence proceeds to the
  standard reliability/prompt path; adapters not invoked. (Functional only —
  no accuracy claim.)

## 5. Explicit legacy verification
- `ELDERDOCAI_RETRIEVAL_BACKEND=legacy` → output **equals direct hybrid_search**
  (`True`), across chunk_id/text/similarity_score/source_document/ranking order
  (compared via full dict equality of the same controlled query).

## 6. GeriLit functional verification (3 controlled queries, live)
Queries exercised falls, polypharmacy, social-wellbeing portions of the corpus.
Each query: router selects geri_lit; top-3 returned; every result has chunk_id,
pmcid, source_locator, text, similarity/rerank score, explicit authority_score,
document_category=="PMC"; evidence reached reliability path.
Sample PMCIDs: PMC7400355/PMC12188283/PMC7400355, PMC12125148/PMC8245739/
PMC12880692, PMC11899645/PMC11578716/PMC5121430. Functional only.

## 7. GeriLit determinism
- **WITHIN_PROCESS_DETERMINISTIC=True** (same query twice → identical chunk IDs,
  ordering, scores at 6 dp, source locators, text).
- **CROSS_PROCESS_ORDERING_IDS_LOCATORS_TEXT_STABLE=True**.
- Note: an earlier serialized comparison showed a 4-dp vs 6-dp rounding artifact
  (9.3933 vs 9.393291); normalized apples-to-apples → same query → same ordered
  evidence. deterministic = TRUE.

## 8. Invalid backend behavior
- `ELDERDOCAI_RETRIEVAL_BACKEND=INVALID_VALUE` → router logs warning and resolves
  to `legacy`; no crash, no GeriLit retrieval; output equals direct legacy
  (`True`); downstream pipeline valid.

## 9. GeriLit fallback behavior
- Validator C08: simulated GeriLit init failure → **fallback to legacy**, evidence
  from legacy (`['legacy-fb']`), no fabricated evidence.
- Validator C09: simulated failure with `fallback=False` → **empty evidence list**
  (safe downstream REJECT/empty path).
- Unit tests G/Gb also cover this (mock/RuntimeError-only; no real artifact touched).

## 10. Both-mode behavior
- `ELDERDOCAI_RETRIEVAL_BACKEND=both` → **legacy-only** (validator C06/C07 + live
  check `BOTH_MODE_LEGACY_ONLY=True`): no fusion, no score combination, no
  duplicate merge, GeriLit path never invoked. Documented:
  **both_mode_behavior = legacy_only** (intentional for Task 10B).
## 11. Offline / model-loading verification
- Ran with `HF_HUB_OFFLINE=1` for legacy default/explicit/invalid/both; GeriLit
  uses its own offline env vars and local cache. No network/download occurred;
  BGE + CrossEncoder + FAISS + BM25 all loaded from local cache. No model-version
  change.

## 12. API contract verification
- `/ask` request schema unchanged; response still contains `answer, sources,
  reliability, decision, care_context, profile_used` (verified by tests/test_api.py
  7/7 PASS and no code change to api/main.py). No backend-selection field added;
  frontend untouched.

## 13. Care-state regression
- No change to `get_adaptive_context`/`get_assistance_plan`/profiles/transitions/
  check-in logic; `api/main.py` and `carebuddy_service.py` unchanged
  (`git diff` clean for both). Care-state handling intact by inspection + API tests.

## 14. Frozen-artifact integrity
All recomputed hashes unchanged (validator C16–C19):
- chunks `62bfdde3…` · embeddings `b0905d2f…` · FAISS `b7016b9d…` · BM25
  `17f50324…` · row_mapping `4d649f81…` · Gold16 `e4478892…` · Gold96 `d4e1b6e6…`
  · accepted_manifest `f80cd4…` — ALL UNCHANGED.

## 15. Source-diff verification
- `git diff --unified=0 -- scripts/rag_chat.py`: exactly 4 changed lines
  (import + call replacement) — validator C20–C21 pass.
- No unrelated source changes; new `scripts/retrieval_router.py` and
  `scripts/geri_lit_adapter.py` are the only new integration source files; tests
  unchanged.

## 16. Reproducibility results
- Fresh-process runs: within-process GeriLit determinism TRUE; cross-process
  ordering/ID/locator/text stability TRUE; legacy default/explicit/invalid/both
  all reproduce identical output to direct `hybrid_search` in separate `python`
  invocations.
- Environment requirement: run from repo root (or with repo root on sys.path) —
  same pre-existing project convention.

## 17. Known limitations
- `both` = legacy-only (true multi-source fusion is a future task).
- pytest not installed; tests run via `python -m unittest` (identical coverage).
- Cross-process rerank scores match to 4 dp (deterministic ordering/IDs); exact
  6-dp equality verified within-process only.
- Guidance: run all validation from repo root for import-path conventions.

## 18. Recommendation for next phase
- The controlled integration is **ready for research evaluation** (e.g., dedicated
  GeriLit-Gold retrieval/grounding experiments) once the Gold96-v2/GeriLit-Gold
  benchmark is human-reviewed and frozen. No further code changes needed for the
  route; future tasks may implement true multi-source fusion only as a separate,
  explicitly authorized experiment.

## Validation artifact
`metadata/validation/phase3_task10c_validation.{json,md}` — 22 checks (C01–C22),
each with check_id/description/expected/actual/pass; all PASS.

## Git state (final)
- Branch `feature/mimic-pmc-migration`, HEAD `8b0696c` (unchanged).
- Modified: `scripts/rag_chat.py` (4 lines) + pre-existing `.gitignore`.
- Untracked: Task 10C metadata files (validator, report, validation JSON/MD) +
  Task 10B router/adapter/tests + pre-existing untracked dirs/files.
- **Nothing staged, committed, or pushed.**

## STOP condition
Task 10C complete. Not done (per constraints): no retrieval metrics, no
statistical analysis, no relevance labels, no human review, no Gold16/Gold96
change, no multi-source fusion, no care-state/API/frontend change, no frozen
artifact modification, no manuscript change, no commit/push.