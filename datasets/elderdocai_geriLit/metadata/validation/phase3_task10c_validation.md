# ElderDocAI-GeriLit Phase 3 Task 10C - Validation

- **timestamp**: 2026-09-21T04:54:19.299474+00:00

| ID | Description | Expected | Actual | PASS |
|---|---|---|---|--|
| C01 | default backend legacy (env absent) | legacy | legacy | True |
| C02 | explicit legacy | legacy | legacy | True |
| C03 | explicit geri_lit | geri_lit | geri_lit | True |
| C04 | both backend allowed | both | both | True |
| C05 | invalid backend -> legacy | legacy | legacy | True |
| C06 | both -> legacy-only (no fusion) | ['legacy-1'] | ['legacy-1'] | True |
| C07 | both never invokes geri_lit path | False | False | True |
| C08 | geri_lit failure -> legacy fallback | ['legacy-fb'] | ['legacy-fb'] | True |
| C09 | no-fallback -> empty evidence | [] | [] | True |
| C10 | adapter deterministic | True | True | True |
| C11 | rerank->similarity | 0.9 | 0.9 | True |
| C12 | authority explicit | 0.85 | 0.85 | True |
| C13 | category PMC | PMC | PMC | True |
| C14 | provenance preserved | PMC1:BODY:s:0..1 | PMC1:BODY:s:0..1 | True |
| C15 | input not mutated | 0.9 | 0.9 | True |
| C16 | frozen GeriLit unchanged | True | True | True |
| C17 | Gold16 unchanged | True | True | True |
| C18 | Gold96 unchanged | True | True | True |
| C19 | accepted_manifest unchanged | True | True | True |
| C20 | rag_chat diff = 4 changed lines | 4 | 4 | True |
| C21 | new router/adapter exist | True | True | True |
| C22 | nothing staged |  |  | True |
