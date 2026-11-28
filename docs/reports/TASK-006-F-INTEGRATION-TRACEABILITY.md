# TASK 006-F — Integration Traceability Matrix

**Candidate disposition: integrated and exact-head hosted CI passed; independent acceptance and scientific/operational gates remain open.** Draft [PR #29](https://github.com/samuelindra123/xynoos-cyber-model/pull/29) passed Foundation CI [run 46158564561](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/46158564561) at exact head `6af3580b8d885098bc5a6e2429d9810497808899`. All milestone PRs remain unmerged. `main` stays at `32eca2b900c4f7a88294f7a24a5dfae0604ae798`.

| Requirement / phase | Source or evidence | Verification | Status |
|---|---|---|---|
| Phase 0 repository forensics | F baseline report; `AGENTS.md`; compute/engineering policies | Main SHA, ancestry, PR base/head and CI records, one worktree, no tags; F CI run `46158564561` | Pass; exact-head hosted CI passed |
| Runtime locking/admission/crash | R2 `21d9990`, C3 `b7656ab`, F test `task006f_runtime_independent.rs` | Full `xynoos-training` package debug/release on `00905b2`; 10 repeated lock-race runs, 3 repeated C3 race runs; F hosted CI | Software pass; independent reviewer/provider boundary open |
| Numerical correctness | Native tensor/autograd/training source and frozen oracle fixtures | Numerical 295 cases/2670 outputs; 25 gradient rows, 101 learning rows; AdamW 33 records; native independent controls | Pass within tested fixture domain |
| Objective A corpus | PR #28 `f122f4e`, v3 manifest/tokenizer/audit | Python v3 generation and independent audit; Rust v3 data tests; independent stdlib split cross-check | Integrity/leakage checks pass; scientific admission fails |
| Evaluation protocol | D1 v3 baseline and public evaluation code | Baseline reproduced in recorded v3 evidence | Baseline plumbing pass; public test exposed, validation adequacy not established |
| Checkpoint integrity and deterministic local resume | D2 `3cf67c9`; D2 test evidence | Corruption/semantic/identity tests and uninterrupted-vs-resume state equality | Local pass; remote persistence unverified |
| Resource ceilings | Compute policy, D3, E v3 smoke record | Check workflow caps and compare smoke/projection to work ceiling | Limits preserved; complete workload fit not demonstrated |
| First-training package | F first-training readiness report | Exact config/hash/protocol/acceptance gate review | Not frozen/admitted due scientific, workload, storage, and review blockers |
| Integration candidate | F branch merging #23 and #28 into E | Local merge reviewed; new F runtime tests pass in debug/release; PR #29 exact-head run `46158564561` | Exact head `6af3580` passed all hosted gates; no merge |

## Integration history and hosted evidence

The candidate is based on E `a5844e5606d2bdeaf9d4762110e63d93e71323eb`, with local merge commits incorporating exact C3 head `b7656abbc38703b3a34652cc5c7f9435a2b0c32c` and v3 head `f122f4e283b06d2cd20dd544fc0b18af4c8b6c98`. E’s tree lacked the exact C3 audit head and v3 implementation; PR #28 diverged from E’s descendant chain. Conflicts were resolved in `ci.yml` and `ROADMAP.md`. CI expected Rust checks were increased from 189 to 191 to account for the two active F tests. No numerical fixture, tolerance, optimizer equation, or compute ceiling was changed. The full training package passes in debug and release on `00905b2abc11d46002da57f2607a3ec18a73f956`. Integrated exact-head Foundation CI run `46158564561` passed all steps at `6af3580b8d885098bc5a6e2429d9810497808899`. The documentation-only evidence refresh is not a separate implementation change and will also receive exact-head hosted CI.

Historical exact-head Foundation runs: #23 `45824251360`, #24 `45781385665`, #25 `45784299844`, #26 `45824896343`, #27 `45857473668`, #28 `45855781546`; all recorded as passed on their respective SHA. These do not validate the integrated F head. F PR #29 run `46158564561` validates exact head `6af3580b8d885098bc5a6e2429d9810497808899`; the final documentation evidence commit receives its own exact-head run.

## Critical open traceability items

1. Independent human/reviewer acceptance of R2/C3 and the integrated diff.
2. Scientifically reviewed Objective A protocol with adequate held-out evaluation and a truly untouched test.
3. Measured complete workload fit within hard limits, on the intended bounded provider, at least three times.
4. Owner-selected private artifact destination and verified interrupted-transfer/restore/resume/cleanup exercise.
5. Frozen exact source/config/data/tokenizer identity and predeclared measurable quality thresholds.
6. Owner authorization of the exact immutable training dispatch. This is a separate future action and is not implied by the existence of this candidate.
