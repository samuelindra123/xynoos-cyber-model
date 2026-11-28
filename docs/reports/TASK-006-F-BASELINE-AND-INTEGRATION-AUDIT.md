# TASK 006-F — Baseline and Integration Audit

**Disposition: integration candidate assembled locally; no accepted training source SHA.** `main` and `origin/main` were both `32eca2b900c4f7a88294f7a24a5dfae0604ae798` at audit start and remain unchanged. The current branch is a local candidate based on TASK 006-E with the exact C3 audit head and Objective A v3 head integrated. No PR was merged.

## Repository baseline

| Check | Finding |
|---|---|
| Canonical checkout | `/home/samuel/Documents/xynoos-cyber-model`; one worktree only. No additional permanent clone/worktree was created. |
| Starting branch and cleanliness | `feat/task-006-e-first-training-readiness` at `a5844e5606d2bdeaf9d4762110e63d93e71323eb`; clean before branch creation. |
| `main` | `32eca2b900c4f7a88294f7a24a5dfae0604ae798`; matches `origin/main`. |
| Integration branch | `feat/task-006-f-integration-assurance`; local merge commits integrate C3 `b7656abbc38703b3a34652cc5c7f9435a2b0c32c` and v3 `f122f4e283b06d2cd20dd544fc0b18af4c8b6c98`. |
| Historical references | A `26777fa`; B `ae9e038`; C `231d47d`; R2 `21d9990`; C3 exact head `b7656ab`; D1 `92da229`; D2 `3cf67c9`; D3 `20bcb3c`; E `a5844e5`; v3 `f122f4e`. Full 40-character SHAs are in the task specification or linked reports. |
| Policies reviewed | Repository `AGENTS.md`, `docs/COMPUTE_POLICY.md`, `docs/design/EFFICIENCY-FIRST-MODEL-ENGINEERING.md`, `docs/DATASET_POLICY.md`, `docs/EVALUATION.md`, and `docs/ROADMAP.md`. |
| Tags | None found. |

Local branch history preserves both merge parents. The current candidate is not accepted merely because its inputs had successful CI. Open historical branches remain as upstream refs; three old TASK 003 refs show deleted upstreams. They were not deleted or rewritten. `git worktree list --porcelain` confirms one canonical worktree.

## Dependency and pull request audit

The intended dependency chain is `#18 A architecture → #19 B engine → #20 C numerical audit → #21 R2 runtime remediation → #23 C3 audit → #24 D1 evaluation → #25 D2 persistence → #26 D3 compute → #27 E readiness`. PR #28 adds Objective A v3 from the D1 base and is a parallel dependency of E; E’s prose referred to v3 before its implementation and corpus were present in the E tree. F locally integrates the exact #23 and #28 heads into the E lineage, resolving reviewed conflicts in `ci.yml` and `ROADMAP.md` while preserving downstream gates.

| PR | Base → exact head | State at audit | Exact-head Foundation CI |
|---:|---|---|---|
| [#18](https://github.com/samuelindra123/xynoos-cyber-model/pull/18) | `main` → `26777fa95383600c003ff094216d3d1c310b051c` | Open draft; no review decision | Historical checks passed per branch history; no approval |
| [#19](https://github.com/samuelindra123/xynoos-cyber-model/pull/19) | A → `ae9e038af2b04a2fda4a4c624079a1ec82fd086c` | Open draft | [run 45662756847](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45662756847) passed |
| [#20](https://github.com/samuelindra123/xynoos-cyber-model/pull/20) | B → `231d47d2d95f52770fa20f4da93a91be762178d0` | Open draft | [run 45731891128](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45731891128) passed |
| [#21](https://github.com/samuelindra123/xynoos-cyber-model/pull/21) | C → `21d9990760380195b9d9379b6d43f848ad6c17df` | Open draft | [run 45772399582](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45772399582) passed |
| #22 | R2 parent → `2c95414b904e24c9988bab390292aa40bd570a09` | Closed; superseded by #23 | Historical run passed; not an integration input |
| [#23](https://github.com/samuelindra123/xynoos-cyber-model/pull/23) | R2 → `b7656abbc38703b3a34652cc5c7f9435a2b0c32c` | Open draft; no review decision | [run 45824251360](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45824251360) passed |
| [#24](https://github.com/samuelindra123/xynoos-cyber-model/pull/24) | C3 lineage → `92da22973914ab36055c4aa6d32143747d9a372f` | Open draft | [run 45781385665](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45781385665) passed |
| [#25](https://github.com/samuelindra123/xynoos-cyber-model/pull/25) | D1 → `3cf67c9e389e7b00ab5b9df9cf045c6f465f255c` | Open draft | [run 45784299844](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45784299844) passed |
| [#26](https://github.com/samuelindra123/xynoos-cyber-model/pull/26) | D2 → `20bcb3cfb9f257932d2e099845b9f3032d8732ff` | Open draft | [run 45824896343](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45824896343) passed |
| [#27](https://github.com/samuelindra123/xynoos-cyber-model/pull/27) | D3 → `a5844e5606d2bdeaf9d4762110e63d93e71323eb` | Open draft | [run 45857473668](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45857473668) passed |
| [#28](https://github.com/samuelindra123/xynoos-cyber-model/pull/28) | D1 → `f122f4e283b06d2cd20dd544fc0b18af4c8b6c98` | Open draft; no review decision | [run 45855781546](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45855781546) passed |

CI success is evidence for the tested PR head only. These PRs were unmerged and lack an independent approval decision. The local integration candidate needs its own exact-head hosted CI and independent owner/reviewer acceptance; it must not be substituted for `main` or called an accepted training source.

## Integration findings

### F-BASE-01 — split implementation histories

- **Severity:** High, release-blocking until integrated and independently reviewed.
- **Component:** TASK 006-C3 runtime audit and Objective A v3 corpus/trainer.
- **Reproduction:** Compare E’s tree at `a5844e5` with PR #23 head `b7656ab` and PR #28 head `f122f4e`; inspect `git merge-base --is-ancestor` and changed-file lists.
- **Evidence:** Neither exact head was an ancestor of E. #28 branched from D1, while E descended from D2/D3. E documentation described v3, but its source tree did not contain the v3 implementation.
- **Root cause:** Parallel draft PRs evolved after E had been prepared; passing checks belonged to different source revisions.
- **Remediation:** Local merge commits on this F branch integrate the exact heads. Conflicts in CI required-file/check accounting and roadmap were reviewed and resolved. The PR-specific CI references above remain historical, not F validation.
- **Verification:** `git` ancestry and full local source tree confirm the integration. F-specific hosted CI is pending in the delivery PR.

### F-BASE-03 — race-test synchronization flake

- **Severity:** Medium; verification reliability, not a demonstrated product runtime defect.
- **Component:** Existing process-lock and C3 multiprocess test synchronization.
- **Reproduction:** One full local training-package run failed the process-lock concurrency test because it waited for result file paths to exist before verifying their contents; isolated rerun passed.
- **Evidence:** Failure was `2` complete `contended` records versus expected `3`; workers create result files before writing content. No second journal owner was observed. Follow-up controls passed ten times for the process-lock test and three times for the C3 race test.
- **Root cause:** File-existence polling can observe a result file while the writer is still publishing its short content.
- **Remediation:** F changes both wait predicates to await the complete `contended` contents before assertions; no runtime algorithm changed.
- **Verification:** Focused repeats passed; complete `xynoos-training` debug/release suites passed on `00905b2abc11d46002da57f2607a3ec18a73f956`. Exact-head hosted validation is still required before this candidate is considered green.

### F-BASE-02 — no accepted training revision

- **Severity:** Critical for dispatch.
- **Component:** Release governance/source identity.
- **Reproduction:** Inspect PR states/reviews and compare `main` SHA with the F candidate.
- **Evidence:** `main` remains TASK 005 baseline; all TASK 006 dependency PRs remain drafts, with no review decision. The candidate has not been merged or accepted.
- **Root cause:** Deliberate staged review; CI alone is not owner approval.
- **Remediation:** Separate independent reviewer must accept numerical/runtime/scientific evidence, then owner must approve the exact immutable run bundle and normal protected integration process.
- **Verification:** F preserves `main`; no merge was performed. Exact F PR/CI references are recorded in the final handoff when available.

## Source and privacy boundary

No task-006 training workflow was dispatched. No Colab runtime, paid compute, external corpus, private model-weight upload, or PR merge occurred. `main` was not changed. Historical TASK 005 foundation remains the only accepted baseline.
