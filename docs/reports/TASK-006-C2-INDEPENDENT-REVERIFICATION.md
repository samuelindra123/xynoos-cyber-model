# TASK 006-C2 — Independent Re-verification of TASK 006-R

**Verdict: REJECTED**

The remediation substantially fixes the TASK 006-C logging, journal, checkpoint, admission, and local detached-runtime gaps. Re-verification found a reproducible stale-lock race that lets two processes acquire one recovered lineage, plus two source-inspection defects in provenance and startup sidecar publication. TASK 006-D is not eligible for separate owner authorization.

This is an independent audit, not owner acceptance. No implementation, existing C evidence, PR, or main-branch file was changed. No workflow was dispatched and no TASK 006-D experiment was run. Test details are in [TASK-006-C2-TEST-EVIDENCE.md](TASK-006-C2-TEST-EVIDENCE.md).

## Audit identity and repository state

| Check | Result |
|---|---|
| Repository / branch | `samuelindra123/xynoos-cyber-model`, `feat/task-006-r-runtime-remediation` |
| Requested head | `999cfce89f62cfaf7a5fc5ab2da1d9c7c1c21afe`; local `HEAD` and `origin/feat/task-006-r-runtime-remediation` match |
| Requested parent | `231d47d2d95f52770fa20f4da93a91be762178d0` is the PR base and `git merge-base`; the head commit's immediate parent is `2b7961ab25be6f81d1ad137433bf4766d32a9ea6`, as expected for this two-commit PR |
| PR #21 | Open, draft, unmerged; base branch `audit/task-006-c-native-training-verification` at the requested base SHA |
| Working tree | Clean at audit start. The only intended final workspace changes are these two uncommitted audit reports. No branch switch or commit was made. |
| Main | GitHub reports `main` at `32eca2b900c4f7a88294f7a24a5dfae0604ae798`; no write operation was performed. |
| Exact-head CI | Foundation CI run `45754218041`, job `139688909833`, completed successfully for PR head `999cfce…`. The job log prints `Tested commit 999cfce89f62cfaf7a5fc5ab2da1d9c7c1c21afe`. |

Foundation CI's checkout step pins `github.event.pull_request.head.sha || github.sha` and checks `git rev-parse HEAD` against that same value. Its exact-head run passed debug and release suites at 150 checks each, oracle and fixture gates, bitwise trajectory comparison, candidate/resource checks, Clippy, formatting, and Rust documentation. This is existing hosted evidence; it is not evidence that the manual training workflow ran.

The repository's default branch is `main`. `.github/workflows/train-cpu.yml` is not present on `main` (GitHub contents API returned 404), and Actions history contains no run for that workflow path. GitHub requires a `workflow_dispatch` workflow to exist on the default branch before it can be manually triggered ([GitHub documentation](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)). Since this PR remains an unmerged draft, the proposed training workflow is not currently dispatchable from this repository state. It must not be merged or dispatched as part of this audit.

## Confirmed defects

### C2-F01 — Stale lock cleanup can admit concurrent recovery

**Severity: High.**

`RunJournal::acquire_lock` in `crates/xynoos-training/src/journal.rs:431-459` uses `create_new`, then reads a stale PID and unlinks the lock path before retrying. It does not verify that the path still names the stale file it inspected. Two processes can both read the stale PID; one can remove the stale path and acquire a replacement lock, then the second can remove that replacement and acquire its own lock. `LockFile::drop` (`journal.rs:353-356`) also removes the path unconditionally. The same stale-PID delete-and-retry pattern appears in `acquire_launch_lock` in `crates/xynoos-training/src/bin/xynoos-train.rs:840-875`.

This was independently reproduced without constructing a model or running training. A temporary, two-process Rust harness seeded a valid journal and a `.run.lock` with a nonexistent PID, synchronized both callers, and staggered their unlink syscall using a temporary `LD_PRELOAD` delay shim. Both separate processes returned successfully from `RunJournal::open_existing` in the same attempt (`OPEN <pid>` from each). The process integration test rejects a second resume while a live process is registered; it does not cover simultaneous stale-lock recovery. Reproduction details and output are in the test-evidence report.

`Trainer::construct` (`crates/xynoos-training/src/trainer.rs:298-318`) acquires this journal lock before model construction, so the race can admit two resumptions of the same checkpoint and lineage. They can both charge from stale in-memory journal snapshots, train from the same committed state, and race journal/checkpoint publication. The per-process memory and work budgets do not prevent combined resource use or duplicate lineage work.

**Separate remediation:** replace PID-file stale deletion as the ownership primitive with a process-shared lock whose ownership is held for the whole run, or implement an inode-checked stale reaper with an atomic ownership protocol. Add a controlled cross-process stale-lock race regression before accepting recovery concurrency. Apply equivalent protection to detached launch admission. Do not accept the current live-lock test as proof of stale-lock safety.

### C2-F02 — CLI identity hard-codes a clean source claim

**Severity: Medium.**

`prepare` in `crates/xynoos-training/src/bin/xynoos-train.rs:568-575` passes `true` as `source_clean` to `RunIdentity::from_standard_dataset`. It validates that the caller supplied a 40-character lowercase commit-looking string, but it does not compare that value with the current checkout or check `git status`. `RunIdentity::validate_static` in `crates/xynoos-training/src/config.rs:228-245` requires the flag to be true and persists it as part of identity/configuration.

The manual workflow separately checks exact `github.sha` and a clean checkout, so this finding is limited to direct CLI/library launch outside that workflow. In that mode, a dirty checkout or unrelated revision can be recorded as clean. The remediation report correctly says the binary does not attest its build provenance, but the hard-coded clean bit still overstates what was verified.

**Evidence class:** source inspection only; no real run was started to create such a checkpoint. **Separate remediation:** either attest the checkout SHA and clean state before setting the bit, or represent the source state as declared/unverified and prevent reports from labeling it clean.

### C2-F03 — Initial sidecar write can strand a fresh artifact root

**Severity: Low.**

`write_sidecar` in `crates/xynoos-training/src/bin/xynoos-train.rs:999-1024` opens canonical `experiment.json` with `create_new` and writes directly to it. Unlike checkpoint, journal, handoff, and final-report publication, it does not write a sibling temporary file and rename it, and it does not remove the file on a write/sync failure. `dispatch` calls it before `execute` (`xynoos-train.rs:143-153`), so the startup failure-report path is not reached for this failure.

On a partial write or failed sync, a later fresh `run` sees a nonempty artifact root and rejects it; `resume` cannot verify a truncated sidecar. Recovery requires manual artifact-root cleanup. This is a source-inspection failure-path finding; no filesystem fault was injected for the sidecar. **Separate remediation:** publish the sidecar through the same create-new temporary, file-sync, rename, and directory-sync protocol and test pre/post-rename errors.

## Requirement-by-requirement traceability matrix

| Requirement | Result | Evidence and limits |
|---|---|---|
| 1. Exact PR head, parent/base, branch, tree, CI checkout SHA | **Pass** | Requested branch/head match local and GitHub. Requested parent matches PR base and merge-base; direct head parent is the preceding implementation commit. Exact-head CI log prints the requested SHA. See repository-state section and test evidence. |
| 2. CLI arguments, strict configuration, experiment identity, source provenance | **Partial** | `parse_args`, `object`, `required`, typed extractors, `prepare`, and sidecar verification reject unknown/duplicate/missing fields and bind config/data/tokenizer identities. Direct CLI falsely sets `source_clean=true` (C2-F02). The workflow's exact revision check is source inspection plus CI YAML validation, not a training-workflow run. |
| 2. Resource admission before model allocation | **Pass for official source** | `prepare` constructs and checks manifest-backed train/validation streams, tokenizer fit identity, combined retained reservation, and config before `Trainer::new`. `Trainer::construct` checks source identity/memory and acquires its run journal before `Transformer::new`. `validate` does not construct a Transformer. Arbitrary in-process `BatchSource` remains a trusted extension. |
| 3. Trainer logging pre-commit and post-commit behavior | **Pass** | Logging errors propagate as typed stages. Microbatch write/flush fault tests show no optimizer commit and retained precharge; optimizer-update logging failure shows the committed state/checkpoint/report is retained and the call fails. Final-report failure test returns a runtime-artifact error. Reproduced by package test run. |
| 4. Missing/corrupt journal and checkpoint reconciliation | **Pass with concurrency defect** | Missing journal, corrupt journal, counter regression, and older checkpoint/newer journal tests pass. Journal monotonicity and full-attempt runtime reservation recovery are covered. C2-F01 means two concurrent processes can defeat the single-writer assumption specifically while reaping a stale lock. |
| 4. Interrupted-run recovery and failed-recovery preservation | **Pass for a single recovery process** | Detached SIGKILL test retains checkpoint/journal and runtime reservation, then resumes same lineage. Corrupt-journal recovery returns failure and preserves checkpoint SHA-256 byte-for-byte. Stale-lock concurrency remains unsafe (C2-F01). |
| 5. Checkpoint v1 semantics and checksum resealing | **Pass within tested mutation matrix** | The mutation helper edits encoded semantic fields and recomputes the terminal SHA-256. The matrix rejects nonfinite/negative state, bias-power mismatch, optimizer/schedule/counter/identity/inventory/cursor/order/accumulation and trailing-data mutations without mutating caller model/optimizer state. Not every possible combined field mutation has a unique vector. |
| 5. Atomic checkpoint publication/durability uncertainty | **Pass under documented local POSIX assumptions** | Injected pre-rename failure leaves the old checkpoint; injected post-rename directory-durability failure returns `PublicationDurabilityUncertain` while the new checkpoint is readable. No guarantee is made for hostile TOCTOU, remote filesystems, or power loss beyond requested syncs. |
| 6. Detached controller exit, setsid, PID, stop, resume, SIGKILL, cleanup | **Pass for tested local Linux path** | Process test checks a different session, progress after controller return, active duplicate resume rejection, graceful stop/status/inspection, same-lineage resume, SIGKILL reservation recovery, stale process cleanup, and process disappearance. Does not establish SSH/Colab/provider-disconnect survival or hostile filesystem safety. |
| 7. Concurrency and race conditions | **Fail** | Live concurrent resume is rejected, but independent controlled stale-lock recovery admits two `RunJournal` holders (C2-F01). PID existence checks, path inspection, and unlink are not a kernel lock. |
| 8. Resource caps | **Pass as application admission/accounting** | `RunLimits`, `TrainingMemoryPlan`, `TrainingRunBudget`, `ResourceLimits`, and candidate policy enforce configured per-run limits. CLI checks concrete source reservations before model allocation. The workflow adds lower pilot caps. These are not OS memory/RSS enforcement. |
| 8. Reserved memory versus actual RSS | **Pass as reporting distinction; no new measurement** | Runtime reports charged run-owned bytes, autograd ledger bytes, and Linux `VmHWM` separately. `TrainingMemoryPlan`/`TrainingRunBudget` are reservation ledgers; `VmHWM` is process high-water. None is a hard RSS cap. The remediation report's sample numbers were not independently remeasured here. |
| 9. Manual workflow authorization/revision/path/caps/artifacts/failure behavior | **Partial, source inspection only** | Explicit boolean authorization, exact `github.sha`, clean checkout, relative symlink-free config path, bounded values/timeouts, report-only upload, integrity check, and cleanup are present. The workflow is absent from default branch, has no run history, and has not been executed. It intentionally does not persist checkpoints. |
| 10. Workflow dispatch and TASK 006-D execution state | **Verified not dispatched** | Actions run history contains no `.github/workflows/train-cpu.yml` execution. No TASK 006-D workflow run or production experiment was started. Bounded tests include tiny deterministic trainer fixtures only. |
| Explicit: multi-batch deterministic validation | **Unresolved** | `Trainer::validate_once` evaluates at most the first validation batch per event. TASK 006-D itself says this is insufficient for rate selection. |
| Explicit: unchanged-weight baseline evaluation | **Unresolved** | Current CLI has no baseline evaluation command/path. |
| Explicit: locked test evaluation | **Unresolved** | The identity includes test split metadata, but training CLI only constructs and evaluates train/validation streams; there is no separately locked test evaluation command. |
| Explicit: multi-batch work-budget recalculation | **Unresolved** | Current one-batch estimate is `7,382,315,232 / 8,000,000,000`, leaving `617,684,768`. Each added 32-token validation batch at each of four events adds `400,000,000`; a third total batch at every event would exceed the cap (`8,182,315,232`). The actual frozen evaluation batch count is not decided. |
| Explicit: private checkpoint persistence | **Unresolved** | Workflow excludes checkpoint upload and removes runner-local state. The proposal requires an owner-approved private persistence route before an experiment can resume after the run. |
| Explicit: actual workflow execution/provider disconnect | **Unresolved** | No workflow execution exists. Local `setsid` tests do not test GitHub runner disconnect, SSH, Colab, artifact transfer, checkpoint persistence, or remote cleanup. |

## Remaining limitations and evidence classification

### Independently reproduced

- Logging writer write/flush failure and pre/post-commit state outcomes through the runtime's fault-injection tests.
- Checkpoint semantic mutation and checksum resealing matrix, pre/post-rename checkpoint publication behavior, and journal recovery cases through the targeted package suite.
- Detached local controller return, safe stop/resume, SIGKILL recovery, and checkpoint preservation after corrupt-journal recovery through temporary-root process tests.
- Stale `.run.lock` race through a separate two-process harness; both `open_existing` calls returned a live journal lock holder.
- Exact PR head, parent/base, current branch, and clean start state via Git; exact CI checkout SHA via hosted job logs.

### Existing CI evidence

Foundation CI run `45754218041` passed for the exact requested head, including 150 debug and release Rust checks, numerical and data oracles, bitwise trajectory agreement, repository hygiene, Clippy, formatting, and docs. CI's YAML validation does not execute `train-cpu.yml`.

### Claims verified by source inspection only

- Strict CLI/config admission, config/data/tokenizer identity binding, and model allocation order.
- Workflow input authorization, exact revision/clean-tree checks, config path constraints, cap checks, timeout/artifact logic, and cleanup.
- Direct CLI `source_clean` misstatement (C2-F02) and non-atomic initial sidecar failure window (C2-F03).
- Resource reservations and cap plumbing. No RSS cap is implemented.

### Unverified claims

- Any execution of `train-cpu.yml`, an actual TASK 006-D run, external provider/SSH/Colab disconnect survival, checkpoint transfer/persistence, or remote cleanup.
- Remediation-report memory and throughput figures as fresh measurements in this audit.
- All hostile filesystem, PID-reuse, power-loss, and remote-filesystem behaviors.

### Outstanding owner decisions

- Whether to authorize a separately scoped fix for C2-F01 through C2-F03 and then request a new independent review.
- Whether/when to accept or merge PR #21; this audit does not accept or merge it.
- How to provide owner-approved private checkpoint persistence for any future D run.
- Whether any future external compute is authorized; nothing in this audit authorizes it.

## TASK 006-D readiness recommendation

**Do not authorize TASK 006-D now.** First close the stale-lock double-admission defect and re-verify it, correct the direct-CLI provenance claim, and prepare/freeze the larger dataset and deterministic multi-batch validation. Add separately bounded unchanged-weight baseline and single locked test evaluation paths, recalculate work for the chosen validation batch count without weakening caps, and select an owner-approved private checkpoint route. Then make the workflow available from the default branch and validate it through an explicitly authorized validation-only invocation. A later training dispatch and any provider-disconnect exercise require separate owner authorization. No 006-D authorization is requested by or implied in this report.
