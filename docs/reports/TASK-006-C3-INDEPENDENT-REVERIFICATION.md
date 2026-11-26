# TASK 006-C3 — Independent Re-verification of TASK 006-R2

**Audit outcome: R2 NEEDS ONE API-LEVEL PROVENANCE CORRECTION.** The three C2 findings are closed for the supported CLI path. The separate C3 pass found that public Rust constructors can still accept an unverified `source_clean=true`; see C3-F01. D1 must wait until that API gap is closed and the corrected revision is re-verified.

This audit was performed on a separate audit branch after freezing R2. It adds auditor-authored process and fault-injection tests in [the C3 test module](../../crates/xynoos-training/tests/task006c3_independent.rs). The same autonomous agent authored both R2 and this audit. No second human or independent agent reviewed the implementation, so this is a separate verification pass, not independent organizational certification.

## Audit identity and evidence boundary

| Item | Evidence |
|---|---|
| Repository | `samuelindra123/xynoos-cyber-model` |
| R2 implementation under review | `3952485160135fe9f76e66845d4e65b49d63c341` |
| C3 process/fault tests | `f979325` and `2df2598` on `audit/task-006-c3-runtime-reverification` |
| PR #21 | Open draft; no merge performed. |
| Exact R2 hosted run | Foundation CI `45768465704`, job `139735057716`, checked out exact SHA `395248…`. Debug and release test suites and all preceding gates passed. The job ended red only at the stale fixed test-count assertion (expected 150). The auditor corrected child test output capture and the count on the C3 branch; that correction still needs hosted evidence on its new SHA. |
| Source status | The C3 working tree was clean for the recorded test commands. |
| Training/compute boundary | No workflow dispatch, TASK 006-D experiment, external dataset acquisition, paid compute, Colab runtime, or public artifact upload occurred. One bounded Objective A CLI execution is part of the sidecar restart software test and is not a research experiment. |

The preserved C2 finding and evidence reports are included in this branch: [C2 re-verification](TASK-006-C2-INDEPENDENT-REVERIFICATION.md) and [C2 test evidence](TASK-006-C2-TEST-EVIDENCE.md).

## Re-verification results

### C2-F01 — stale-lock double admission: closed for supported local Linux filesystems

R2 replaces PID-file deletion with persistent lock inodes and kernel-held exclusive locks. Journal recovery never unlinks `.run.lock`; the kernel releases the handle when its owner exits. Separate launch and lineage locks serialize admission and transfer ownership to the runtime only after the child holds the lineage lock.

The C3 test module independently coordinates six OS processes against one lineage over 12 repeated rounds with stale legacy PID text in the lock inode. Every round has exactly one journal owner and five typed contention results. A separate Python standard-library `fcntl.flock` process also blocks `RunJournal::open_existing`; the journal opens after the independent process releases the file lock. Existing R2 process tests kill owners immediately after raw lock acquisition and after journal recovery, then restart against the persistent lock path.

The local advisory-lock contract assumes a trusted Linux/POSIX filesystem with working advisory locks. Remote filesystems and hostile directory replacement races remain outside the documented contract.

### C2-F02 — CLI source identity: closed for the official CLI path; C3-F01 remains

The CLI now checks the canonical GitHub repository, compiled build revision, checkout `HEAD`, tracked modifications, visible untracked files, and ignored implementation files under the workspace source paths. The runtime binds repository and revision into the immutable config digest and reports.

The independent C3 temporary-repository tests accept a clean canonical checkout and reject wrong repository, wrong configured revision, modified tracked source, and ignored untracked source. The CLI regression suite also exercises strict config admission at the frozen source SHA.

The public library constructors remain a bypass: callers can currently provide `source_clean=true` to `RunIdentity::new` or `RunIdentity::from_standard_dataset` without obtaining a `SourceIdentity` from checkout attestation. That is an open medium-severity finding, even though the official CLI does not use the bypass.

### C2-F03 — initial sidecar publication: closed for tested local Linux filesystems

R2 stages the sidecar in a mode-0600 sibling file, syncs it, publishes it atomically without replacing an existing identity, syncs the directory, and removes the staging link. Startup removes orphan staging files and accepts only a complete matching sidecar with no journal or progress state.

The separate C3 fault test applies `RLIMIT_FSIZE=0` to the CLI child. Linux terminates it on the first staging write, leaving a zero-byte temporary file and no canonical sidecar. A clean restart removes the orphan, publishes a valid sidecar, completes the bounded Objective A software test, and writes a final report. R2 unit tests inject failures across each stage/publication/cleanup boundary and verify pre-publication cleanup, post-publication completeness, and existing-artifact preservation.

## Scope limits

- R2's 31 training-library tests, 4 CLI unit tests, 3 CLI process tests, and 5 lock-process tests passed locally across the clean R2 revision and the C3 audit branch. The exact R2 hosted run reports debug and release suites passed; its only failing step was the stale test-count assertion.
- Checkpoint semantic validation and atomic publication, journal monotonicity, trainer logging failures, optimizer transaction controls, resume equivalence, and resource accounting are covered by the existing package regressions and earlier auditor-authored TASK 006-C controls. C3 re-ran the complete workspace suites on hosted CI at R2 SHA `395248…`; no test failure was reported.
- Local detached-session tests do not establish survival after GitHub runner cancellation, SSH disconnect, Colab CLI disconnect, remote artifact transfer, or provider cleanup.
- Advisory-lock crash recovery depends on local Linux kernel/filesystem behavior. Power loss beyond requested file/directory syncs, network filesystems, and hostile directory races are not claimed.
- `RunIdentity::new` remains public and allows a library caller to self-assert source cleanliness. This is the only confirmed R2 code finding in this pass.

## Readiness recommendation

**Do not advance to TASK 006-D1 yet.** Make source identity opaque to public training APIs: only checkout attestation should create a usable source identity, and dataset-backed run identity construction should require that attestation. Then rerun the relevant API, CLI, debug/release, and exact-head hosted checks. A reviewer separate from the implementation author must still inspect the frozen R2 diff and C3 evidence before treating this audit as independent acceptance.
