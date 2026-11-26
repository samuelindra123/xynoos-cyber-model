# TASK 006-C3 — Independent Re-verification of TASK 006-R2

**Audit outcome: R2 TECHNICAL FINDINGS CLOSED; C3 HOSTED AND INDEPENDENT-REVIEW GATES PENDING.** The three C2 findings and C3-F01 are closed in the current R2 revision. C3's auditor-authored tests pass in debug and release. The corrected R2 workflow passed on the exact R2 branch revision. The C3 branch adds five audit checks and updates the total to 170; its exact hosted run remains pending. A separate human/agent review is still required for independent acceptance.

This audit was performed on a separate audit branch after freezing R2. It adds auditor-authored process and fault-injection tests in [the C3 test module](../../crates/xynoos-training/tests/task006c3_independent.rs). The same autonomous agent authored both R2 and this audit. No second human or independent agent reviewed the implementation, so this is a separate verification pass, not independent organizational certification.

## Audit identity and evidence boundary

| Item | Evidence |
|---|---|
| Repository | `samuelindra123/xynoos-cyber-model` |
| R2 implementation under review | `e6336e3af48ab5be7a55c99a90b5b84c94d0992d` |
| C3 process/fault tests | `e1f8b2c`, `e447f51`, and `3ab09a1` on `audit/task-006-c3-runtime-reverification` |
| PR #21 | Open draft; no merge performed. |
| Corrected exact R2 hosted run | Foundation CI `45772399582`, job `139747896665`, checked out exact SHA `21d9990760380195b9d9379b6d43f848ad6c17df`. All 29 steps succeeded, including corrected direct-test count, debug and release suites, all oracles and resource checks. The preceding run `45771271381` on `e6336e3` correctly exposed a stale expected count (164 versus 165 observed); `21d9990` corrected the assertion and CI passed. C3 expects 170 after five added audit tests; its exact hosted evidence remains pending. |
| Source status | The C3 working tree was clean for the recorded test commands. |
| Training/compute boundary | No workflow dispatch, TASK 006-D experiment, external dataset acquisition, paid compute, Colab runtime, or public artifact upload occurred. One bounded Objective A CLI execution is part of the sidecar restart software test and is not a research experiment. |

The preserved C2 finding and evidence reports are included in this branch: [C2 re-verification](TASK-006-C2-INDEPENDENT-REVERIFICATION.md) and [C2 test evidence](TASK-006-C2-TEST-EVIDENCE.md).

## Re-verification results

### C2-F01 — stale-lock double admission: closed for supported local Linux filesystems

R2 replaces PID-file deletion with persistent lock inodes and kernel-held exclusive locks. Journal recovery never unlinks `.run.lock`; the kernel releases the handle when its owner exits. Separate launch and lineage locks serialize admission and transfer ownership to the runtime only after the child holds the lineage lock.

The C3 test module independently coordinates six OS processes against one lineage over 12 repeated rounds with stale legacy PID text in the lock inode. Every round has exactly one journal owner and five typed contention results. A separate Python standard-library `fcntl.flock` process also blocks `RunJournal::open_existing`; the journal opens after the independent process releases the file lock. Existing R2 process tests kill owners immediately after raw lock acquisition and after journal recovery, then restart against the persistent lock path.

The local advisory-lock contract assumes a trusted Linux/POSIX filesystem with working advisory locks. Remote filesystems and hostile directory replacement races remain outside the documented contract.

### C2-F02 and C3-F01 — source identity: closed for CLI and public Rust APIs

The CLI now checks the canonical GitHub repository, compiled build revision, checkout `HEAD`, tracked modifications, visible untracked files, and ignored implementation files under the workspace source paths. The runtime binds repository and revision into the immutable config digest and reports.

The public API now seals build provenance inside `attest_source_checkout`; callers provide only the configured repository and revision. It returns an opaque `SourceIdentity` with private fields. `RunIdentity::from_standard_dataset` requires that value, while the unchecked constructor is crate-private. A compile-fail doctest verifies external code cannot forge `SourceIdentity` directly.

The independent C3 clone-based test accepts a clean clone at the compiled SHA and rejects wrong repository, wrong revision, modified tracked source, and ignored untracked implementation source. The CLI regression suite exercises configuration admission at the same source identity.

### C2-F03 — initial sidecar publication: closed for tested local Linux filesystems

R2 stages the sidecar in a mode-0600 sibling file, syncs it, publishes it atomically without replacing an existing identity, syncs the directory, and removes the staging link. Startup removes orphan staging files and accepts only a complete matching sidecar with no journal or progress state.

The separate C3 fault test applies `RLIMIT_FSIZE=0` to the CLI child. Linux terminates it on the first staging write, leaving a zero-byte temporary file and no canonical sidecar. A clean restart removes the orphan, publishes a valid sidecar, completes the bounded Objective A software test, and writes a final report. R2 unit tests inject failures across each stage/publication/cleanup boundary and verify pre-publication cleanup, post-publication completeness, and existing-artifact preservation.

## Scope limits

- R2's 32 training-library tests, 4 CLI unit tests, 3 CLI process tests, 5 lock-process tests, and 1 compile-fail doctest passed locally in debug and release at `e6336e3…`.
- The five C3 auditor-authored tests passed locally in debug and release after the API correction. Checkpoint semantics/publication, journal monotonicity, logging failures, optimizer transactions, resume equivalence, and resource accounting are also covered by existing package regressions and prior auditor-authored TASK 006-C controls.
- The corrected R2 hosted run `45772399582` passed all steps on exact SHA `21d9990`. C3's workflow count is 170 after adding five C3 checks; that exact branch run remains pending.
- Local detached-session tests do not establish survival after GitHub runner cancellation, SSH disconnect, Colab CLI disconnect, remote artifact transfer, or provider cleanup.
- Advisory-lock crash recovery depends on local Linux kernel/filesystem behavior. Power loss beyond requested file/directory syncs, network filesystems, and hostile directory races are not claimed.
- Advisory locks are verified only for local Linux/POSIX filesystem behavior, not remote filesystem semantics or hostile directory replacement.

## Readiness recommendation

**R2 technical remediation passed corrected hosted CI and C3's local checks.** C3's five audit tests and the full training package pass locally in debug and release. C3 hosted CI and review by someone separate from the implementation author remain necessary before treating this audit as independent acceptance.
