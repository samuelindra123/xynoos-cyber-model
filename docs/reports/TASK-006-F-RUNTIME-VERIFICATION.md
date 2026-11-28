# TASK 006-F — Runtime Independent Verification

**Outcome: local Linux runtime controls passed; separate reviewer acceptance and remote-provider lifecycle remain open.** Runtime claims below apply to the tested Linux/POSIX local filesystem behavior, not arbitrary network filesystems or power-loss semantics beyond the documented sync protocol.

## Evidence

At exact C3 audit head `b7656abbc38703b3a34652cc5c7f9435a2b0c32c`, `cargo test -p xynoos-training --locked --offline -- --show-output` passed the training library, CLI, process-lock, C3, and doctest suites. Exact-head hosted checks passed in [PR #23 run 45824251360](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45824251360). Added for F, the auditor-authored `task006f_runtime_independent.rs` independently passes in debug and release on the integrated source. During F's broader local run, an existing process-race test exposed a test-only synchronization flake; F fixed it and the focused test passed ten consecutive lock-process runs and three C3 race runs.

The added SIGKILL control starts a subprocess that owns the journal, confirms another process receives `RunLockContended`, kills the owner with signal 9, confirms kernel lock release and stable `.run.lock` inode, then resumes and verifies precharged tokens/work remain charged and the reserved attempt runtime is charged in full. The second control corrupts the journal checksum and then removes the journal; both resume attempts fail closed without rewriting corrupt bytes or recreating missing state.

Prior C3 evidence additionally exercises repeated multiprocess admission races, a separate Python `fcntl.flock` owner, process death near lock acquisition, PID reuse detection using boot identity and process start time, stop/status/resume, sidecar publication fault boundaries, concurrent launch, checkpoint integrity, and cleanup. Findings C2-F01 through C2-F03 and C3-F01 were technically closed by R2/C3 for the stated local platform.

## Finding F-RUN-01 — crash recovery and journal integrity

- **Severity:** High; verified closed for the tested local filesystem path.
- **Component:** RunJournal, advisory lock ownership, cumulative accounting.
- **Reproduction:** `cargo test -p xynoos-training --test task006f_runtime_independent --locked --offline -- --show-output`; repeat with `--release`.
- **Evidence:** Both profiles passed. Tests observe contention before SIGKILL, kernel release after SIGKILL, stable lock-file inode, persisted token/work counters, full reserved runtime charging, checksum rejection, fail-closed missing-journal rejection, and no repair/recreation.
- **Root cause:** Earlier stale PID-file ownership could permit unlink/recreate races; current design uses a persistent kernel-backed exclusive lock inode and authenticated journal state.
- **Remediation:** R2/C3 persistent lock protocol; fail closed on absent/corrupt journal during resume; F adds direct process-kill/corruption controls.
- **Verification:** Passed locally in debug and release. Exact F hosted CI remains necessary for the final candidate.

## Finding F-RUN-02 — platform and independent acceptance boundary

- **Severity:** High for production dispatch.
- **Component:** Runtime portability and release assurance.
- **Reproduction:** Run the local C3/F process tests on a different provider/filesystem or ask an independent reviewer to inspect the frozen R2 diff and test evidence.
- **Evidence:** Local tests and C3 report explicitly scope advisory locking to local Linux/POSIX semantics. No separate human reviewer has accepted R2/C3; the same autonomous author implemented and audited R2/C3.
- **Root cause:** No independent organization/reviewer sign-off and no authorized provider lifecycle exercise.
- **Remediation:** Obtain separate code review. After storage/provider authorization, exercise process termination, restore, resume and cleanup on that actual provider before relying on it.
- **Verification:** Not yet performed; therefore this release gate remains open.

## Runtime behavior matrix

| Requirement | Result | Evidence or limitation |
|---|---|---|
| OS exclusive journal ownership and concurrent admission | Pass, local Linux/POSIX | C3 repeated process race and independent `fcntl` contention; F direct contention during owner-held journal |
| Stale-owner recovery, PID reuse and process identity | Pass within tested local contract | PID start time plus boot ID; C3 process tests |
| Crash/SIGKILL lock release | Pass | F subprocess SIGKILL test |
| Atomic sidecar publication | Pass for tested local fault boundaries | R2/C3 staged publish/sync tests; requires local filesystem semantics |
| Corrupt and missing journal handling | Pass, fail closed | F checksum corruption and missing-journal test |
| Cumulative resources / stop semantics | Pass within tested suite | Monotone precharge, safe-point stop, full reservation accounting; abrupt stop does not refund reservation |
| Graceful/abrupt resume equivalence | Pass for local deterministic state | Existing checkpoint/resume tests; remote resume remains unverified |
| Orphan process and concurrent same-lineage execution | Pass for tested process controls | C3 subprocess admission, stop and cleanup tests |
| Provider disconnection survival and cleanup | Unverified | No provider runtime or transfer was authorized or started |

## Finding F-RUN-03 — multiprocess test observed a partially written result

- **Severity:** Medium; test reliability issue, not a demonstrated runtime defect.
- **Component:** `lock_process::repeated_multiprocess_recovery_has_one_owner_per_lineage` and the analogous C3 concurrency test.
- **Reproduction:** Run the full training package while worker processes contend. The pre-fix test waited until three result paths existed, then immediately read their contents; `fs::write` creates the path before all bytes are visible. One run observed only two complete `contended` strings and failed `left: 2, right: 3`. The isolated unmodified test passed, confirming the timing sensitivity.
- **Evidence:** Failure occurred once at the assertion after the existence-only wait. No lock double-admission or second owner was observed. After the fix, the lock-process test passed 10 repeated invocations and the six-process C3 race test passed 3 repeated invocations; F SIGKILL/journal controls passed in debug and release.
- **Root cause:** Test synchronization used path existence instead of a complete result record.
- **Remediation:** Change both race-test wait predicates to require the full `contended` contents before asserting the ownership result. No runtime code or resource policy changed.
- **Verification:** Repeated focused controls pass. A complete post-fix training-package debug/release result and exact F hosted CI remain part of final integration verification.

Do not remove `.run.lock` during recovery. Preserve journal/checkpoint state and use `xynoos-train status`; resume only after the previous process is gone and artifacts validate.
