# TASK 006-C2 — Independent Test Evidence

This file records the commands and observations used by [the independent re-verification report](TASK-006-C2-INDEPENDENT-REVERIFICATION.md). All repository tests that create artifact roots use their own temporary directories. The separate concurrency probe used a new temporary root and did not construct a model or run training.

## Repository and hosted evidence

| Evidence | Reproduction / observation | Result |
|---|---|---|
| Checkout | `git status --short --branch`, `git branch --show-current`, `git rev-parse HEAD`, `git rev-parse HEAD^`, `git merge-base HEAD origin/audit/task-006-c-native-training-verification` | Start tree clean; branch and head match request; direct parent is `2b7961a…`; merge-base/PR base is `231d47d…`. |
| PR metadata | Read-only GitHub PR #21 metadata | Open draft, unmerged; requested head/base match. |
| CI checkout | `gh run view 45754218041 --repo samuelindra123/xynoos-cyber-model --log --job 139688909833` | `Verify checkout revision` prints the expected `999cfce…` SHA. Job conclusion is success. |
| CI gates | Read-only job steps/log | Debug and release each report 150 checks; 101-observation learning trajectory agrees bitwise; all listed foundation/oracle/resource gates succeeded. |
| Workflow history | Read-only Actions API query for runs whose path is `.github/workflows/train-cpu.yml` | No run records returned. `train-cpu.yml` is absent from `main`; its contents endpoint returns 404. No dispatch was attempted. |
| Main state | Read-only GitHub branch metadata | `main` was at `32eca2b900c4f7a88294f7a24a5dfae0604ae798`. No remote write operation occurred. |

## Local package regression run

Command:

```text
cargo test -p xynoos-training --locked -- --test-threads=1
```

Observed result:

```text
unit tests:          28 passed, 0 failed
binary unit tests:    0 tests
process integration:  2 passed, 0 failed
doctests:             0 tests
unit-test duration: 202.21 seconds
process duration:    25.61 seconds
```

The run independently reproduced the checked-in cases below. These are regression tests, not a TASK 006-D research run. One bounded Objective A fixture test performs two deterministic optimizer updates; no multi-run sweep or experiment dispatch occurred.

### Trainer logging failures

Reproduced by these passing tests in `crates/xynoos-training/src/trainer.rs`:

- `microbatch_log_write_failure_discards_partial_window_and_preserves_precharge`
- `microbatch_log_flush_failure_stops_before_optimizer_commit`
- `postcommit_metric_log_failure_keeps_committed_update_and_checkpoint`
- `final_report_publication_failure_returns_nonzero_runtime_error`

Observed contract: precommit failures leave completed updates, committed cursor, and committed tokens unchanged while attempted tokens remain charged. Postcommit metric failure retains the completed update and checkpoint in the final report and returns an error. A final-report publication failure returns a nonzero runtime-artifact error.

### Journal recovery

Reproduced by these passing tests in `crates/xynoos-training/src/journal.rs` and `trainer.rs`:

- `admitted_work_survives_crash_and_runtime_reservation_is_not_refunded`
- `checksum_valid_invalid_counter_progression_is_rejected_read_only`
- `older_checkpoint_cannot_refund_journal_precharges_or_runtime_reservation`
- `resume_and_status_fail_closed_when_journal_is_missing`
- `simultaneous_live_lineage_open_is_rejected`
- `resume_with_missing_journal_preserves_checkpoint_and_does_not_recreate_lineage`

Observed contract: missing/corrupt/regressed state fails closed; an intact journal retains precharges newer than its checkpoint; a persisted attempt reservation is charged on recovery. The simultaneous-open unit test covers a live lock owner, not two processes concurrently reaping the same stale lock.

### Checkpoint semantics and publication

Reproduced by these passing tests in `crates/xynoos-training/src/checkpoint.rs`:

- `checksum_valid_semantic_matrix_rejects_state_identity_order_and_cursor_mutations`
- `auditor_semantic_mutation_with_recomputed_integrity_is_rejected`
- `publication_failures_distinguish_pre_rename_from_post_rename`
- `streams_round_trip_and_reject_corruption_without_mutating_a_model`
- `rejects_noncanonical_fields_truncation_oversize_and_symlink_artifacts`

The semantic helper changes bytes and recomputes the checkpoint SHA-256 before decode. The decoder rejects resealed semantic mutations; valid caller model revision and optimizer step remain unchanged. The publication test confirms the old checkpoint remains readable after pre-rename failure and that post-rename durability uncertainty reports `PublicationDurabilityUncertain` while the newly renamed checkpoint is visible.

### Detached lifecycle and failed recovery

Reproduced by passing `detached_process_survives_controller_exit_supports_stop_status_resume_and_crash_recovery` in `crates/xynoos-training/tests/cli_process.rs`:

1. Launches `run --detach`; checks the child has a distinct session and observes update/checkpoint/log progress after the controller returns.
2. Tries a concurrent detached resume while the first PID is live; it is rejected with exit code 2 and “already active”.
3. Sends the stop request and checks report/status/checkpoint inspection; resumes the same experiment identity.
4. Starts a second lineage, waits for progress, sends `SIGKILL`, checks the journal reservation and stale lock, then resumes.
5. Copies the recovered checkpoint into a separate temporary root, corrupts that root's journal, and verifies failed recovery preserves checkpoint SHA-256 byte-for-byte.

This test uses isolated fixture datasets and artifact roots under its `Scratch` temp directory. It does not cover a simultaneous stale-lock cleanup race, arbitrary hostile filesystem changes, or remote provider disconnect.

## Independent stale-lock race probe

The source-level race in `RunJournal::acquire_lock` was independently exercised outside the repository tree:

1. A temporary Rust harness linked the public `xynoos-training` crate and created a valid journal in a new temporary artifact root.
2. The harness replaced `.run.lock` with PID `2147483647` (confirmed absent under `/proc`). Two independent worker processes waited on a shared gate and then called `RunJournal::open_existing` with the same root, digest, default checkpoint counters, and limits.
3. A temporary C `LD_PRELOAD` shim delayed the first worker's `.run.lock` unlink by 25 ms and the second worker's by 250 ms, widening the read/remove/recreate race without modifying repository code.
4. Both processes returned an `Ok(RunJournal)` result in the same attempt:

```text
REPRODUCED attempt=0: OPEN <worker-a-pid>
OPEN <worker-b-pid>
```

The only intended observable was lock admission; neither worker called `begin_attempt`, constructed a Transformer, or performed a training update. Temporary code and artifacts were outside the repository and were not committed. This reproduces violation of the journal's single-holder invariant; it does not measure downstream checkpoint corruption. The path-level consequence follows from `Trainer::construct` acquiring the journal before `Transformer::new`.

## Source inspection and non-execution checks

The following were audited from source; they are not claimed as execution results:

- `parse_args`, strict JSON object/required-field helpers, `prepare`, `TrainingConfig`, identity validation, concrete manifest/tokenizer admission, and allocation order.
- `.github/workflows/train-cpu.yml` authorization boolean, exact revision and clean status checks, config path constraints, caps, timeout, artifact list, and cleanup.
- `RunIdentity::from_standard_dataset` called with `source_clean=true` and the direct CLI's lack of checkout SHA/dirty-tree attestation.
- `write_sidecar` writing the canonical file directly without temporary publication/cleanup.
- Separate charged-memory, autograd-ledger, and `VmHWM` telemetry; none is an OS RSS cap.

No `train-cpu.yml` dispatch, TASK 006-D experiment, checkpoint upload, Colab/GPU allocation, or provider-disconnect exercise took place.
