# TASK 006-R — Process-Level Runtime Verification

**Result: PASS for the bounded local Linux process contract; provider/SSH persistence not tested.**

The tests run a real `xynoos-train` executable, not a mock Trainer. They use only the small project-authored TASK 004 dataset fixture, the existing tokenizer artifact and fit record, explicit bounded limits, and the accepted `[1,32]` model input. No TASK 006-D configuration or experimental run is dispatched.

## Process path

| Check | Method | Observed result |
|---|---|---|
| Executable and non-allocating validation | `xynoos-train validate --config CONFIG --artifact-root ROOT` | Reports fixed model ID, 914,400 parameters, source/data/tokenizer/split identities and explicit limits without creating a checkpoint. |
| Detached launch | `xynoos-train run --config CONFIG --artifact-root ROOT --detach` | The controller exits after spawning the native child through `setsid`. A bounded PID handshake returns the runtime PID; the test verifies its Linux session differs from the controller session. |
| Controller exit and progress | Integration test waits for actual JSONL `optimizer_update` and `checkpoint` events. | Both appear while the controller has exited; a checkpoint exists and the runtime remains live. Standard output is detached and stderr is redirected to the artifact root. |
| Concurrent launch | A second detached resume is attempted while the first process is active. | Rejected with exit code `2`; active runtime PID and journal admission remain owned by the first process. |
| Graceful stop and status | `xynoos-train stop --artifact-root ROOT`; then read-only `status`. | `stop.request` is observed at a safe boundary; final report and checkpoint are published; lock/PID are removed; status is `finished`. |
| Checkpoint inspection | `xynoos-train inspect-checkpoint --config CONFIG --artifact-root ROOT` | Bounded checkpoint v1 integrity and metadata decode succeeds; no training state is changed. |
| Same-lineage resume | `xynoos-train resume --config CONFIG --artifact-root ROOT --detach` after the graceful stop. | It loads the prior checkpoint and valid journal, keeps the same experiment ID, and increments checkpoint sequence. The existing `stop.request` is intentionally latched, so this resume immediately reaches a safe requested stop and does not perform another optimizer update. Remove only `stop.request` before a later intended continuation. |
| Abrupt process failure | A second detached run is killed with `SIGKILL` after a committed checkpoint/update. | The process exits; checkpoint, journal, stale lock and runtime reservation remain. Status reports interruption and positive reserved runtime. Resume charges the outstanding reservation and does not reduce prior cumulative work. |
| Corrupt-lineage recovery | Copy the committed checkpoint/sidecar to a test root, corrupt its journal, then run `resume`. | Exit code `2`; an atomic `failure-report.json` identifies trainer initialization failure; checkpoint SHA-256 remains byte-identical. |
| Cleanup | Normal stop and recovery use bounded waits; test guards kill any unexpected surviving child. | Normal execution removes the lock/PID and no training process remains. Abrupt failure leaves only stale on-disk recovery markers, not a live process. |
| Source admission | CLI validation rejects extra JSON fields, unsupported source kinds, wrong dataset/tokenizer identities, and an over-budget metadata fixture. | All rejection paths return exit code `2`; no checkpoint/model allocation occurs. The valid small standard stream passes. |

The debug detached process integration case passed in 17.05 seconds. Its child reported a 42,439,416-byte owned-memory reservation peak, 30,682,296-byte autograd ledger peak, and 36,007,936-byte Linux process `VmHWM`. The isolated release process case passed in 1.77 seconds and reported 34,705,408-byte `VmHWM`. `VmHWM` is measured per native training process; controller memory is not added to it, and application reservations remain distinct from actual RSS.

The integration suite has two active process tests. The first exercises detached lifecycle, same-lineage resume, concurrent launch rejection, abrupt exit recovery, corruption rejection, and checkpoint preservation. The second exercises strict source/configuration admission and the pre-model memory rejection path. They are included in the 150 active debug and release checks.

## Runtime and failure behavior

The runtime returns `0` for completed or safely stopped bounded runs and `2` for invalid configuration or runtime failures. Training errors after initialization attempt the normal final checkpoint and report. Startup and recovery errors publish a separate `failure-report.json` when the artifact filesystem accepts writes. If it cannot, the executable prints a nonzero stderr diagnostic; it does not promise report persistence on a fully failed filesystem.

The logger now propagates `Logging(stage)` errors. Tests inject write and flush failures at the Trainer boundary before optimizer commit, a metric failure after a successful optimizer commit, and final-report publication failure. Precommit failure discards pending gradients and retains the prior committed update. A postcommit metric error does not label that update as rolled back; its committed parameter/optimizer state and counters are captured in the safe checkpoint/report where writes succeed. Checkpoint publication continues to distinguish pre-rename failure from `PublicationDurabilityUncertain` after rename.

Resource and time limits are enforced at existing safe points. An explicit stop request is a file, not a POSIX signal handler. `setsid` gives the child a separate Linux session and all stdio is detached from the controller. A blocked synchronous kernel or filesystem operation cannot be preempted, and the workflow’s outer kill grace may end a process abruptly if that operation does not return.

## Manual GitHub Actions workflow

`.github/workflows/train-cpu.yml` has only `workflow_dispatch`; push and pull-request events cannot execute it. It requires an explicit authorization boolean, repository-relative config path, exact checked-out SHA, clean source checkout, standard source kind, and explicit ceilings of at most 24 updates, 768 consumed/supervised tokens, 8B work, 64 MiB run-owned memory, 4 MiB data allowance, 120-second cumulative runtime, 90-second per-attempt runtime, and checkpoint/validation cadence of six. It has a 10-minute job timeout, 240-second outer process timeout with 30-second kill grace, non-cancelling per-ref concurrency, bounded JSONL/checkpoint/report verification, a 2 MiB report-upload cap, three-day report retention, nonzero failures, and runner cleanup.

The workflow was syntax- and source-reviewed and passed Foundation CI on the implementation head in [run 45753759238, job 139687426641](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45753759238). It was **not dispatched**. It uploads report, logs, handoff, and integrity metadata only; it neither uploads the checkpoint/model weights nor offers post-run private checkpoint retrieval. That persistence route needs separate owner authorization before an actual experiment can run there. Workflow-side Python checks do not add a Python dependency to the native executable or training process.

## Limits of this evidence

- This establishes local Linux detached-process behavior and controller-exit survival only. It does not test terminal/SSH network loss, Colab CLI disconnection, a remote worker, NFS/object storage, or provider cleanup.
- The artifact directory is trusted local storage. Journal SHA-256 is unkeyed and does not prevent deliberate replacement or deletion.
- `status` is read-only, but it validates current source/data identities first. If those inputs disappear, it may report an admission error rather than reconstructing missing artifacts.
- A graceful stop leaves a latched `stop.request`. Removing that file is an explicit operator action before resuming the same lineage; it does not clear or rewrite cumulative journal counters.
- Workflow validation and artifact checks do not constitute authorization to dispatch TASK 006-D. The single-batch validation and missing baseline/test evaluation paths remain proposal gates.
