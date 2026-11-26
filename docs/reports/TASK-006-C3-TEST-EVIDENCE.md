# TASK 006-C3 — Test Evidence

This report records the separate audit tests and the exact R2 hosted evidence. Test fixtures use temporary repositories, artifact roots, and project-authored Objective A data. The sidecar restart case is a bounded software recovery check, not a TASK 006-D experiment.

## Auditor-authored tests

| Command | Result | What it establishes |
|---|---|---|
| `cargo test -p xynoos-training --test task006c3_independent --locked --offline -- --test-threads=1` | 5 passed, 0 failed; 19.07 seconds | Independent Python `flock` process blocks journal open; 12 rounds of six-process contention admit one owner; source attestation rejects wrong repository/revision and dirty/ignored source; real file-size fault interrupts sidecar staging and restart recovers. |
| `cargo test --release -p xynoos-training --test task006c3_independent --locked --offline -- --test-threads=1` | 5 passed, 0 failed; 8.06 seconds | Same auditor-authored controls pass with optimized native binaries. |
| `cargo test -p xynoos-training --test lock_process --locked --offline -- --test-threads=1` | 5 passed, 0 failed; 0.24 seconds | Repeated multiprocess recovery, crash after journal ownership, immediate crash after OS lock acquisition, persistent lock inode, restart recovery, and typed contention/recovery errors. Child test output is suppressed so CI counts direct suite results accurately. |

The independent sidecar fault uses Python's standard `resource` module to set `RLIMIT_FSIZE` to zero for the CLI child. The expected `SIGXFSZ` termination leaves no `experiment.json` and one zero-byte `.experiment.json-init-*.tmp`; restart removes that file and writes a complete final report. The recovery CLI invocation uses the tiny repository-authored Objective A fixture and does not use an external dataset.

## Frozen R2 regression and hosted run

R2 local package execution after the implementation commit recorded:

| Target | Result |
|---|---|
| `xynoos-training` library | 31 passed |
| `xynoos-train` binary unit tests | 4 passed |
| `cli_process` integration tests | 3 passed after correcting test output capture and checking the new typed contention message |
| `lock_process` integration tests | 5 passed |

The exact R2 Foundation CI run is [45768465704](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45768465704), job `139735057716`, at `3952485160135fe9f76e66845d4e65b49d63c341`. Its log confirms exact checkout, formatting, Clippy, all numerical/data oracles, debug and release Rust test suites, trajectory comparison, and remaining mathematical gates passed. The job failed only because the workflow still expected 150 raw result lines while the R2 branch had 163 direct test checks and child process test harnesses added 80 nested summaries. The C3 branch silences R2 child-worker output and updates the exact count to 168 (163 R2 checks plus five C3 checks). That workflow correction has not yet received hosted CI evidence.

The earlier C2 test-evidence report records the independent optimizer/checkpoint controls and failure reproductions used before R2. This C3 pass did not substitute implementation-authored tests alone for independent process and sidecar reproductions.

## Evidence not claimed

- No Colab, TPU, GPU, SSH, GitHub training workflow, or paid compute was started.
- No claim is made about detached work surviving provider disconnection or remote artifact transfer.
- No training-quality claim is inferred from the tiny Objective A fault-recovery test.
- C3 is not independent human certification; the implementation and audit were authored by the same agent.
