# TASK 006-C3 — Test Evidence

This report records the separate audit tests and the exact R2 hosted evidence. Test fixtures use temporary repositories, artifact roots, and project-authored Objective A data. The sidecar restart case is a bounded software recovery check, not a TASK 006-D experiment.

## Auditor-authored tests

| Command | Result | What it establishes |
|---|---|---|
| `cargo test -p xynoos-training --test task006c3_independent --locked --offline -- --test-threads=1` | 5 passed, 0 failed; 18.58 seconds at C3 head `d9e3b12` | Independent Python `flock` process blocks journal open; 12 rounds of six-process contention admit one owner; sealed source attestation accepts a clean clone at the compiled SHA and rejects wrong claims and dirty/ignored source; real file-size fault interrupts sidecar staging and restart recovers. |
| `cargo test --release -p xynoos-training --test task006c3_independent --locked --offline -- --test-threads=1` | 5 passed, 0 failed; 7.16 seconds at C3 head `d9e3b12` | Same auditor-authored controls pass with optimized native binaries. |
| `cargo test -p xynoos-training --test lock_process --locked --offline -- --test-threads=1` | 5 passed, 0 failed; 0.24 seconds | Repeated multiprocess recovery, crash after journal ownership, immediate crash after OS lock acquisition, persistent lock inode, restart recovery, and typed contention/recovery errors. Child test output is suppressed so CI counts direct suite results accurately. |

The independent sidecar fault uses Python's standard `resource` module to set `RLIMIT_FSIZE` to zero for the CLI child. The expected `SIGXFSZ` termination leaves no `experiment.json` and one zero-byte `.experiment.json-init-*.tmp`; restart removes that file and writes a complete final report. The recovery CLI invocation uses the tiny repository-authored Objective A fixture and does not use an external dataset.

## Frozen R2 regression and hosted run

R2 local package execution after the implementation commit recorded:

| Target | Result |
|---|---|
| `xynoos-training` library | 32 passed |
| `xynoos-train` binary unit tests | 4 passed |
| `cli_process` integration tests | 3 passed after correcting test output capture and checking the new typed contention message |
| `lock_process` integration tests | 5 passed |

The corrected R2 Foundation CI run `45772399582`, job `139747896665`, checked out exact SHA `21d9990760380195b9d9379b6d43f848ad6c17df`; all 29 workflow steps succeeded, including the corrected direct-test count, debug and release suites, all numerical/data oracles, and resource checks. The prior run `45771271381` at `e6336e3` exposed the stale expected total (164 expected, 165 observed); the workflow assertion was corrected without changing tolerances or checks. C3 Foundation CI run `45772922015`, job `139749599697`, passed all 29 steps on exact SHA `228c0560224fb86e7f7f97343f708620d576f542`; the 170 direct checks passed for both debug and release (138 preserved + 12 prior R + 15 R2 + 5 C3).

The earlier C2 test-evidence report records the independent optimizer/checkpoint controls and failure reproductions used before R2. This C3 pass did not substitute implementation-authored tests alone for independent process and sidecar reproductions.

## Evidence not claimed

- No Colab, TPU, GPU, SSH, GitHub training workflow, or paid compute was started.
- No claim is made about detached work surviving provider disconnection or remote artifact transfer.
- No training-quality claim is inferred from the tiny Objective A fault-recovery test.
- C3 is not independent human certification; the implementation and audit were authored by the same agent.
