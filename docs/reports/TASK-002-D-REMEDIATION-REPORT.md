# TASK 002-D — Numerical Remediation Report

## Status and scope

Session B's remediation was submitted for independent review by Session A through draft PR #2. The TASK 002-C audit found one confirmed implementation issue, C-001, and validation gaps in T-15 and T-16. These changes resolve C-001 for T-05 and T-06, demonstrate rejection of malformed fixture input for T-15, and pin a reproducible Rust toolchain for T-16. See the [audit and traceability reports](https://github.com/samuelindra123/xynoos-cyber-model/tree/51c4d1cf2faa3b3b639c3e18d6fa79fc8a5515b6/docs/reports), [remediation handoff](https://github.com/samuelindra123/xynoos-cyber-model/blob/9f9dd8a2750e74e196feeac8defde4bd2f3d78ef/docs/design/TASK-002-D-REMEDIATION-HANDOFF.md), and [PR #2](https://github.com/samuelindra123/xynoos-cyber-model/pull/2).

Work continued from `70045ba2dfdd65ad9edbe7ba3122d815d28c914a` on `feat/task-002-b-numerical-foundation`. The C-001 implementation and nine capacity tests were already present in `56c874b90374804e9b34faecb10d36f3dda386fc`; CI passed on that initial head. Commit `8a3719f9ba8d0b9c847a87189b97a96318d893e5` added project-boundary validation, negative controls, toolchain pinning, and contract updates. See [initial CI](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45530208545).

## C-001 and the capacity contract

The C-001 decision is to reject excess retained capacity. `TensorLimits` limits both `len * size_of::<f32>()` and `capacity * size_of::<f32>()` to 64 MiB per tensor, with support for lower limits. `from_vec` validates payload and length, then checks capacity before copying the shape. An accepted tensor retains the original allocation, including spare capacity within budget. On failure, the consumed `Vec` is dropped. This avoids copying values and reports rejection through `RetainedCapacityLimitExceeded { bytes, limit }`. Capacity multiplication overflow produces `RetainedCapacityByteCountOverflow`. Newly collected buffers reserved with `try_reserve_exact` are checked against their reported capacity before values are written. See the [tensor API contract](../../crates/xynoos-tensor/README.md) and [compute policy](../COMPUTE_POLICY.md).

Ten capacity tests pass in both profiles. They cover pointer-checked zero-copy adoption, capacities at and above the budget, empty vectors with and without capacity under a zero-byte budget, scalars, length-error precedence, arithmetic overflow, and constructor and operation output capacity. Additional tests accept a synthetic capacity exactly at 64 MiB and reject one `f32` more using arithmetic checks. Requests above the ceiling are clamped to the default maximum. Large-boundary cases use synthetic counts; adoption cases use small vectors.

## Oracle evidence

CI's Python 3.12.3 regenerated the fixture byte-for-byte with seed `0x58594e4f`, 295 cases, and 2,670 output values. Fixture SHA-256: `cdfde7ba9f44f46ba7e757157019730e1124bcd966e796b73286bddf3894fa0f`. Finite comparisons use `atol=1e-6` and `rtol=1e-5`; NaNs are compared by classification and infinities by sign.

The same Rust fixture consumer runs the complete fixture, a hand-calculated `1 + 2 = 3` positive control, and six independent in-memory mutations. The harness checks expected diagnostics on rejection. All six diagnostics were observed in debug and release CI logs: [job log](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45540891961/job/138958846528).

| Record mutation | Observed diagnostic |
|---|---|
| Remove the last field | `line 2: invalid fixture record` |
| Replace `add` with `unknown` | `line 2: unknown operation unknown` |
| Replace output word with `not-hex` | `invalid binary32 fixture value` with `InvalidDigit` |
| Change scalar output shape `[]` to `[1]` | `line 2: add output shape` |
| Add a second expected output value | `line 2: add output count` |
| Change expected value 3 to 4 | `line 2: add output 0: expected 4.0, got 3.0` |

## Toolchain and GitHub Actions results

`rust-toolchain.toml` pins Rust 1.100.0 with the minimal profile, rustfmt, and Clippy. A local installation attempt showed a deprecation warning for implicit installation through `rustup show`; the workflow therefore runs `rustup install --no-self-update` at the repository root. CI verifies `rustc` against the pin, records tool versions, and runs locked Cargo commands offline after installation. The workflow has one GitHub-hosted Ubuntu 24.04 job, a 15-minute timeout, at most two Cargo build jobs, read-only `contents` permission, and cancellation of superseded runs.

Run `45540891961` passed all steps. Its checkout log records `8a3719f9ba8d0b9c847a87189b97a96318d893e5` and `rustc 1.100.0 (0ef1bc386 2026-11-09)`. See [the run](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45540891961) and [PR #2 checks](https://github.com/samuelindra123/xynoos-cyber-model/pull/2/checks).

| Check | Result |
|---|---|
| Checkout revision, required files, Markdown, and YAML | Passed |
| Privacy and artifact policy | Passed for 44 tracked files |
| Rust pin, `cargo metadata --locked --offline`, and `cargo fmt --all -- --check` | Passed |
| Clippy with warnings denied | Passed |
| Oracle regeneration | Passed: 295 cases and 2,670 outputs |
| Debug workspace tests and doctests | Passed: 13 unit, 10 contract/property, 8 oracle, 1 doctest |
| Release workspace tests and doctests | Passed: 13 unit, 10 contract/property, 8 oracle, 1 doctest |
| Rust documentation build | Passed |

## Remaining limitations

The tensor budget covers numerical capacity reported by `Vec`. It excludes shape metadata, allocator overhead and hidden rounding, aggregate live tensors, RSS, and OS overcommit. Tests intentionally use small inputs and synthetic overflow cases. Matmul work can be O(mkn), so callers remain responsible for bounding computation.

The oracle provides evidence for selected fixtures and tested parser rejection paths. Python `math.fsum` and Rust's sequential binary64 accumulation may differ under extreme cancellation. These tests establish neither universal accuracy nor bitwise equivalence across targets. The Rust toolchain is pinned, while Python and runner images may change; privacy scanning remains heuristic.

## Session A re-verification handoff

1. Compare PR #2's head SHA and CI URL with its `headRefOid`, the run's `headSha`, and the job's tested-commit line; require all steps to succeed.
2. Review `Tensor::from_vec`, `collect_data`, and capacity errors against validation order, pre-write capacity checking, pointer adoption, zero-budget behavior, and synthetic 64 MiB boundaries. See [tensor implementation](../../crates/xynoos-tensor/src/tensor.rs) and [typed errors](../../crates/xynoos-tensor/src/error.rs).
3. Match the positive control, six mutations, and seed against debug and release logs. Reproduce with `python3 tools/numerical_oracle.py --check` and `cargo test -p xynoos-tensor --test numerical_oracle --locked --offline -j 1 -- --show-output` if desired. See the [fixture consumer](../../crates/xynoos-tensor/tests/numerical_oracle.rs).
4. Compare the pinned toolchain with CI's version, limits, permissions, and offline Cargo commands. Record independent C-001, T-15, and T-16 verdicts on the reviewed head, using the canonical single-worktree policy. See [toolchain pin](../../rust-toolchain.toml), [workflow](../../.github/workflows/ci.yml), and [workspace policy](../../AGENTS.md).
