# TASK 003-C — Independent Verification Report

## Executive summary

The implementation is structurally sound in the inspected ownership, reverse-mode, and expected-error transaction paths. The exact implementation head has successful Foundation CI. Source review and its tests support the core derivative formulas, matrix offsets, single-use gradient ownership, and bounded synthetic regression result. This audit does not establish process RSS limits or general neural-network training correctness.

**Verdict: PASS — READY FOR OWNER ACCEPTANCE.** This is a bounded first-order CPU autograd implementation, with documented exclusions. No confirmed acceptance-blocking defect was found in this audit.

## Repository and CI evidence

The audit branch is based on implementation SHA `9f36a29cd7df7b07298d564771a8b823113e111c`, matching open PR #5 head. PR #4 is open at planning SHA `8bc693ce26f0bbb5f8c647162cd5afe39880843a`. The accepted TASK 002 baseline is `c5250543e5a39b087b7daab3d86d69a2af65a703`. Foundation CI run `45551441899`, job `138993302987`, completed successfully on the exact implementation SHA. Its steps include revision verification, repository/privacy checks, Rust toolchain installation, locked offline metadata, formatting, Clippy with denied warnings, both oracle fixture checks, debug and release tests/doctests, trajectory comparison, and documentation.

Workflow configuration pins Rust through `rust-toolchain.toml`, checks out the PR head SHA, uses `--locked --offline` after toolchain installation, sets a 15-minute timeout and two build jobs, and grants only `contents: read`. CI reports no skipped steps. Per the implementation report, each profile ran 69 Rust tests/checks; the run logs report both debug/release suites successful. The requested actual run and job are linked from PR #5 and the implementation report.

## Lifecycle, identity, and math

`Tape<'p>` holds an immutable borrow of `ParameterSet`, copies input/parameter leaves into budget-owned tensors, and records owned outputs. The borrow ends when the tape is consumed/dropped; parameter gradients move into an owned batch. Private owner/index fields plus checked owner and bounds lookups reject cross-tape and cross-set handles. A failed first backward seals the tape (`Failed` is set before validation/preflight); later attempts reject. Forward calls require `Recording`. Failed backward temporaries drop with the local result. No `unsafe`, reference cycles, or value/gradient interior mutability were found.

Reverse traversal is descending node order and edge order is explicit. Repeated operands contribute once per edge, accumulation is binary64 then binary32, and disconnected/non-differentiable leaves have no adjoint. MSE deliberately requires an `Input` target per the accepted spec; its prediction derivative includes the upstream seed. This rules out silently losing a trainable target contribution by rejecting such targets.

Hand-derived formulas agree with kernels: add `(g,g)`, subtract `(g,-g)`, multiply `(g*b,g*a)`, scale `c*g`, sum broadcast of scalar seed, mean `g/n`, dot `(g*b,g*a)`, transpose inverse index mapping, and MSE `g*2*(p-t)/n`. Matmul uses the direct row-major formulas `dA[i,q]=Σ_j G[i,j]B[q,j]` and `dB[q,j]=Σ_i A[i,q]G[i,j]`, increasing reduction index and one final f32 conversion. Its `i/k, i%k` and `i/n, i%n` indexing occurs only when corresponding output derivative is nonempty, so zero divisors are avoided. Layout checks rank and inner dimension before execution; work estimates charge output cells and multiply-add work, including nonempty zero-inner outputs.

The implementation report's independent oracle covers 25 analytical/finite-difference coordinates and three negative controls. Its tolerance equations match the prescribed fixed tolerances, and perturbations use actual representable f32 distances. Reported controls detect sign-flip, zero-coordinate, and rectangular permutation errors. The independent Python reference does not use production backward calculations.

## Transactions and resource accounting

Gradient accumulation preflights inventory, shapes, work and values; stages replacement sums for existing gradients; and commits by moving owned buffers only after candidate generation succeeds. Missing gradients remain `None`; `zero_grad` stages explicit zero buffers and swaps only after all succeed. SGD validates inventory/shapes/finiteness, checks revision/work, stages every candidate using f64 product/subtraction and a single f32 cast, then commits values, clears gradients and advances revision. Targeted tests explicitly fail on later zero-grad and SGD candidates and assert value, gradient, revision, and ledger preservation. Revision advancement is checked; empty SGD is a no-op.

The shared per-set `Rc<Cell<usize>>` ledger follows RAII charges attached to tensors and metadata tables. Detached vector allocations are charged before reserve and reconciled to actual retained capacity before publication; charge drops release bytes. Overlapping tapes/batches share the set ledger, and moves preserve charges. The fixed-size ledger itself, external caller allocations, allocator bookkeeping, stack, and process RSS are excluded as documented. Allocation aborts are outside transactional guarantees. Downward limits and the 128 MiB cap are source-defined; no 128 MiB allocation was needed. Work budgets use checked arithmetic and per-phase checks; they are application-level estimates, not CPU instruction or time bounds.

## Learning and TASK 002 regression

The native test follows the specified five-sample dataset, zero initialization, tape-recorded scalar predictions and MSE, backward, owned transfer, SGD at 0.05, exactly 100 updates and fresh tape per update, plus initial observation. The Python reference alone uses the closed-form recurrences `w-2 -> 0.8(w-2)` and `b-3 -> 0.9(b-3)`. Reported terminal values are `w=1.999999762`, `b=2.999920368`, loss `6.348966686e-9`, with 101 states and bitwise debug/release agreement at checksum `e50d201a133838fe64158e0dcf94df894efcf17241f1f16e52232a4e75d41c6a`. The reported thresholds and strict monotonicity passed. This demonstrates only this synthetic linear regression case.

TASK 002 fixtures remain unchanged: 295 cases, 2,670 outputs, SHA-256 `cdfde7ba9f44f46ba7e757157019730e1124bcd966e796b73286bddf3894fa0f`, and six negative controls with the accepted tolerances. CI reruns both profiles and oracle checks. Tensor snapshot/capacity additions are additive and no arithmetic source changes are indicated by the implementation diff/report.

## Findings, limitations, readiness

No confirmed CRITICAL, HIGH, MEDIUM, or LOW findings. Informational limits are the documented first-order dense CPU op set, single-threaded per-set accounting, allocator-abort exclusion, and lack of RSS guarantees. This audit did not independently retrieve archived hosted log text beyond GitHub run/job metadata; detailed test counts, fixture metrics and trajectory checksum are reconciled against the checked-in implementation report and successful exact-head job. A bounded local package test was started on this audit branch; its final result is recorded in the audit PR checks.

TASK 004 should wait for owner acceptance and a deliberate scope/specification; this verdict does not authorize starting it. Integration should be conflict-aware: land/accept planning PR #4 as owner intends, then implementation PR #5 against its base, then retain this audit PR targeting the implementation branch for its evidence. Do not merge these automatically or alter accepted history. No remediation handoff is required.
