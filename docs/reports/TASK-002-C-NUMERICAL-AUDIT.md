# TASK 002-C Numerical Audit

## Method and independent checks

Source reviewed: `shape.rs`, `tensor.rs`, `ops.rs`, `error.rs`, the contract tests, Python oracle and TSV consumer. Bounded command checks passed: `cargo test --workspace --locked --offline -j 1`; `cargo clippy --workspace --all-targets --locked --offline -j 1 -- -D warnings`; `python3 tools/numerical_oracle.py --check`. The deterministic generator reported seed `0x58594e4f`, 295 cases, 2,670 outputs. Hand-checked independent cases included `[2,3] × [3,2]` with output `[58,64,83,90]`, transpose of a non-square matrix, zero inner dimension, signed zero, cancellation, and scalar versus empty shapes. Existing tests include additional property and special-value cases.

## Shape and allocation

Shape products are accumulated from the right. This both checks suffix products and gives the intended canonical-stride rejection for e.g. a zero dimension left of an overflowing suffix. Rank zero yields one element; any representable zero product yields none. In-bounds checks precede offset arithmetic, and checked offset arithmetic returns typed errors. Tensor fields are private, shape and data are owned, and `data.len()` is the element count. Allocation paths use fallible reserve for project-created buffers and validate logical output shape before filling.

`from_vec` is the exception to memory-limit interpretation: it adopts a caller-owned allocation, and its Vec capacity is unconstrained by logical byte validation. This can retain large spare capacity after accepting a short logical payload. `TensorLimits` is therefore a logical payload cap, not a bound on actual retained allocation. Per-tensor limits also do not constrain total live memory, metadata, allocator overhead, OS overcommit, or aggregate process usage.

## Arithmetic and floating point

Elementwise operations use f32. Dot and matmul convert operands to f64 before multiplication, add sequentially in increasing index order, then cast once to f32. Sum also sequentially accumulates f64 and casts once. Empty reductions initialize to positive zero. The documented `[1e30, 1, -1e30]` yields zero under the specified sequential f64 operation: at that magnitude, adding one rounds away; `math.fsum` can recover one by a different, higher-accuracy accumulation strategy. This is policy difference, not evidence of a Rust bug. Finite comparison tolerance is `1e-6 + 1e-5 * abs(expected)`; classification checks are appropriate for NaN/infinity, while bit checks are used for signed zero/subnormal contract cases. No NaN payload or cross-target universal bitwise promise is made.

Potential later ML concerns: f32 storage loses precision; sequential reductions are order-dependent; f64 reduction support/performance varies by CPU; gradient accumulation may amplify rounding; f32 conversion can overflow to infinity. These are documented design tradeoffs rather than defects against the approved contract.

## Matrix algorithms

Matmul checks rank and inner dimensions, allocates `[rows, columns]` through checked limits, returns early for empty output or zero inner dimension, iterates left rows and right columns, and computes each cell in increasing inner index. The row and column iterators use `chunks_exact`, `step_by(columns)`, and `zip`; validated tensor lengths and early returns make lengths agree, and zero divisors are avoided. Transpose similarly returns for empty output before `step_by(columns)`, then copies columns into output rows. Complexity is O(mnk), not bounded by a payload cap. The API leaves workload limits to the caller; compute quotas may be appropriate before untrusted/model-scale use.

## Oracle integrity

Python oracle is separate from Rust kernels, uses a fixed seed, `struct` binary32 conversions, `math.fsum` finite reductions, explicit special-value rules, and writes deterministic fixture bytes. Rust consumer rejects unknown ops, parses fields, dispatches all rows, compares output shape/count and checks aggregate cases/output bounds. Existing parser uses assertions/panics for malformed committed fixtures, which is suitable for failing CI. `python3 tools/numerical_oracle.py --check` passed. I did not run copied-fixture or mutation negative controls; therefore mutation sensitivity is not independently established. The Python oracle's accurate summation differs intentionally from Rust sequential contract, with stated tolerance and dedicated exact cases.

## Test / CI conclusions

The reported 15 tests per profile correspond to 3 unit, 10 integration contract, 1 oracle integration, and 1 doctest. The suite has meaningful coverage for indexing, shape overflow, allocation rejection, scalar/empty distinctions, arithmetic, reduction identity/order, matrix layout, transpose, and special values. It is not exhaustive. GitHub CI run 45527030055 succeeded on the reviewed SHA; no stale-head discrepancy found. The workflow uses moving `stable`, so toolchain-level reproducibility is incomplete even though dependencies are locked and offline-resolved.
