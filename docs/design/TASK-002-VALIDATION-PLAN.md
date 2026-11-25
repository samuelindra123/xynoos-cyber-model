# TASK 002 — Validation Plan

**State:** Proposed plan for Session B; thresholds and CI changes require implementation review.
**Purpose:** Establish evidence for shape, ownership, arithmetic, and resource contracts without relying on CI status alone.

## Independent numerical oracle

Add a small, checked-in Python 3 script under `tools/` or `tests/oracle/` using only the Python standard library (`struct`, `math`, `json`/`subprocess` as needed). It must not call production Rust kernels or install NumPy/PyTorch.

1. Represent test inputs as exact binary32 values by round-tripping through `struct.pack/unpack('!f', x)`.
2. For elementwise operations, compute using Python float operands corresponding to binary32 inputs, then round output through binary32 conversion.
3. For dot, sum, and matmul, convert binary32 inputs to Python double, multiply in double, sum terms with `math.fsum`, then round the result to binary32. Document that this oracle is independent in language and reduction implementation, while sharing the mathematical specification.
4. Produce deterministic JSON fixtures or invoke a small Rust test helper through a documented line protocol. Keep fixture sizes bounded (default total <= 10,000 scalar outputs); do not make CI install numeric frameworks.
5. Check the oracle itself against hand-calculated values, including binary32 rounding and signed infinities. Python float-to-binary32 overflow should be handled explicitly as signed infinity where the mathematical reference overflows binary32.
6. Record oracle version/runtime and seed in test output. Do not treat the Python script as the sole proof: retain hand-computed Rust tests and property tests.

The Rust and Python paths must not both call the same production helper. An implementation that copies the Rust loop verbatim into Python is not sufficiently independent.

## Floating-point acceptance

Use the mixed tolerance `|actual - expected| <= atol + rtol*|expected|` with proposed starting values `atol=1e-6`, `rtol=1e-5` for finite f32 outputs. Session B must justify and freeze final constants using representative edge fixtures before declaring the gate complete. No per-test tolerance inflation to silence failures. Compare NaN by classification, infinity by sign, and do not compare NaN payload bits. Unexpected non-finite output for finite oracle output fails. General arithmetic must not use exact equality.

Exact assertions are appropriate for shape, errors, indexing, construction preserving exact f32 input bits, zero initialization, and simple binary-exact arithmetic cases. The same algorithm cannot serve as both implementation and oracle without explicitly recording the verification limitation.

## Test matrix

| Area | Required cases |
|---|---|
| Hand reference | Small signed examples for each op; manually calculate dot and matmul; include `[2,3] x [3,2]` non-square matrices. |
| Construction | scalar `[]`, vector `[n]`, matrix, rank >2, zero-filled, constant-filled, copied data, mismatched element count. |
| Layout and indexing | Verify row-major flat order; valid first/last index; wrong index rank; each axis out of bounds; flat index `numel`; scalar access rules. |
| Shape arithmetic | Product exactly at accepted limit; product overflow; byte multiplication overflow; canonical stride overflow; allocation limit exceeded before reserve. |
| Empty behavior | `[]` scalar vs `[0]` empty; `[2,0,3]`; elementwise empty; empty dot and sum; zero-sized output matmul; `k=0` matrix multiply. |
| Elementwise ops | Positive/negative/zero values, matching shapes, all mismatch cases, empty inputs, NaN/infinity classification. |
| Dot and sum | Length 0/1/multiple, cancellation, alternating magnitude, negative values, subnormal-ish and large finite values, NaN and infinities. |
| Matmul | Square, rectangular, identity, zero matrices, negative values, `k=0`, incompatible inner dimension, dimension/byte overflow, non-finite operands. |
| Transpose | 1xN, Nx1, non-square, square, empty dimensions; double transpose equals original values/shape. |
| Deterministic randomized | Fixed published seeds; small dimensions (0..8); bounded magnitudes initially (e.g. `[-10,10]`) plus separate exponent-stratified cases. Compare all defined operations to independent oracle. |
| Properties | Add/subtract shape preservation; `A+0=A`; `A*1=A`; transpose twice; sum of concatenated small vectors where finite; matmul by identity; dot commutativity within tolerance. State floating-point limitations for each. |
| Resource and errors | Tensor limit, output limit, fallible reservation error mapping where reliably inducible without exhausting process memory; assert failure occurs before attempted large allocation using a tiny configured test limit. |
| Build modes | Debug and release tests produce oracle-equivalent answers; no general promise of bitwise equality. |

Do not attempt to trigger real system-wide OOM in tests. Allocation failure injection is optional if a clean test seam exists; otherwise verify checked rejection with tiny configured limits and unit-test mapping of `TryReserveError` without exhausting memory.

## Property-based and randomized testing

Property tests may use deterministic generated loops in the first milestone; adding a property-testing crate is optional and requires dependency review. If a crate is selected, pin through `Cargo.lock`, document license and transitive dependencies, constrain case count and input dimensions, and seed deterministically in CI. Generate valid shape/value pairs separately from invalid input. Keep every CI case bounded; no unseeded flaky arithmetic property is acceptable.

## Validation tiers

### Tier 1 — Local smoke verification

- **Scope:** formatting, compile check, focused unit tests, a handful of oracle fixtures, docs hygiene.
- **Command pattern:** `cargo fmt --all -- --check`; `cargo test -p xynoos-tensor` (or actual crate name); `python3 <oracle-script> --smoke`.
- **Bounds:** no more than 10,000 scalar oracle results; individual tensor test limit <= 1 MiB; expected under 2 minutes on a resource-constrained CPU.
- **Required result:** all selected commands exit 0; report exact commands/results. Do not run long benchmarks or compile unrelated large dependencies locally.
- **Failure:** stop on failure, preserve output, fix or report; do not increase limits or disable tests to make the run green.

### Tier 2 — Standard GitHub Actions CI

- **Scope:** formatting, Clippy, debug/release tests, Rust docs, independent oracle, deterministic property cases, dependency/license policy checks where applicable.
- **Command pattern:** `cargo fmt --all -- --check`; `cargo clippy --workspace --all-targets -- -D warnings`; `cargo test --workspace`; `cargo test --workspace --release`; `cargo doc --workspace --no-deps`; `python3 <oracle-script> --ci`.
- **Bounds:** single Ubuntu job initially, `timeout-minutes: 15`; concurrency cancellation; max oracle outputs 10,000; maximum per-test tensor <= 1 MiB; no downloaded datasets or generated model artifacts. Use only stable Rust unless an explicit tool requires otherwise.
- **Dependencies:** `cargo test --locked` after `Cargo.lock` exists. Run an advisory and license check when third-party dependencies exist; pin checker version and document policy. Until then, do not add a network dependency scanner that has nothing to inspect. Repository token permission remains read-only.
- **Required results:** every required job passes on the PR head SHA; pending/skipped/cancelled/failed is not success. Attach logs or links in the PR. CI success proves only declared checks ran.
- **Failure:** block review readiness; inspect exact failing step, fix the cause, and rerun. Do not relabel failed jobs optional without owner-reviewed rationale.

### Tier 3 — Manual research validation

- **Scope:** larger deterministic property sweeps, bounded fuzzing, sanitizer checks, memory-boundary tests, performance measurement, and cross-target/reproducibility exploration.
- **Command pattern:** dedicated manual `workflow_dispatch` or documented CI job; `cargo test` with a fixed seed and case budget; `cargo-fuzz` with a fixed short duration/corpus cap if adopted; sanitizer/nightly job only when relevant and supported.
- **Bounds:** explicit input/seed, timeout <= 30 minutes per job, one job at a time, test tensors <= 64 MiB each, no training/data download, benchmark workload capped by iteration/time count. Use remote runners; do not mirror stress workloads on personal devices.
- **Required results:** preserve source SHA, exact command, seed, toolchain, limits, result, and limitations. Performance claims include a baseline and variance; sanitizer claims identify tool and coverage.
- **Failure:** record the finding and regression test; do not discard failing seeds. A successful fuzz duration is bounded evidence, not proof of absence of defects.

## CI architecture

Start with one required validation job to avoid redundant runner cost. Split out a job only when it has a distinct toolchain or materially different bounded workload. Pin third-party Actions to full commit SHAs. Use minimal `contents: read`, job-level timeout, concurrency cancellation, locked dependency resolution, and no secrets. Do not add training or benchmark jobs to pull-request CI. Manual extended workflows must be separately named and clearly excluded from routine acceptance.

