# TASK 002 — Session B Implementation Handoff

**Recipient:** Session B implementation agent
**State:** Planning only; all architecture choices below remain proposals until the project owner approves them.
**Baseline:** TASK 002-A planning branch; do not infer production work has been approved by the existence of this document.

## 1. Exact implementation scope

After project-owner authorization, implement one small Rust workspace crate for dense CPU tensor values. Implement the owning row-major tensor representation and the mandatory API below, typed errors, bounded allocation, documentation, and meaningful tests. No production C or model code is part of this task.

## 2. Explicit non-goals

No gradients/autodiff, layers, tokenizer, optimizer, training, inference runtime, checkpointing, datasets, GPU, distributed execution, sparse tensors, mixed precision, quantization, broadcasting, tensor views, custom allocator, multithreading, C FFI, unsafe Rust, SIMD intrinsics, benchmark claims, or large workload.

## 3. Proposed repository and crate structure

```text
Cargo.toml                 # workspace
Cargo.lock                 # committed for reproducibility
crates/
  xynoos-tensor/
    Cargo.toml
    src/lib.rs
    src/error.rs
    src/shape.rs
    src/tensor.rs
    src/ops.rs
    tests/
      tensor_contract.rs
      numerical_oracle.rs
      fixtures/
    README.md
 tools/
   numerical_oracle.py     # standard-library only, if process fixture protocol selected
```

Keep crate boundaries minimal: one `xynoos-tensor` crate; do not create a generic backend abstraction or separate kernel crates before a second real backend exists. Oracle tooling may instead live under the crate tests if that is simpler.

## 4. Tensor invariants

- Data type `f32`; each tensor exclusively owns a `Vec<f32>`.
- Shape is dynamic `Vec<usize>`; rank equals shape length.
- Contiguous row-major data; rightmost axis varies fastest.
- Scalar is shape `[]` with one element. Any zero dimension yields zero elements.
- Every shape product, byte count, stride/offset calculation is checked.
- `data.len() == numel`; shape/data fields are private and changed only by checked constructors.
- Default per-tensor logical payload limit is 64 MiB; check inputs and each output before allocation. This is not an exact RSS cap because allocator capacity and metadata add overhead. Limit must be explicit/configurable in API without silently permitting unbounded allocation.
- Allocate using fallible reservation then initialize; map capacity/allocator errors to `TensorError`. Never test by actually exhausting memory.
- Constructors from slices copy exactly; no views or implicit aliasing. Operations allocate independent outputs. Mutable access cannot change shape.
- No public operation panics on invalid input. No `unsafe`; preferably crate-level `#![forbid(unsafe_code)]`.

## 5. Public API expectations

Use documented `Result<_, TensorError>` APIs. Names may follow Rust style, but provide equivalents of:

- `Tensor::from_data(shape, data, limits)`
- `Tensor::zeros(shape, limits)` and `Tensor::filled(shape, value, limits)`
- `shape()`, `rank()`, `numel()`, `data()`
- checked `get_flat`/`get_flat_mut` and rank-checked `get`/`get_mut`
- `add`, `subtract`, `multiply`, and `scale`
- `dot` on equal-length rank-1 tensors
- `matmul` on rank-2 tensors
- `transpose` on rank-2 tensors
- `sum` over flat storage

Do not implement `Index` if its panic behavior would undermine the checked-access contract. If included, document it clearly and retain checked APIs.

## 6. Operation definitions

Follow the complete formulas, output shapes, empty behavior, allocation contracts, and edge cases in [the numerical foundation specification](TASK-002-NUMERICAL-FOUNDATION-SPEC.md). Key rules:

- Elementwise operations require identical shapes; no broadcasting.
- Dot requires equal-length rank-1 inputs and returns scalar `f32`.
- Matmul accepts `[m,k]` and `[k,n]`, produces `[m,n]`, and returns zero-filled results for `k=0`.
- Transpose is rank-2 and returns a newly allocated contiguous copy with shape `[n,m]`.
- Sum of any empty tensor and dot of empty vectors return positive zero.
- Dot, sum, and matmul accumulate in sequential order using f64 and cast once to f32; document that this is a proposed baseline behavior.
- Inputs may contain NaN/infinities. Arithmetic follows primitive floating-point propagation; no NaN payload guarantee. Non-finite output is not itself an API error.

## 7. Error and ownership contracts

Define errors for shape product/stride overflow, allocation cap exceeded, allocation failure, shape mismatch, rank mismatch, index out of bounds, and invalid matrix dimensions. Error values should include relevant expected/actual shape or index information without machine-specific data. Document ownership and allocation behavior on each method. No implicit clone inside operation loops; each operation returns one output allocation, except scalar outputs.

## 8. Dependencies and security

Start with standard library only. Commit `Cargo.lock`. If a dependency becomes necessary, provide name, exact version, purpose, license, transitive/native build effects, maintenance status, portability impact, and alternatives in the PR before adding it. Do not add C build scripts, network downloads during tests, datasets, binaries, secrets, machine-specific reports, or unsafe code.

## 9. Testing matrix

Implement the cases in [the validation plan](TASK-002-VALIDATION-PLAN.md): hand-calculated positive/negative/zero examples; scalar/vector/matrix/higher-rank/empty shape cases; checked overflow and all shape/index mismatch errors; non-square matrices; `k=0`; transpose; NaN/infinity classification; large and small magnitudes; cancellation; deterministic random cases; algebraic properties within tolerance; resource cap rejection; debug and release behavior.

Use an independent standard-library Python oracle with fixed seed and bounded fixtures, plus manual reference tests. Proposed initial tolerance for finite f32 results is `atol=1e-6`, `rtol=1e-5`; calibrate once and explain any chosen final values. Never use exact equality for arbitrary floating-point results.

## 10. CI and documentation requirements

One bounded PR job is initially sufficient. Run formatting, Clippy with warnings denied, debug and release tests, Rust docs, oracle tests, deterministic bounded properties, and dependency/license/advisory checks once dependencies exist. Pin Actions by full commit SHA, use read-only token permissions, job timeout <= 15 minutes, concurrency cancellation, `--locked`, and no training/benchmark/download jobs. See validation tiers for exact bounds.

Document crate purpose, tensor layout and scalar/empty semantics, precision/accumulation, API errors, allocation limits, local smoke command, CI checks, and known limitations. State explicitly that this is not yet an ML engine and makes no performance claim.

## 11. Measurable acceptance criteria

TASK 002 implementation is eligible for independent review only when all items below are evidenced on the PR head SHA:

1. The crate builds in debug and release on the declared stable toolchain with locked dependency resolution.
2. Formatting and Clippy pass with no warnings under the specified commands.
3. Public constructors preserve all tensor invariants; scalar and empty tensors follow the specified distinct semantics.
4. Every mandatory operation matches hand-calculated fixtures and the independent oracle under the fixed tolerance/special-value policy.
5. Non-square matmul, dimension mismatch, `k=0`, transpose, shape overflow, byte overflow, invalid indices, and allocation-cap rejection have regression coverage.
6. Invalid user inputs and configured resource-limit failures return typed errors; no expected invalid-input path panics.
7. No unsafe code, C source, FFI, unreviewed dependency, data/model artifact, or unrelated milestone work is present.
8. Debug and release outputs both satisfy oracle comparisons; no claim of bitwise cross-profile equality is made.
9. PR CI is fully completed and passing; pending, skipped, cancelled, or failed runs do not satisfy this condition.
10. Documentation explains operation semantics, numerical policy, allocation limits, commands actually run, limitations, and evidence SHA.
11. Changes are isolated on a feature branch; the PR targets `main`; no merge occurs without explicit owner authorization.

These are review gates, not an automatic project-owner approval.

## 12. Git and PR requirements

- Begin from current `main`; create a focused branch such as `feat/task-002-tensor-foundation` after authorization.
- Do not commit to `main`, force-push, or merge.
- Keep TASK 002 implementation changes separate from TASK 002-A planning docs.
- PR title should identify TASK 002 numerical foundation; description includes API, design deviations, errors/limits, dependencies, validation commands/results, CI links, resource impact, and open risks.
- Do not modify unrelated files. Obtain project-owner review before merge.

## 13. Resource boundaries

Local: formatting, static checks, crate build, bounded smoke tests, and small oracle fixtures only. No large dependency installation, stress test, large artifact, benchmark run, or training. GitHub Actions: normal bounded correctness suite. Extended stress/property/fuzz/sanitizer/performance experiments: manually dispatched with explicit seed, timeout, and allocation cap; no dataset download or paid infrastructure.

## 14. Unresolved owner decisions

1. Approve or revise Rust-first safe baseline.
2. Approve `f32` storage with `f64` accumulation.
3. Approve dynamic rank and rank-0 scalar semantics.
4. Approve 64 MiB default per-tensor allocation cap and configuration shape.
5. Approve the exact mandatory operation set and standard-library-only initial dependency policy.
6. Approve Python standard-library oracle in CI and the final fixed tolerance values after calibration.
7. Confirm whether private GitHub vulnerability reporting is enabled (TASK 001 audit follow-up).

Session B must not treat this handoff as authorization to start until the project owner approves the architecture and milestone.

