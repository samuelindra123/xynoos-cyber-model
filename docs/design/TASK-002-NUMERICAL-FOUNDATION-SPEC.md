# TASK 002 — Numerical Foundation Specification

**State:** Proposed; project-owner approval required before implementation.
**Scope:** Minimal dense CPU tensor foundation only.

## Goals and non-goals

Provide a safe, understandable, independently verifiable tensor representation and the smallest useful set of dense numerical operations needed to establish shape, ownership, and arithmetic conventions. This is a reference baseline, not a performance-optimized engine.

Out of scope: neural-network layers, autodiff, gradients, optimizers, tokenization, training, checkpoint formats, GPU/distributed execution, sparse tensors, mixed precision, quantization, broadcasting, arbitrary tensor views, custom allocators, multithreading, C FFI, and unsafe/SIMD kernels.

## Architecture alternatives comparison

Ratings are qualitative engineering judgment for this milestone, not benchmark results. “High” performance potential describes headroom in principle only; no option has been measured for XYNOOS workloads. Rust and C can both call optimized numerical libraries, and a library may be faster than a first custom kernel. Memory safety refers to ordinary implementation patterns; no language prevents arithmetic or algorithm defects.

| Criterion | A — Rust-first custom | B — C-first custom | C — Rust plus narrow C kernels | D — Existing numerical primitives |
|---|---|---|---|---|
| Memory safety | High in safe subset; unsafe remains possible | Low by default; manual lifetime and bounds discipline | High at API boundary if Rust owns state; C internals remain manual-risk | Depends on library implementation and API; Rust-safe wrappers reduce caller risk |
| Numerical correctness | Must implement and verify kernels | Must implement and verify kernels | Must verify both sides and ABI conversions | Lower custom-kernel burden when operations are validated, but project still verifies semantics/integration |
| CPU performance potential | High, including future optimized Rust kernels; unmeasured | High; unmeasured | High; unmeasured, with FFI cost/complexity | Potentially high through mature backends; workload/target-dependent and unmeasured here |
| Development complexity | Medium | Medium-high due to manual memory/error handling | High due to dual toolchains and FFI | Low-medium for integration, medium for dependency/backend selection |
| Dependency complexity | Low with standard library | Low with standard C library | Medium-high (Rust and C toolchains/build) | Medium-high, varies by crate and optional backend |
| Portability | High for safe standard Rust baseline | High for portable C subset; toolchain details remain | Medium; ABI/toolchain/platform matrix grows | Varies by crate, backend and native-library packaging |
| Maintainability | High for narrow safe API | Medium; requires strong ownership conventions | Lower for a small team until clear bottlenecks exist | High when dependency is active and documented; upgrade/compatibility burden exists |
| Debugging | Standard Rust tooling; shape logic remains project-owned | Familiar native debugging but manual memory defects harder | Cross-language stack and ABI debugging | Dependency internals can complicate diagnosis; integration surface may be smaller |
| Testing | Direct unit, property, oracle and profile tests | Same plus memory instrumentation emphasis | Differential tests plus FFI lifetime/ABI/sanitizer tests | Validate wrapper semantics and compare results; library tests are not a replacement for project tests |
| Future differentiation | Flexible; explicit kernels can later expose primitives | Flexible but higher safety burden | Flexible but gradient boundary must cross FFI carefully | Possible above primitives, depending on library APIs and operation coverage |
| Future Transformer support | Flexible; matmul and elementwise semantics are project-owned | Flexible with same implementation burden | Flexible with extra boundary overhead/complexity | Often good operation coverage, but exact API and backend behavior must be checked |
| Future quantization | Full control, but implementation and validation cost is ours | Full control, with manual safety cost | Flexible but split ownership complicates kernels | Depends strongly on library/backend support; may require custom extension |
| Long-term research value | High for learning and owning a reference implementation | High for low-level systems learning, greater memory-safety risk | Useful if a measured kernel boundary warrants it; otherwise over-complex | High for model research using validated primitives; less custom-kernel research value |

### Option assessment

- **A — Rust-first:** Best fit for an early reference implementation where shape, ownership, and errors are central. Safe ownership is a language property; the quality and speed of a custom kernel remain implementation questions. This is the proposal for TASK 002.
- **B — C-first:** Gives direct control over allocation and data layout and is suitable for systems-kernel research, but every allocation, lifetime, and index contract needs manual enforcement. No present evidence requires that additional audit burden.
- **C — Hybrid:** Can retain a safe Rust public API while isolating a measured kernel in C. It adds FFI ownership, ABI, native build, and cross-language testing. No actual hotspot has been measured, so this is deferred.
- **D — Existing primitives:** Can reduce custom numerical code and may provide optimized matrix operations. `ndarray` demonstrates established Rust array/shape/stride conventions; BLAS defines standard dense GEMM operations. The exact dependency's licensing, maintenance, target support, API, and backend behavior still require review. For a research project whose stated direction includes native infrastructure, using a library for all representations would also leave core learning goals unaddressed. A later hybrid decision may be better than reimplementing mature kernels indefinitely.

### Established facts versus hypotheses

Established from primary documentation: Rust `Vec` stores contiguous initialized elements and offers fallible reservation APIs; Rust `f32` is binary32 with floating-point rounding and NaN caveats; `ndarray` documents row-major default arrays and dynamic dimensions; Netlib BLAS DGEMM specifies dense matrix multiplication. These facts support API and risk analysis. They do **not** establish that custom Rust is faster, that a selected crate is correct for this project, or that f64 accumulation is an optimal training choice. Those remain hypotheses to test against bounded workloads.

## Proposed ADRs (all pending owner approval)

### ADR-002-A — Primary implementation language

- **Context:** The project names Rust and C as primary languages but has no numerical code. The first implementation must make ownership and invalid-input behavior easy to audit.
- **Alternatives:** Rust-first; C-first; hybrid Rust/C; existing numerical crate as the tensor engine.
- **Decision:** Use Rust for the TASK 002 tensor core, in safe Rust. Keep C available for a later, separately reviewed kernel only if measured results justify it.
- **Justification:** Rust's ownership model and standard collection APIs reduce manual lifetime and bounds hazards. A single-language baseline avoids FFI contracts and duplicate build/test surfaces. No claim is made that Rust will outperform C.
- **Trade-offs/risks:** Rust does not prevent arithmetic bugs, shape errors, panics, or resource exhaustion. A custom matrix kernel can be slow and must be checked carefully.
- **Validation:** Review all indexing and allocation paths; `cargo fmt --check`, Clippy with warnings denied, debug/release tests, and independent oracle comparisons.
- **Reconsider if:** Measured workloads show a specific bottleneck and a safe Rust optimization/library is inadequate, or an interoperability requirement appears.

### ADR-002-B — Data type and accumulation

- **Context:** A compact initial type is useful for a CPU-first model path, while simple reductions can accumulate rounding error.
- **Alternatives:** `f32` storage/accumulation; `f64` throughout; multiple dtypes; `f32` storage with wider accumulation.
- **Decision:** Store `f32` only. For dot product, sum, and matrix multiplication, convert each `f32` operand to `f64`, accumulate in a documented increasing-index order, then cast once to `f32`. Elementwise add/subtract/multiply and scalar multiplication use `f32` operations.
- **Justification:** The behavior is explicit, gives one baseline type, and improves accumulation precision without mixed-storage support. It remains a proposal requiring measured evaluation later.
- **Trade-offs/risks:** `f64` accumulation may cost more on some CPUs; it does not guarantee correctly rounded real-number results for all lengths. Casting a finite `f64` result may produce infinity in `f32`.
- **Validation:** Reference cases, cancellation and magnitude tests, tolerance comparisons, and explicit NaN/infinity behavior. Do not require cross-target bitwise equality for all arithmetic.
- **Reconsider if:** Reproducible model workloads or measured performance/accuracy needs support a different accumulation policy.

### ADR-002-C — Shape, rank, and memory layout

- **Context:** Future matrix and tensor operations need unambiguous shape and indexing.
- **Alternatives:** Fixed rank 1/2 only; dynamic rank; dynamic rank with general strides/views.
- **Decision:** Dynamic-rank shape represented by `Vec<usize>`, with row-major contiguous storage. Rank 0 is a scalar with one element. Any zero dimension makes element count zero. No views or user-provided strides are exposed. Canonical strides are derived from the rightmost axis when needed and checked for overflow; implementation may cache them only if it validates the invariant.
- **Index mapping:** For rank `r > 0`, canonical stride `s_i = product(shape[i+1..r])`; flat offset is `sum(index[i] * s_i)`. Rank-0 scalar has no indices and offset 0. Rightmost dimension varies fastest. Shape product and all offset arithmetic are checked.
- **Empty shapes:** `[]` means one scalar. Shapes such as `[0]` or `[2, 0, 3]` have zero elements. Access always fails when element count is zero. If canonical stride derivation overflows for a zero-sized shape, reject the shape with `ShapeOverflow` rather than invent a stride.
- **Justification:** Dynamic rank avoids a representation redesign for ordinary higher-rank dense data while omitting complex view semantics. The operation surface remains small.
- **Trade-offs/risks:** Dynamic rank increases API validation work. Stored strides could become inconsistent; derived strides avoid that state.
- **Validation:** Rank 0, rank 1, rank 2, higher-rank flat ordering, zero dimensions, checked product/stride overflow, and invalid index tests.
- **Reconsider if:** A measured, concrete operation needs non-contiguous views or another layout.

### ADR-002-D — Ownership and memory

- **Context:** Operations need clear aliasing and allocation behavior.
- **Alternatives:** Shared reference-counted buffers; views; owning copy; caller-owned buffers.
- **Decision:** A Tensor owns one `Vec<f32>` and shape metadata. Inputs from slices are copied. Each operation allocates and returns a new owning Tensor; no implicit aliasing, in-place mutation, or view is part of this task. Read-only access may return `&[f32]`.
- **Allocation safeguard:** Use checked arithmetic and `try_reserve_exact` before initializing storage. Expose an explicit `TensorLimits` argument or equivalent documented policy on allocating entry points, with a conservative default cap of 64 MiB of logical f32 payload per tensor. The cap applies to each input construction and each output allocation; arithmetic must also reject when a result would exceed it. This is a bound on requested tensor payload, not an exact process-memory/RSS cap: the allocator may reserve extra capacity and metadata consumes memory. Future model-sized tensors require a separately reviewed policy change. No local test may allocate near the cap.
- **Memory accounting:** `data_bytes = element_count * size_of::<f32>()` checked. Report logical data bytes separately from metadata and allocator overhead. Do not claim exact process RSS or allocator bytes.
- **Justification:** Owning buffers make lifetime and copy behavior reviewable; a default cap discourages accidental unbounded allocation during early research.
- **Trade-offs/risks:** Copies increase memory traffic; the proposed cap is an initial policy limit, not a known model requirement.
- **Validation:** Construction from borrowed input leaves source unchanged; output ownership is independent; limit boundary, overflow, and fallible reserve paths are covered.
- **Reconsider if:** Required workloads cannot fit the cap and bounded, reviewed resource profiles justify a larger configurable limit.

### ADR-002-E — Error behavior

- **Context:** Invalid dimensions, mismatches, indices, and allocation failures must be inspectable and recoverable.
- **Alternatives:** Panics/assertions; `Option`; typed errors; process abort.
- **Decision:** Public fallible APIs return `Result<T, TensorError>`. Variants must distinguish at least shape overflow, allocation limit exceeded, allocation failure, shape mismatch, rank mismatch, out-of-bounds access, and invalid matrix dimensions. Do not panic for user-controlled invalid input or resource failures. Internal `debug_assert!` may express unreachable invariants but must not replace validation.
- **Justification:** Error variants let callers and tests distinguish failure classes.
- **Trade-offs/risks:** Error type is public API surface and may evolve; allocator behavior can still abort in platform/runtime situations outside recoverable collection reservation.
- **Validation:** Every variant has a focused triggering test; panic-catching is not a substitute for `Result`.
- **Reconsider if:** Integration reveals a simpler error taxonomy that preserves diagnosability.

### ADR-002-F — Numerical primitives and dependencies

- **Context:** Custom kernels maximize learning/control but make correctness solely the project’s responsibility. Mature crates reduce surface area but add dependency/version/license/build considerations.
- **Alternatives:** Standard library custom reference kernels; `ndarray` representation/operations; a Rust linear-algebra crate; BLAS backend; C math library.
- **Decision:** Use standard-library-only implementation for the tiny baseline and document its reference status. Do not add a production numeric dependency in TASK 002. Evaluate a library only with an explicit comparison of API fit, maintenance, license, transitive dependencies, portability, and correctness tests.
- **Justification:** Only a handful of operations are required. One dependency can be justified later without making the first API depend on complex backend behavior.
- **Trade-offs/risks:** Custom matmul is not performance tuned. Small tests can miss numerical or edge defects.
- **Validation:** Independent Python oracle, deterministic property tests, debug/release runs, and no benchmark claims absent measurements.
- **Reconsider if:** The baseline makes future differentiation or matrix workloads impractical, supported by evidence.

### ADR-002-G — Unsafe Rust and C integration

- **Context:** Low-level code may eventually improve performance but increases audit burden and memory-safety risks.
- **Alternatives:** Forbid unsafe; allow localized unsafe with justification; C/FFI kernels.
- **Decision:** No unsafe Rust and no C/FFI in TASK 002. `#![forbid(unsafe_code)]` is recommended for the crate. Do not introduce SIMD intrinsics.
- **Justification:** A portable safe reference implementation is the correctness baseline and the current workload has no measured bottleneck.
- **Trade-offs/risks:** This may constrain low-level optimization; it does not prevent safe code from being computationally inefficient.
- **Validation:** CI rejects unsafe code; review build scripts and dependencies for native code.
- **Reconsider if:** A reproducible benchmark establishes a material bottleneck, a safe approach is insufficient, and a separate design review specifies invariants, fallback, sanitizer coverage, and differential tests.

### ADR-002-H — Independent verification and compute tiers

- **Context:** Production arithmetic must not certify itself using only duplicated logic.
- **Alternatives:** Rust-only examples; third-party numerical framework; independent standard-library oracle.
- **Decision:** Use hand-calculated tests plus a deterministic Python standard-library oracle (`math.fsum`, `struct` binary32 conversion) run in bounded CI. Local smoke is optional and small; full normal checks run in GitHub Actions; stress/fuzz/sanitizer/benchmark work stays manual/bounded.
- **Justification:** Python is a distinct implementation language and `math.fsum` supplies an independent, higher-accuracy summation approach. No heavy framework installation is required.
- **Trade-offs/risks:** The oracle itself can contain bugs and binary conversion assumptions; cross-process test protocol adds maintenance.
- **Validation:** Review oracle independently; hand-check known values; test oracle on known binary32 cases; retain a small runtime and deterministic seed.
- **Reconsider if:** The test harness becomes flaky or a clearer independent oracle is available.

## Tensor representation contract

The proposed crate exposes an owning dense `Tensor` with:

- `shape: Vec<usize>`; rank equals `shape.len()`.
- contiguous `data: Vec<f32>` in row-major order.
- `numel` equal to checked product of dimensions, with rank-0 scalar product defined as 1 and any zero dimension giving 0. Validate zero dimensions before multiplying shape extents so a zero-sized shape does not spuriously overflow its element count.
- no externally mutable shape/data pair; constructors are the only way to create a Tensor.
- invariant `data.len() == numel`, `data.capacity()` managed by Rust, and `numel * 4 <= max_tensor_bytes`.
- checked shape product and checked byte calculation before allocation or copy.
- bounds-checked logical indexing. Flat access accepts `i < numel`. Multidimensional access requires index rank equal to tensor rank and every index less than its corresponding dimension.
- no implicit panic-based indexing in the public API. If `Index` traits are implemented, document their panicking semantics and keep the checked getter as the required safe API; preferably omit them in TASK 002.
- empty tensors are valid values and own zero elements. Scalar shape `[]` is distinct from empty shape `[0]`.
- constructors from existing data require exact element count; no truncation/padding.
- `TensorLimits` should make byte cap explicit and validate before reserving. No resize/growth interface is needed.

## Mandatory numerical API

All operations require exact input shapes as stated; no broadcasting. Unless an operation returns an error, output is newly allocated and owns its storage. `f32` NaN and infinities are accepted and flow according to primitive IEEE-style arithmetic; no promise is made about NaN payload preservation. Arithmetic output is not automatically rejected for non-finite values. Tests explicitly verify classification/propagation, not NaN bit patterns.

| Operation | Definition and contract | Complexity / notable edges |
|---|---|---|
| `from_data(shape, &[f32], limits)` | Copy input into owning tensor; input length must equal checked shape element count. | O(n), allocation; zero-length input valid for an empty shape. |
| `zeros(shape, limits)` | Allocate and initialize each element to positive `0.0`. | O(n), not uninitialized memory. |
| `filled(shape, value, limits)` | Allocate with every element equal to value, including NaN/infinity if supplied. | O(n); zero-sized output does no element writes. |
| `shape`, `rank`, `numel`, `data`, `get`, `get_mut` | Shape/data inspection; checked flat and multidimensional element access. Mutable access changes values only, never metadata. | O(rank) for multidimensional offset; all bounds checked. |
| elementwise add/subtract/multiply | Corresponding pairwise `f32` operation; input shapes exactly equal; output shape same. | O(n); empty + empty returns empty; mismatch is an error. |
| scalar multiply | Each element times one `f32` scalar. | O(n); follows float special-value behavior. |
| dot | For rank-1 inputs with equal length n, `sum_i f64(a_i)*f64(b_i)`, cast once to `f32`; returns scalar value. | O(n); empty vectors return `+0.0`; mismatched lengths/ranks error. |
| matrix multiply | Rank-2 A `[m,k]`, B `[k,n]`; `C[i,j] = f32(sum_{p=0}^{k-1} f64(A[i,p])*f64(B[p,j]))`; output `[m,n]`. No transposition flags. | O(mkn); check `k` equality and output allocation. Zero `m`/`n` yields empty output; `k=0` yields zero-filled `[m,n]`. |
| transpose | Rank-2 `[m,n]` becomes newly allocated `[n,m]`, same mathematical values, row-major storage. | O(mn), allocates/copies; square and non-square cases required. |
| sum | Reduce all flat elements in increasing storage order via f64 accumulation; cast once to f32 scalar. | O(n); empty tensor returns `+0.0`; NaN propagates. |

Dot and matrix product define products in f64 from exactly represented `f32` inputs. The reduction order is serial and specified. The output cast may overflow to infinity; this is a numerical result rather than an allocation/shape error. A later optimization changing reduction order must retain tolerance-based differential tests and document reproducibility effects.

## Floating-point comparison policy

For finite expected result `e` and actual result `a`, accept if `abs(a-e) <= atol + rtol*abs(e)`. Proposed initial oracle tolerance: `atol=1e-6`, `rtol=1e-5` for f32 outputs; calibrate with the implementation and the documented operation order, then fix the chosen tolerances in the test code with rationale. Tolerances are not adjustable per failing case. Use exact equality only for constructors, shape/index metadata, and operations with exactly representable hand examples where exactness is a documented property.

- If expected is NaN, require actual `is_nan()`; do not compare payloads.
- If expected is positive/negative infinity, require same sign infinity.
- For signed zero, use numeric equality unless the specific operation explicitly promises a sign; constructors promise positive zero only for `zeros` and empty reductions.
- Reject unexpected NaN/infinity for finite oracle values unless the reference result itself rounds to infinity.
- Avoid asserting bitwise identical arithmetic across compiler targets or debug/release configurations.

## References

- Rust standard library `Vec` allocation and contiguous storage: https://doc.rust-lang.org/std/vec/struct.Vec.html
- Rust `try_reserve`: https://doc.rust-lang.org/std/vec/struct.Vec.html#method.try_reserve
- Rust `f32` behavior: https://doc.rust-lang.org/std/primitive.f32.html
- `ndarray` row-major and dynamic dimension documentation (consideration only): https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html
- Netlib BLAS DGEMM definition (later comparison only): https://www.netlib.org/lapack/explore-html/d7/d2b/dgemm_8f_source.html
- Python `math.fsum`: https://docs.python.org/3/library/math.html#math.fsum

