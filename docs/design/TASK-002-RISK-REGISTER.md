# TASK 002 — Risk Register

**State:** Proposed controls for implementation review. Severity reflects impact if realized; likelihood is a qualitative estimate before implementation.

| Risk | Severity / likelihood | Impact | Prevention | Detection / required regression |
|---|---|---|---|---|
| Shape element-count integer overflow | HIGH / medium | Undersized allocation followed by invalid indexing or incorrect shape | `checked_mul` for every dimension and byte count; no unchecked shape arithmetic | Tests for overflowing dimension products and exact boundary; error before allocation |
| Offset or canonical-stride overflow | HIGH / low-medium | Incorrect element addressed or panic | Checked stride and offset arithmetic; one canonical row-major formula | Rank mismatch, huge dimensions, offset boundary tests; reject overflow |
| Out-of-bounds access | HIGH / low with safe Rust | Panic or wrong result; unsafe future code could cause memory corruption | Checked getter; keep safe Rust; do not expose unchecked indexing | First/last/past-end and every-axis bounds tests; `#![forbid(unsafe_code)]` |
| Shape mismatch | MEDIUM / medium | Silent element pairing/matmul corruption | Exact-shape validation for elementwise ops; explicit rank and inner-dimension checks | Each operation's mismatch cases must return the documented error |
| Allocation overflow or allocator failure | HIGH / low-medium | Abort, panic, or process instability | Checked `numel * 4`, per-tensor byte limit, `try_reserve_exact` before initialization, convert reserve error | Tiny configured-limit rejection tests; mapping tests; never force system OOM |
| Unbounded allocations/resource exhaustion | HIGH / medium | Excessive memory/time and local/CI disruption | Default 64 MiB per-tensor limit; validate outputs before allocate; test limits small; no implicit resize | Boundary tests; CI test-size cap; review every allocating public API |
| Hidden memory duplication/copies | MEDIUM / medium | Peak memory exceeds apparent logical tensor size | Document copy semantics; allocating operations return one new output; no hidden clone in loops | Review allocation paths; tests for independence; document logical bytes vs overhead |
| Invalid tensor state | HIGH / low | Shape/data disagreement breaks every operation | Private fields; checked constructors; invariant `data.len()==numel` | Constructor failure tests; debug invariant checks; external API compile tests if useful |
| Floating-point instability/order dependence | MEDIUM / medium | Numerically inaccurate or differing results | Define f32 storage/f64 sequential reduction order; tolerance policy; avoid fast-math | Cancellation, magnitude extremes, NaN/inf tests; independent oracle regression fixtures |
| NaN/infinity ambiguity | MEDIUM / medium | Invalid exact comparisons or accidental rejection | Explicit classification rules; accept non-finite inputs; no payload promise | Tests assert NaN classification and infinity sign; unexpected nonfinite output fails |
| Undefined behavior | CRITICAL / low if safe subset | Memory unsafety or security defects | No unsafe Rust/C in TASK 002; forbid unsafe; no raw pointer arithmetic | CI lint/source review; no regression test can prove absence, so policy is mandatory |
| FFI ownership / C pointer misuse | HIGH / low if deferred | Double free, use-after-free, ABI mismatch, corruption | No FFI in TASK 002; future FFI requires separate ADR, ownership table, narrow API | If ever added, sanitizer-enabled integration tests and explicit lifetime/error tests |
| Unsafe Rust introduced for speed | HIGH / low if policy enforced | Memory unsafety and larger audit surface | `#![forbid(unsafe_code)]`, no performance claim without benchmark | CI compile failure for unsafe; review dependencies/build scripts for native code |
| Data races | HIGH / low | Nondeterministic output or memory corruption | Single-threaded operations in this milestone; no shared mutable buffer | Concurrency absent; if added later, race detector/loom-style testing as relevant |
| Error hidden by panic or unwrap | MEDIUM / medium | Caller cannot recover from invalid input/allocation failure | Public `Result<T, TensorError>`; avoid panic-based APIs for external dimensions | Invalid input tests; code review for `unwrap` in public paths |
| Dependency supply-chain/license risk | MEDIUM / low initially | Vulnerable or incompatible transitive code; build/portability drift | Standard library only baseline; dependency review, lockfile, license/advisory policy for additions | CI advisory/license audit once dependencies exist; review build scripts and native code |
| Debug/release behavior drift | MEDIUM / low-medium | Overflow or arithmetic changes hidden in release | No reliance on debug overflow checks; explicit checked ops; run both profiles | Debug and release oracle fixtures on every PR |
| Transpose/matmul indexing bug | HIGH / medium | Incorrect results on common non-square shapes | Derive formulas in docs; simple safe loops; non-square hand examples | Rectangular tests, identity, transpose twice, oracle-generated tests |
| CI or tests mistaken for proof | MEDIUM / medium | False confidence and premature acceptance | Report exactly what tests establish; independent review; no fabricated results | PR checklist includes limitations and relevant SHA; review code and oracle separately |

## Mandatory TASK 002 policy

The baseline should forbid unsafe Rust and C FFI. A future request to enable either requires a separate proposal tied to a reproducible benchmark and a demonstrated limitation of the safe baseline. It must specify a fallback path, ownership/lifetime invariants, toolchain and target scope, sanitizer coverage, differential correctness tests, and owner approval. Theoretical vectorization or possible future speed is not sufficient evidence.

## Residual risks

Safe Rust cannot establish mathematical correctness, prevent all resource exhaustion, or guarantee that external dependencies are secure. A reference implementation can be slow. The proposed per-tensor cap may need revision when real model dimensions are known; revise it with explicit workload/resource evidence before increasing it. Numerical tolerance can hide errors if chosen too loosely, so lock tolerances after oracle calibration and retain exact hand cases.

