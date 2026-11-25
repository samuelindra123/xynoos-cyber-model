# TASK 003-B — Native reverse-mode autograd, MSE, and transactional SGD

## Scope and baseline

The implementation starts from the accepted TASK 002 main commit
`c5250543e5a39b087b7daab3d86d69a2af65a703`. The design is pinned to
`8bc693ce26f0bbb5f8c647162cd5afe39880843a` in the separate planning PR.
The implementation adds one workspace crate with a single workspace dependency,
`xynoos-tensor`. Both production crates forbid unsafe code. The tensor crate
adds a fallible snapshot copy and a read-only retained-capacity query. Its
arithmetic, errors, fixture and numerical tolerances retain their accepted
TASK 002 behavior. [Planning PR #4](https://github.com/samuelindra123/xynoos-cyber-model/pull/4)
· [Pinned design](https://github.com/samuelindra123/xynoos-cyber-model/tree/8bc693ce26f0bbb5f8c647162cd5afe39880843a/docs/design)

## Ownership, math, and transactions

`ParameterSet` owns parameters and persistent gradients. `Tape` borrows it
immutably, snapshots parameters and constant inputs, and records operation
outputs in insertion order. It verifies opaque owner identities before slot
lookup. The reverse walk is deterministic, allocates adjoints lazily, and
propagates once for each reachable edge, including repeated operands. A failed
backward seals the tape and releases temporary adjoints. A successful backward
seals forward recording and exposes the adjoints until consumption or drop.

`Tape::into_parameter_gradients` consumes the tape and yields an owned,
single-use `GradientBatch`. The batch tracks originating set and revision. Merging
multiple snapshots and accumulating microbatches use ordered binary64 addition
with binary32 storage. Missing gradients are distinct from explicit zeros.
`zero_grad` stages explicit zeros for every parameter. SGD validates inputs,
generates candidates using binary64 product and subtraction followed by one
binary32 cast, and commits every candidate together. An insertion or nonempty
SGD commit advances the revision with checked arithmetic. Only a successful SGD
commit clears gradients. Resource, validation, and numerical failures release
staging and preserve existing state.

MSE requires equal nonempty shapes and an input leaf as the constant target.
It computes differences, squares, sums, and division in binary64. Its backward
kernel multiplies by the actual upstream gradient, computes each contribution
in binary64, and casts once. Mean rejects empty inputs, while sum and dot retain
valid empty cases. The rectangular matrix kernels use sequential binary64 sums,
including the two backward products, and compute directly from row-major values.
Zero-dimensional derivatives preserve their input shapes. All tape and update
boundaries reject non-finite inputs, results, and gradients. [Public API and operation contracts](../../crates/xynoos-autograd/README.md)

## Resources

The per-set budget covers retained tensor shape and data capacities, metadata,
active snapshots, adjoints, outstanding batches, and staging across overlapping
lifetimes. Moving an adjoint carries its charge. Detached metadata growth is
preflighted against both allocations and released if actual capacity fails
reconciliation. The ceilings are 128 MiB per set, 100,000 values and operations,
and 100 million estimated work units per pass. All ceilings may be configured
downward. The tensor numerical limit remains 64 MiB. Separate boundaries cover
forward, backward, transfer, accumulation, zeroing, and SGD. Matrix estimates
include initialized cells and scalar work even for a zero inner dimension.
Allocation accounting uses vector-reported capacity and excludes allocator
bookkeeping, fixed-size handles, external inputs, stack and process RSS.

## Independent verification

The first layer checks small scalar, vector and matrix derivatives by hand.
The second checks analytical derivatives for shared and repeated edges,
polynomials, a scaled MSE composition, and both derivatives of a weighted
`[2,3] @ [3,4]` matrix product. The third compares every coordinate in those
composed cases against independent standard-library Python results and central
differences of the native forward pass. Perturbations are rounded to
binary32 before measuring the actual denominator. The primary step is
`1e-3 * max(1, |x|)`. A half step checks sensitivity to perturbation scale.

The fixed analytical tolerance is `2e-6 + 2e-5 * max(|g|, |g_ref|)`.
The fixed finite-difference tolerance is `3e-3 + 3e-2 * max(|g|, |g_fd|)`.
The fixtures contain 25 coordinates covering one scalar, six vector and
eighteen matrix coordinates. Negative controls independently introduce a sign
flip, a zeroed nonzero coordinate, and a rectangular layout permutation. The
checker first verifies the original gradient before applying any mutation.

## Synthetic learning proof

The experiment builds five scalar predictions for each objective using
`scale(w, x)` and addition of `b`. It reduces five scalar MSE terms with
addition and a constant factor of one fifth. The dataset is
`x=[-2,-1,0,1,2]` and `y=[-1,1,3,5,7]`. Weight and bias start at zero.
It uses native backward, owned transfer, gradient accumulation, and SGD at
`0.05` for exactly 100 full-batch updates with a fresh tape per observation.
An additional forward observation gives 101 states. The frozen experiment
length is checked against the 10,000-step project ceiling.

The Python reference starts from error coordinates `(-2,-3)`. It multiplies
them by `0.8` and `0.9` on each update and computes the loss as
`2*(w-2)^2 + (b-3)^2`. Each parameter observation must agree within `1e-6`.
Multiplication by the fixed integer samples is exact in this range. The
parameter tolerances permit at most `3e-6` displacement in each prediction,
plus at most half a binary32 ULP for prediction addition below magnitude
eight. The loss comparison uses the squared residual displacement bound
`2*sqrt(L_ref)*3.25e-6 + (3.25e-6)^2` plus
`6*f32::EPSILON*L_ref` for scalar MSE, ordered sum, and averaging roundoff.
The trajectory must strictly decrease, finish below both loss thresholds and
meet the original weight and bias error thresholds. CI compares all three
binary32 fields bitwise across all 101 debug and release observations.

## Validation and independent review handoff

Local verification uses formatting, Clippy, and small contract tests with two
build jobs. Foundation CI runs the full exact-head validation with Rust
1.100.0, locked offline Cargo, a 15-minute timeout, two build jobs, both
profiles, doctests, documentation, privacy checks, and fixture regeneration.
The workflow prints the fixture hashes, gradient error maxima, negative-control
verdicts, representative learning results, trajectory hash, and TASK 002
oracle results. The implementation PR remains subject to Session A review and
owner acceptance.

Session A can independently audit the owner and revision checks, both derivative
matrix index formulas, backward sealing, and transaction failures after earlier
candidates have staged. In particular, compare the predicted transfer peak with
the exact-boundary test, replay the three gradient controls, and verify the
101-state trajectory and original tensor oracle on the exact PR head.

The crate provides first-order derivatives and single-threaded per-set resource
accounting. It uses dense CPU kernels, scalar-root backward, strict shapes,
and SGD without optimizer state. The synthetic experiment is linear regression.

## Measured GitHub Actions results

The initial implementation run completed in both profiles on
`c42ff590f8f246892b9a64c1924e2c724e5be569`. The exact current PR head and
its checks are recorded on the draft PR. [Initial CI evidence](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45551198050)
· [Draft implementation PR #5](https://github.com/samuelindra123/xynoos-cyber-model/pull/5)

For the 25 gradient coordinates, the maximum absolute differences were zero
against the analytical references, `9.313225746e-8` against Python central
differences, and `8.214126062e-4` against native-forward central differences.
The same frozen comparison detected all three deliberately wrong gradients in
each profile. The Python fixtures regenerated with the following SHA-256 values.

| Fixture | Records | SHA-256 |
| --- | ---: | --- |
| Gradients | 25 | `34fe93f5813e0a0756b932452e7fa511798019026e01039404e0c14d432f1d2b` |
| Learning reference | 101 | `3b4976dc2b8783048ebe3865f0c6d9110be94a46642c7b25be20408d345d54f1` |
| Native trajectory | 101 | `e50d201a133838fe64158e0dcf94df894efcf17241f1f16e52232a4e75d41c6a` |

| Updates completed | Weight | Bias | Loss |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 17 |
| 1 | 0.4000000060 | 0.3000000119 | 12.41000080 |
| 10 | 1.785251617 | 1.953964829 | 1.186423540 |
| 25 | 1.992444277 | 2.784630775 | 0.04649810493 |
| 50 | 1.999971509 | 2.984538555 | 0.0002390571462 |
| 75 | 1.999999762 | 2.998889923 | 0.000001232164891 |
| 100 | 1.999999762 | 2.999920368 | 0.000000006348966686 |

All 101 observations agreed bitwise between debug and release. The maximum
absolute differences versus the independent trajectory were `2.380111719e-7`
for weight, `5.000470042e-7` for bias, and `8.010864239e-7` for loss. All
strict monotonicity, final loss, and final parameter thresholds passed.

The original TASK 002 numerical fixture retained SHA-256
`cdfde7ba9f44f46ba7e757157019730e1124bcd966e796b73286bddf3894fa0f`, 295 cases,
2,670 expected values, six negative controls, and its fixed tolerances. Its 32
tests continued to pass in each profile. The tensor snapshot and retained
capacity extension added a doctest-independent compatibility test.
