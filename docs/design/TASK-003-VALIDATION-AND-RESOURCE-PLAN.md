# TASK 003 — Validation and Resource Plan

**Status:** Planning proposal. Bounds below are proposed defaults for review and must be fixed in public API/constants before implementation.

## 1. Aggregate resource contract

Existing tensor limits apply independently to every tensor and do not bound an autodiff pass. Add one `ResourceLimits` policy shared by parameter sets, tapes, and optimizer steps. Suggested initial ceilings: 128 MiB total simultaneously accounted retained owned tensor-buffer bytes and metadata for one active training session; 100,000 tape values; 100,000 operation nodes; 100,000,000 estimated scalar work units per pass; and 10,000 synthetic optimizer steps. Keep the existing 64 MiB cap on each tensor. Limits are configurable downward for tests; an API may not raise the project-wide ceilings. The 128 MiB session ceiling includes parameter values and persistent parameter gradients, active tape leaf snapshots/forward values/adjoints, operation/value metadata, scratch buffers, and SGD staging. SGD has no persistent optimizer state. The tape borrows the parameter set immutably so its baseline cannot change during forward/backward.

Accounting must include:

- Each parameter, gradient, tape value, and temporary tensor's logical `numel * size_of::<f32>()` and retained `Vec<f32>` capacity. These overlap conceptually, so charge retained capacity once and separately report logical payload; enforce both the per-tensor ceilings and the session aggregate. Include retained shape-vector capacity if exposed by the minimal tensor accounting extension.
- Tape value and operation vector capacities, ID arrays/edge metadata, and parameter table capacities, measured with checked `capacity * size_of::<T>()` and checked summation.
- Parameter value tensors, persistent parameter gradient buffers, tape adjoints, staged parameter outputs during transactional SGD, explicit temporary buffers, and scratch space.
- A checked conservative node/op metadata charge before vector growth. After `try_reserve`, reconcile against actual capacity before publishing the node/value.
- Compute estimates by checked shape arithmetic before executing: elementwise and scale `N`; sum/mean `N`; dot `N`; transpose `N`; matmul forward/backward `2*m*k*n` each pass direction (count multiply-add as two scalar operations). MSE `4*N` conservative. Include all backward work and optimizer element updates. Reject if cumulative units exceed limit.

The `xynoos-tensor` extension should report capacity bytes for its owned vectors because the AD crate cannot otherwise verify actual retained tensor capacity. Tape accounting may conservatively reserve based on requested elements before allocation and reconcile actual capacity afterward. On accounting rejection, discard the local staged output; existing tape state and parameters remain unchanged. `try_reserve` is required before vector growth. Vec capacities can exceed request; actual retained capacity is charged. If post-reservation reconciliation exceeds the budget, drop the candidate and return budget error.

Accounting excludes allocator bookkeeping, alignment/padding outside measured vector capacity, stack frames, code/static memory, OS page tables, unrelated process memory, and any OS overcommit. It is an application-level logical/retained-capacity estimate, not an RSS or OOM guarantee. Never claim exact process memory control; do not add a custom allocator or OS resource manager.

Tape lifetime is one training forward/backward pass. Dropping tape releases its owned vectors/tensors. Do not retain tapes across optimizer steps. Build a fresh tape for each batch/step; transfer gradients into `ParameterSet`, then drop tape before optimizer staging. Account a conservative peak for active parameters/gradients plus the active tape; during SGD charge parameters/gradients plus all staged candidates. Graph growth is rejected before exceeding node/value/work/byte limits. There is no graph reset that could make old IDs appear valid.

## 2. Verification method

Use three independent evidence layers:

1. Hand-derived tiny cases with exact expected gradients: scalar arithmetic, reduction, dot, transpose, and rectangular matmul.
2. Analytic composed functions and shared/repeated dependencies, covering the chain rule and accumulation.
3. Central finite differences on a deterministic set of finite, smooth, well-scaled inputs. The numerical checker is test-only and must call the forward function independently from backward; never use finite differences in production.

For scalar objective `f`, coordinate `i`, and `h_i = c * max(1, |x_i|)` with `c = 1e-3` for `f32` inputs, compare `g_i` to `(f(x+h_i)-f(x-h_i))/(2h_i)`. Evaluate perturbed forward points in `f64` using an independent standard-library Python oracle when feasible; separately test the production `f32` path. Use bounded smooth domains away from zero-derivative ambiguity, cancellation-heavy inputs, overflow, and non-differentiable points. Start acceptance at `abs_error <= 3e-3 + 3e-2*max(|g|, |g_fd|)` for `f32` finite differences; calibrate by documented bounded reference experiments before freezing. Use tighter exact/analytic tolerances for direct derivatives, e.g. `atol=2e-6`, `rtol=2e-5` on small well-conditioned cases. Do not inherit TASK 002 forward tolerances without gradient-specific calibration.

For finite-difference checker sensitivity, include negative controls that supply deliberately wrong gradients (sign-flipped, zeroed nonzero coordinate, and swapped rectangular matrix gradient) and assert the checker rejects each. Do not present checker agreement alone as proof: analytic and hand calculations are primary.

## 3. Minimum test matrix

| Area | Required cases |
|---|---|
| Basic derivatives | Scalar add/subtract/multiply/scale; hand derivative for `x*x`; scalar sum/mean; vector dot; rank-2 transpose. |
| Chain rule | `f(x)=(3x+2)^2`; `f(x)=sum((a⊙b)+a)`; MSE through multiply/add; compare analytic derivative. |
| Shared graph | `x*x+x`; use the same `ValueId` in both multiply input positions; two downstream branches joined at one node; assert contributions add in deterministic order. |
| Rectangular matrices | `A[2,3] @ B[3,4]` with nonuniform values/upstream `G[2,4]`; independently calculate every `dA[2,3]` and `dB[3,4]`; include transposed/empty dimensions. |
| Loss | MSE hand values and derivative, target constant, shape mismatch, empty rejection, large finite values that overflow rejected, NaN/infinity rejection. |
| Lifecycle | Zero-grad, accumulated two tapes/microbatches, no-grad/disconnected parameter unchanged after explicit zeroing, repeated backward rejected, invalid/stale ID rejection, tape drop releases budget, parameter mutation unavailable during tape borrow. |
| Optimizer | Exact one-step SGD; invalid LR (zero, negative, NaN, ±infinity); missing gradient policy; mismatched shapes; non-finite gradients; candidate overflow; allocation/budget error leaves every parameter and gradient unchanged; successful step clears gradients. |
| Numerical checker | Central-difference scalar/vector/matrix coordinates; vary `h`; independent Python standard-library analytic/finite difference fixtures; three bad-gradient negative controls. |
| Resource | Synthetic tiny aggregate byte/node/value/work limits; exact-at-limit and one-over-limit; checked arithmetic overflow; capacity overrun after reservation; forward and backward failure does not mutate parameter state; bounded learned-step limit. |
| Determinism | Same fixed inputs/seed in debug and release; compare finite outputs and gradients using documented tolerances; repeat learning experiment exactly within tolerance. |
| Regression | Preserve existing TASK 002 unit, contract, oracle (295 cases/2,670 values), retained-capacity, privacy/artifact and documentation checks unchanged. |

All tests use small shapes and deterministic fixture values. No actual OOM test, fuzz infrastructure, heavy framework, downloaded model, or dataset. Synthetic budget tests reject before large allocation.

## 4. Deterministic learning proof

Train `y = 2x + 3` from initial `w=0`, `b=0` on fixed scalar pairs `x=[-2,-1,0,1,2]`, `y=[-1,1,3,5,7]`. Use mean squared error across all 5 examples and batch gradients:

`L(w,b) = (1/5) Σ (w*x_i+b-y_i)^2`

`dL/dw = (2/5)Σ x_i(w*x_i+b-y_i)`

`dL/db = (2/5)Σ (w*x_i+b-y_i)`.

Use fixed learning rate `0.05` and 100 full-batch updates. Dataset symmetry gives `Σx=0`, `Σx²=10`, so the updates decouple: `w_(t+1)-2=0.8(w_t-2)` and `b_(t+1)-3=0.9(b_t-3)`. For `t=100`, expected weight error is under `5e-10` and bias error under `8e-5`. Require final `|w-2| <= 1e-5`, `|b-3| <= 1e-4`, final loss `< 1e-8`, and final loss `< initial_loss * 1e-6`. Record initial and per-step loss, final parameters, and step count. A separate standard-library Python reference computes the same full-batch recurrence and loss trajectory; compare each step within the f32-calibrated tolerance. This is a synthetic numerical learning proof, not language-model training.

Each experiment iteration creates a fresh bounded tape for five predictions, forms MSE, runs backward, accumulates gradients, runs transactional SGD, and drops the tape. Keep total steps fixed at 100, no random initialization, no pretrained data, no external framework. The test asserts trajectory is finite, deterministic, monotonic for this convex setup, and ends at the specified thresholds. If f32 implementation/reference error violates these provisional thresholds, adjust only after recording the cause and an independent analytical bound.

## 5. CI policy

Extend the existing single PR validation job rather than adding parallel/heavy jobs. Keep pinned Rust `1.100.0`, locked/offline dependency resolution, zero new dependencies absent a concrete review, `cargo fmt --all -- --check`, Clippy `-D warnings`, debug and release workspace tests, doctests, and docs. Add bounded AD tests and an independent standard-library Python gradient/learning oracle with deterministic fixtures and explicit expected fixture counts/hashes. Preserve current numerical oracle untouched. Keep Actions pinned by commit SHA, permissions read-only, exact PR-head checkout, `timeout-minutes: 15`, `CARGO_BUILD_JOBS=2`, concurrency cancellation, no network downloads in tests, and artifact/privacy checks. CI reports the exact tested SHA and all test results. Documentation assertions/checks should ensure acceptance docs and scope claims remain current.

Heavy validation is unnecessary for this milestone. Do not add paid compute, jobs with unbounded matrices, benchmarks, actual OOM tests, GPU runners, or ML package installation.
