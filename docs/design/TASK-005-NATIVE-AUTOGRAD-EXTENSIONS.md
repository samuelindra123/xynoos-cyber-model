# TASK 005 — Native Autograd Extension Design

**Status:** design proposal only; Session B must review the operation/API and resource contracts before implementation.

## Current boundary

`xynoos-tensor` stores contiguous row-major f32 tensors, rank <=32, checks each logical and retained numerical buffer against 64 MiB, and supports shape validation, elementwise add/subtract/multiply/scale, sum/dot, rank-2 matmul and transpose. Matmul/dot use increasing-index f64 accumulation, narrowing once to f32. `xynoos-autograd` has first-order reverse mode, owned single-pass tapes, finite checks, explicit persistent gradients, revisioned parameter IDs, repeated parameter snapshot merging, and transactional SGD. A tape error seals it; backward errors publish no adjoints. `ResourceLimits` caps aggregate live charges at 128 MiB, 100,000 values, 100,000 operations, and 100,000,000 work units per forward/backward/transfer/etc. phase. Default values equal the maxima.

Missing for a Transformer: rank-3 batched matmul or a rank-2-per-sample equivalent; gather/scatter-add; general reshape/permutation; rowwise reductions; RMSNorm; stable masked softmax; RoPE; SiLU/SwiGLU; masked next-token cross-entropy; exact attention backward; model config and batch validation. Tensor's current elementwise operations require identical shapes and do not broadcast.

## Recommended minimal extension

Keep Tensor itself general but small: add only checked `reshape_copy`, `permute_copy` for explicit permutations, and a batched matmul with fixed rank-3 `[B,M,K] x [B,K,N] -> [B,M,N]` and no implicit broadcast. Add model-specific typed operations to autograd rather than expanding arbitrary tensor broadcasting. First version should use selective fused tape ops for numerically coupled or expensive structures and primitive composed operations for simple pointwise/residual arithmetic.

Proposed dedicated op variants: `EmbeddingGather(E, ids, valid)`, `Rope(q, positions, head_dim)`, `MaskedCausalAttention(q,k,v,key_valid,query_valid)`, `RmsNorm(x,gamma,epsilon)`, `Silu(x)`, `SwiGlu(g,w)`, and `MaskedNextTokenCrossEntropy(logits,targets,loss_mask)`. Matmul/batched matmul remains one op per projection. A single attention op must compute QK scores, scaled stable softmax, and PV output (or a shared QKV attention op); save only bounded state required by backward: q/k/v, probabilities for explicit reference mode, masks, dimensions. Do not save exponentials or duplicate score and probability matrices without accounting. Memory efficient/tiled attention is a later op version and must match reference outputs/VJPs.

Use ordinary primitive adds, multiplies, scales, and matrix multiplies for residuals and three FFN projections; fuse rowwise normalization and SwiGLU to avoid enormous tape node counts. Each operation owns validated shapes, immutable input references via node indices, exact output shape, deterministic work estimate, saved-state byte estimate, finite checks, and explicit backward VJP. No generic arbitrary-axis framework, automatic broadcasting, higher derivatives, or silent fallback.

## Gradient and lifecycle semantics

All gradients are same-shape f32; reductions into shared axes use f64 sums in fixed order followed by f32 conversion. Repeated embedding IDs scatter-add. Repeated parameter snapshots and tied E contributions sum in deterministic reverse node order. Masks/IDs are non-differentiable constants. Parameter-set revision is captured when tape starts; any permitted model mutation invalidates old gradient batches. Existing Rust borrow rules prohibit mutation while a tape exists. Gradient accumulation does not change revision; a successful optimizer step advances it. Any shape, finite, work, or budget error during forward/backward publishes no new value/adjoint/gradient batch and releases all temporary charges. Existing accepted SGD all-or-nothing staging remains unchanged.

## Operation accounting

The existing ledger accounts for retained f32 buffers plus shape metadata and tape/adjoint table capacities, not allocator overhead/RSS. New saved attention state and opaque op metadata must be separately charged. Do not store uncharged Vecs inside `Op`. Preflight checked dimensions, bytes, work and capacity; reserve before allocation; reconcile actual Vec capacity before publishing. A new operation's work contract must bound forward and backward separately and include mask scans, reductions, scatter additions, nonlinear work and matrix contributions. Keep `ResourceLimits` existing defaults and caps intact. If a model-specific ceiling is needed, introduce an explicitly named, separately bounded model policy after evidence from oracle fixtures and a resource prototype; do not globally raise the legacy maximum as incidental fallout.

Initial inference-only forward shape targets may be below 100M work. A conservative first-training fixture estimate for N=8,D=96,F=256,B=1,T=32 is about 60M floating multiply/add operations forward and up to about 180M forward+backward arithmetic for dense products, before elementwise work. The existing per-phase 100M unit definition may therefore be insufficient depending on precise op accounting. Session B must either demonstrate a checked accounting plan within existing limits or submit a narrowly bounded model-specific work-limit proposal; it must not relabel FLOPs as existing work units. The 128 MiB aggregate ceiling should remain and be measured with all simultaneously live params, tape outputs, adjoints, saved attention probabilities and staged gradients.

## Numerical and error contracts

No op may use negative infinity as an internal tensor marker because all current differentiable values are finite-only. Mask by excluding keys in the softmax loop. Stable softmax subtracts maximum over allowed entries and uses f64 exp/sum; empty allowed row gives exact zero output/gradient. Cross-entropy uses max-shifted log-sum-exp. RMSNorm uses f64 sum-of-squares plus epsilon. Every output and VJP is checked finite. Input/config errors, arithmetic overflow, allocation failure, value/op/work/memory limits, and nonfinite stages remain typed and distinguishable. A failed reverse pass seals/discards adjoints as today. No approximate gradient, lossy storage or hidden reduction-order changes.

Each new tape operation requires: standard-library Python oracle independent of Rust; hand-derived scalar/small matrix fixture; multi-coordinate central finite differences at more than one step size; repeated-edge/shared-parameter tests; nonfinite/error rollback tests; exact output/gradient shapes; resource boundary tests; and negative controls that deliberately change scaling, mask, axis or derivative and are rejected by expected-output comparison.

## Optimizer boundary

TASK 005-B may verify gradients without applying any update, or use existing transactional SGD on a tiny fixture to prove integration. It must not add AdamW. TASK 006 should separately specify AdamW's f32 parameter and moment precision, f64 update intermediates, bias correction and epsilon placement, decoupled weight decay exclusions (embedding/norm choice explicit), deterministic step counter, checkpoint state, overflow/nonfinite rules, aggregate moment memory, atomic staging/rollback, revision updates, and independent oracle before implementation.
