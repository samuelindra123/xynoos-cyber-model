# TASK 006 — AdamW Numerical Specification

**Status:** proposed native FP32 reference convention for TASK 006-B. Not implemented.

## 1. Normative equations

For parameter coordinate i, let p be its current FP32 value, g the normalized and clipped FP32 loss gradient, m and v its FP32 first and second moments, and t the one-based optimizer step being staged. Hyperparameters are stored as finite FP32 configuration values and widened to FP64 for arithmetic. Let b1 and b2 be in [0,1), epsilon > 0, learning rate eta > 0, and group decay lambda >= 0.

The stored moments are rounded once to FP32 after these FP64 calculations:

    m* = b1 * widen(m) + (1 - b1) * widen(g)
    v* = b2 * widen(v) + (1 - b2) * widen(g) * widen(g)
    m_t = round_f32(m*)
    v_t = round_f32(v*)

The bias powers are FP64 scalar state initialized to one and advanced only by a successfully committed update:

    q1_t = q1_(t-1) * widen(b1)
    q2_t = q2_(t-1) * widen(b2)
    c1_t = 1 - q1_t
    c2_t = 1 - q2_t
    mhat = widen(m_t) / c1_t
    vhat = widen(v_t) / c2_t

The candidate parameter is evaluated in FP64 in this order:

    decay_scale = 1 - widen(eta) * widen(lambda)
    decayed = decay_scale * widen(p)
    denominator = sqrt(vhat) + widen(epsilon)
    adaptive = widen(eta) * mhat / denominator
    p_candidate = round_f32(decayed - adaptive)

Here round_f32 means IEEE-754 binary32 round-to-nearest, ties-to-even. The staged m_t, v_t and p_candidate must each be representable finite FP32 values. All operations use separate FP64 multiply and add; do not contract multiply-add operations. The denominator adds epsilon after square root of the bias-corrected second moment. The weight-decay factor is applied to the old parameter before subtraction of the adaptive loss-gradient step.

This selects the bias-corrected-moment convention in the Adam paper and the decoupled shrinkage in AdamW. It rejects adding epsilon inside sqrt, adding weight decay to g, dividing an uncorrected moment by a post-scaled learning rate, or applying decay after the adaptive update. Equivalent real-number algebra may differ in finite precision; this operation order is normative.

## 2. Step transaction and state ownership

AdamW is proposed inside xynoos-autograd, alongside the existing Sgd, because ParameterSet owns the parameter values, persistent gradients, revision, identity, and shared resource ledger. Keeping the optimizer transaction there avoids a cross-crate commit in which model weights could change while moment allocation fails.

AdamW::step(&mut ParameterSet, eta) first validates optimizer owner, ordered inventory fingerprint, expected parameter revision, complete gradient inventory, shape equality, finite inputs, finite moments and hyperparameters, checked step increment, checked bias powers, work and memory budget, and the decay guard. It computes every candidate parameter and both moment inventories into detached charged staging. It commits by infallible swaps only after all candidates pass. It then commits step, bias powers, and new expected parameter revision and clears gradients. Any error before that point preserves parameter bits, existing moment bits, step, powers, revision, and gradients. Temporary charges are released.

The optimizer is bound at construction to the runtime parameter-set owner, its current revision, and an ordered structural fingerprint over tensor index, rank, and shape. Another model instance with numerically equal values is still a foreign owner. Any insertion or out-of-band update makes the optimizer stale. Every successful AdamW update advances the expected revision exactly once. The Transformer/checkpoint layer separately verifies canonical tensor names and tied-inventory identity; xynoos-autograd does not depend on xynoos-transformer. The runtime owner number is not serialized into a checkpoint; loading verifies the complete persisted inventory and attaches restored state to the newly created runtime owner.

The step counter uses checked u64 addition before any allocation or mutation. A configured experiment has a stricter step cap of at most the existing 10,000-step ceiling. A test may show that u64::MAX-1 can stage the final representable index and u64::MAX rejects the next update without mutation; experiment configuration will never approach those values.

## 3. Validation and narrowing boundaries

- Learning rate: finite and strictly positive.
- Beta1 and beta2: finite and in [0,1). Zero is valid and gives an immediate moment estimate; one is invalid because bias correction would be zero.
- Epsilon: finite and strictly positive.
- Weight decay: finite and nonnegative.
- Decay guard: require eta * lambda < 1 for every applied schedule rate. This project guard prevents parameter sign inversion from an unreasonable shrinkage setting; it is stricter than the mathematical AdamW definition.
- All parameter, gradient, moment, bias-power, corrected-moment, denominator, update, and candidate values are checked for finiteness at the point they are consumed or published. Second moments must remain nonnegative.
- FP64 intermediate values are narrowed only at the stored FP32 moment and final FP32 parameter boundaries. A narrowed infinity, NaN, or non-finite candidate is a rejection. Underflow to a finite signed zero follows IEEE-754 and is accepted.
- No partial parameter table or moment table may be visible. Staging allocation, shape, resource, arithmetic, revision, or numerical failure preserves the committed state.

If a first-step gradient is exactly zero, both new moments remain zero while nonzero weight decay still shrinks a decayed parameter. On a later zero-gradient step, the prior moments continue to decay and can still produce a nonzero adaptive update; decoupled weight decay is also applied. Do not special-case zero gradients by skipping the transaction.

The scalar bias powers are FP64, not per-parameter optimizer slots. They are serialized so resume does not recalculate a potentially different power sequence. Moments and model parameters are FP32. The future optimizer snapshot exposes immutable values for tests/checkpointing and does not expose mutable internal vectors.

## 4. Parameter groups and decay exclusions

Group assignment follows the accepted inventory; it is not inferred from runtime names using substring rules.

| Inventory members | Rank | Decay |
|---|---:|---|
| Shared embedding/output matrix E | 2 | Yes, once |
| Q, K, V, O, gate, value, down matrices in each block | 2 | Yes |
| Attention RMSNorm gains, FFN RMSNorm gains, final RMSNorm gain | 1 | No |

There are 57 rank-2 tensors and 17 rank-1 gain tensors. There are no bias tensors. The tied embedding/output weight is one parameter and receives one moment pair and one decay operation; the two graph paths already contribute to one gradient. A future architecture with biases or other parameter families needs a versioned explicit group policy rather than silently inheriting these rules.

Groups may set independent learning-rate multipliers or decay values only when the experiment configuration records them in canonical inventory order and the checkpoint stores them. TASK 006-B should initially support only the two groups above and one shared schedule/rate. This avoids complexity without restricting the accepted architecture.

## 5. Alternatives rejected for the reference

- **Coupled L2:** replacing g with g + lambda*p puts regularization into both moments and the adaptive denominator. It is not decoupled weight decay for Adam. The oracle has a negative control with a materially different result.
- **Epsilon inside square root:** sqrt(vhat + epsilon) changes small-gradient scaling. The reference puts epsilon outside sqrt after bias correction.
- **FP64 persistent moments:** improves storage precision but doubles optimizer storage and no accepted mixed-precision requirement motivates it. FP64 remains the independent calculation type.
- **FP32-only update arithmetic:** conflicts with the accepted reference convention that uses FP64 reductions and one FP32 publication boundary.
- **Per-parameter clocks or lazy sparse slots:** the Transformer is dense and every parameter has a gradient; one checked global step is simpler and deterministic.
- **Decoupled decay before/after update variants:** the reference computes decay scale and decayed parameter before applying the adaptive term, then narrows once. Do not interchange operations based only on real-number algebra.

The original Adam paper defines adaptive first/second moments and bias correction; Loshchilov and Hutter identify why L2-gradient regularization and weight decay differ for adaptive methods. This specification independently derives and fixes the project-specific storage and operation order rather than copying framework code. See [Adam](https://arxiv.org/abs/1412.6980) and [AdamW](https://openreview.net/forum?id=Bkg6RiCqY7).

## 6. Independent reference fixture

tools/adamw_reference.py is a Python-standard-library mathematical oracle. It models FP32 storage with explicit pack/unpack narrowing and FP64 intermediate operations. Frozen expected bytes are in tools/fixtures/task006-adamw.tsv. The oracle does not call Rust or derive expected values from a production optimizer.

The fixture covers a first step, four mixed positive/negative/zero-gradient updates, zero-gradient decay, epsilon placement, bias correction, invalid hyperparameters, f32 moment overflow, non-finite gradient rejection, u64 step overflow, and weighted microbatch accumulation/clipping. Controls must fail against deliberately coupled L2 decay, omitted bias correction, epsilon inside sqrt, and equal-weight microbatch averaging. Failure cases assert that the input state is unchanged.

CI runs the oracle in check-only mode and requires exact fixture reproduction. TASK 006-B must additionally compare its Rust implementation against this frozen independent fixture and retain independent Rust-side transaction/resource tests. The Python result is not regenerated from Rust output.
