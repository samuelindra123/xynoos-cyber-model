# TASK 003 — Differentiation and Optimization Specification

**Status:** Planning proposal for owner review. It is not implementation authorization.

## 1. Scope and decision

Add one safe-Rust workspace crate, `xynoos-autograd`, depending only on `xynoos-tensor` and the standard library. Use a bounded, explicitly owned reverse-mode tape with a closed enum of supported operations. Each tape is a single forward/backward transaction. It owns snapshots of leaf values, forward outputs, and (during backward) adjoints; it never stores Rust references into mutable parameters. Parameters remain owned by a separate `ParameterSet`, are copied into a tape for a forward pass, and are changed only by a validated optimizer step.

This is a small reverse-mode engine for scalar losses over dense tensors. It supports ordinary neural-network backpropagation and a later path to matrix-based layers, while avoiding a general-purpose dynamic graph runtime. First-order derivatives only.

## 2. Architecture alternatives

| Option | Benefits | Costs and risks | Decision |
|---|---|---|---|
| A. Manually written backward function per whole model/layer | Minimal storage and direct control; can fuse operations | Composition and shared/repeated dependencies become caller-specific; each layer owns its chain rule and accumulation logic; validation is fragmented | Rejected as the sole architecture. Keep kernels explicitly implemented, but compose them through the tape. |
| B. Dynamic reverse-mode graph attached to values | Natural expression composition and control flow; graph follows operations | Reference-counted nodes/cycles or arena lifetime complexity; retained graph surprise; aliasing and mutation rules; repeated backward and graph release semantics; harder aggregate accounting | Rejected for this milestone. Frameworks document graph retention/release as a user-visible concern; this project does not need arbitrary dynamic lifetime. |
| C. Explicit tape with stable identifiers | Deterministic reverse traversal; arena ownership; easy whole-pass release; shared edges explicit; graph/node limits and accounting; no references or cycles | Requires operation record and saved-value policy; fixed op set; ID validation and backward propagation require careful implementation | **Selected.** Smallest architecture that composes operations while retaining predictable lifetime and resource bounds. |
| D. Hybrid explicit kernels plus minimal tape | Kernels remain hand-written and auditable; tape composes them and tracks dependencies | Slightly more machinery than direct backward APIs | **Selected as implementation form of C**, not a second extension layer: one closed tape enum invokes explicit kernels; no plugin/trait framework. |

Tape IDs are `ValueId(usize)` and `ParameterId { set_id, index }` with private fields and checked lookup. Each `ParameterSet` receives a checked, process-unique set ID; exhaustion returns a typed error rather than wrapping. This prevents a valid ID from one set being accepted by another. IDs are valid only while their owning tape or parameter set lives; public IDs do not contain pointers. A tape is never reset/reused after IDs escape. Dropping the tape releases all graph values and adjoints. A parameter set gets stable IDs at insertion and is not structurally mutated while a tape referring to it exists; Rust borrowing enforces that through `Tape::new(&ParameterSet, limits)` and the caller must end that borrow before mutable optimizer access.

The forward API takes `&mut Tape` and returns a tape-owned ID. There is no autodiff `Tensor` wrapper, aliasing view, global graph, implicit `Clone`, interior mutability, or `retain_graph` flag. Every operation output is an owned `Tensor`. The reverse walk visits recorded operation nodes in descending insertion order; nodes are appended only after all inputs and the result have been validated and budgeted. An input ID must refer to an earlier tape value. A malformed/fabricated/stale ID returns a typed error before indexing.

## 3. Tensor integration

The accepted `xynoos-tensor` API already provides owned row-major `f32` tensors, checked shape invariants, fallible result allocations, strict same-shape elementwise arithmetic, rank-1 dot, rank-2 matmul and transpose, sum, typed shape/allocation errors, and 64 MiB per-tensor logical and retained-`Vec` limits. Those semantics remain unchanged.

Minimal justified extensions, if crate-level access cannot otherwise provide the behavior:

1. A documented `try_clone(limits)` that fallibly copies shape and data under existing limits. Do not implement infallible `Clone`.
2. Read-only capacity accounting, preferably `retained_bytes() -> Result<usize, TensorError>` that checked-adds shape-vector and data-vector capacities in bytes. This is an estimate of the two Rust vector buffers only, not allocator metadata or process RSS.
3. A crate-private or public bounded constructor for already-computed `Vec<f32>` data that validates length, payload and retained capacity without copying. Existing `from_vec` may already serve this purpose.
4. If optimizer commit needs replacement without exposing shape mutation, replace a full validated tensor value in `ParameterSet`; no direct tensor shape mutation is needed.

Do not add broadcasting, views, generalized matrix operations, in-place arithmetic, implicit aliasing, or changes to TASK 002 arithmetic/non-finite behavior. Differentiation validates finite forward values at tape boundaries and returns its own typed `NonFiniteValue` error; ordinary tensor operations continue their existing primitive IEEE propagation contract.

## 4. Value, gradient, and parameter lifecycle

- `ParameterSet` exclusively owns each trainable parameter's `Tensor`, a same-shaped gradient buffer or an explicit `None` before first accumulation, and its stable `ParameterId`. Inputs are tape leaves but are not registered for optimizer mutation unless explicitly added as parameters. The aggregate session budget charges both parameter values and persistent gradient buffers.
- Parameter values are immutable during a tape forward/backward pass. `forward_parameter` copies a checked snapshot into the tape. Optimizer mutation requires an exclusive `&mut ParameterSet`, which cannot coexist with an active borrowed forward call.
- Every tape value has a shape and forward tensor. Each recorded op has input IDs and output ID; leaf records identify an optional parameter ID. `requires_grad=false` leaves have no gradient accumulation target.
- Adjoint buffers are created lazily when a value receives its first contribution. Contributions are added elementwise in deterministic reverse-node and row-major order. Repeated operands contribute once per edge: `x*x` adds `g*x` twice; `x*x+x` adds both product-edge contributions and the direct add edge.
- `backward(root, seed)` accepts a scalar root and a scalar seed (normally `1.0`). It is legal once per tape. A second call returns `BackwardAlreadyRun`; this prevents stale saved values, accidental double accumulation, and ambiguous graph retention. Parameter gradients from a completed tape are accumulated into `ParameterSet` explicitly with shape-checked `accumulate_parameter_gradients`; accumulation uses ordered `f64` addition then casts to `f32`, and rejects non-finite results. This allows microbatch accumulation without repeated backward on one tape.
- Clear parameter gradients explicitly before a new optimization batch using `zero_grad`; by default SGD clears them only after a successful committed step. An optimizer with `clear_grad=false` may be offered only if API simplicity and tests support it; recommended baseline is always clear on successful step.
- Disconnected values have no adjoint (`gradient(id)` returns `None`); a disconnected parameter gets a zero gradient at optimizer preflight only if the caller opted into `zero_grad`, otherwise `MissingGradient` is returned. The recommended simple policy is `ParameterSet::zero_grad()` before forward passes, which gives all parameters zero buffers and permits disconnected parameters to remain unchanged.
- On any backward error, no parameter gradients are modified. Compute adjoints in tape-owned temporary buffers, validate all shapes/finiteness and budget, then mark backward complete and expose gradients. On parameter-gradient accumulation error, validate all candidates before committing any. Tape failure may discard the tape; it never poisons parameters.
- On optimizer failure, parameter values and gradients are unchanged. Tape and its snapshots can be dropped after gradients have been copied/accumulated.

## 5. Exact mathematical contracts

Let each tensor be a finite real-valued array represented in `f32`. Formulas below are real-valued derivatives; the implementation computes products in `f32` unless a reduction/accumulation is explicitly required in `f64`, then checks finite `f32` results. All elementwise operations require identical shapes; no broadcasting. `g = dL/d(output)` must have exactly the output shape. The seed for a scalar root is rank zero.

| Operation | Forward and shapes | Backward contributions | Empty and error rules |
|---|---|---|---|
| Add | `z=a+b`; all shape `S` | `ga += g`, `gb += g` | Empty `S` is valid; zero elements produce no writes. Exact shape required. |
| Subtract | `z=a-b`; shape `S` | `ga += g`, `gb -= g` | Same as add. |
| Elementwise multiply | `z=a⊙b`; shape `S` | `ga += g⊙b`, `gb += g⊙a` | Empty valid. Save/snapshot both forward operands. Repeated input ID adds both contributions. |
| Scalar scale | `z=c a`; shape `S`, finite scalar `c` | `ga += c g`; scalar is a constant, not a differentiable value in this API | Empty valid. Non-finite scalar/result rejected by tape. |
| Sum | `z=Σ_i a_i`; scalar output | `ga_i += g` for every input element | Empty input sum is valid and equals `+0`; gradient has input shape and is empty. |
| Mean | `z=(1/n)Σ_i a_i`; scalar output | `ga_i += g/n` | Reject `n=0` with `EmptyReduction`; otherwise f64 sequential sum/divide followed by finite f32 result. |
| Dot | rank-1 `[n]` inputs, scalar `z=Σ_i a_i b_i` | `ga_i += g b_i`; `gb_i += g a_i` | Equal lengths required. `n=0` is valid and returns `+0`, with empty gradients. Forward accumulation in existing tensor API is sequential f64, cast once. |
| Matmul | `A[m,k] B[k,n] = C[m,n]` | `dA = G[m,n] Bᵀ[n,k]` (`m×k`); `dB = Aᵀ[k,m] G[m,n]` (`k×n`) | Rank 2 and matching inner dims. Rectangular dimensions required in tests. Zero dimensions follow tensor shape rules; empty output has no gradient writes. For `k=0`, output is zero and derivative arrays are zero-sized or shape-correct zeros. |
| Transpose | `Y=Aᵀ`, `[m,n] -> [n,m]` | `dA=(dY)ᵀ`, `[m,n]` | Rank 2. Empty dimensions valid. |
| Mean squared error | `L=(1/n)Σ_i(p_i-t_i)^2`; `p,t` same nonempty shape; scalar output | `dL/dp_i = (2/n)(p_i-t_i)`; target is constant and receives no gradient | Empty rejected. Compute each difference/square and sequential f64 reduction; finite-check result. Target is a non-differentiable input leaf. |

For matmul backward, element formulas are `dA[i,q] = Σ_j G[i,j] B[q,j]` and `dB[q,j] = Σ_i A[i,q] G[i,j]`; sums use sequential f64 accumulation in increasing index order and one f32 cast. Accumulate an existing contribution with deterministic f64 add/cast. These formulas cover rectangular matrices and avoid requiring a public tensor view.

The tape's op set is fixed to `Add`, `Subtract`, `Multiply`, `Scale`, `Sum`, `Mean`, `Dot`, `Matmul`, `Transpose`, and `MeanSquaredError`. Only scalar-root reverse-mode is required. This is sufficient for vector/matrix neural layers and scalar losses. Softmax, cross-entropy, activations, convolution, and batch broadcasting are deferred; add an op only with a concrete consumer and full derivative/error/resource tests.

### Non-finite and rounding behavior

Tape leaves, constants, forward outputs, supplied upstream seed, gradient contributions, and final accumulated gradients must all be finite. Reject NaN and either infinity with typed `NonFiniteValue { stage, value_id/index }`; do not attempt to canonicalize NaNs. This stricter AD boundary protects optimizer updates while preserving TASK 002 behavior. `f32` storage means gradients can underflow/overflow; finite-check after cast. Reductions and each dot-product row use sequential f64 accumulation, not reassociation. Elementwise kernels use f32. No bitwise cross-platform guarantee is made.

## 6. SGD contract

`Sgd::new(learning_rate)` requires a finite, strictly positive `f32` learning rate. A step uses `p_new = p_old - learning_rate * grad` elementwise; all tensors must have identical shapes and all input values, gradients, and candidate outputs must be finite. No momentum or optimizer state is needed. Gradient is cleared only after all parameter replacements commit successfully.

The step is transactional for expected errors: validate parameter/gradient inventory, shapes, finiteness, compute/work budget, and aggregate temporary bytes; build all candidate parameter tensors into a staging collection using fallible allocation; validate every candidate; then replace all parameter values and clear all gradient buffers. Any validation, allocation, or budget error drops staging tensors and leaves parameter values and gradient buffers unchanged. Rust allocation aborts outside fallible Vec reservation are not claimed to be recoverable. An empty parameter set is a valid no-op. Complexity is O(P) arithmetic plus O(P) staging bytes, where P is the total parameter element count; the staging copy is accounted in budget.

Adam is deferred. It adds two full-size state arrays, bias-correction and step-overflow semantics, more numerical edge cases, and duplicate verification burden without being needed for a linear-regression learning proof. Revisit only when an experiment shows SGD is inadequate.

## 7. Public API sketch

Names may adjust in implementation, but behavior may not:

```rust
pub struct ParameterId(usize);
pub struct ValueId(usize);
pub struct ParameterSet { /* owned values and gradients */ }
pub struct Tape<'p> { /* bounded values and op records; borrows parameters immutably */ }
pub struct ResourceLimits { /* aggregate caps below */ }
pub struct Sgd { learning_rate: f32 }

impl ParameterSet {
    fn new(limits: ResourceLimits) -> Self;
    fn insert(&mut self, initial: Tensor) -> Result<ParameterId, AutogradError>;
    fn value(&self, id: ParameterId) -> Result<&Tensor, AutogradError>;
    fn zero_grad(&mut self) -> Result<(), AutogradError>;
}
impl<'p> Tape<'p> {
    fn new(parameters: &'p ParameterSet, limits: ResourceLimits)
        -> Result<Self, AutogradError>;
    fn input(&mut self, value: &Tensor) -> Result<ValueId, AutogradError>;
    fn parameter(&mut self, set: &ParameterSet, id: ParameterId)
        -> Result<ValueId, AutogradError>;
    fn add(&mut self, a: ValueId, b: ValueId) -> Result<ValueId, AutogradError>;
    // corresponding methods for each specified operation
    fn backward(&mut self, scalar_root: ValueId, seed: f32)
        -> Result<(), AutogradError>;
    fn gradient(&self, id: ValueId) -> Result<Option<&Tensor>, AutogradError>;
}
impl ParameterSet {
    fn accumulate_from(&mut self, tape: &Tape) -> Result<(), AutogradError>;
}
impl Sgd {
    fn step(&self, parameters: &mut ParameterSet, limits: &ResourceLimits)
        -> Result<(), AutogradError>;
}
```

Every operation returns `Result`, uses checked IDs and shapes, and reserves before allocating. Avoid generic operation traits, backend traits, public mutable tensor internals, reference-counted nodes, and user-defined backward closures.

## 8. Typed errors

`AutogradError` should distinguish at least: wrapped `TensorError`; invalid/stale `ValueId`; invalid `ParameterId`; shape/rank mismatch; non-scalar backward root; missing upstream/gradient; repeated backward; empty mean/MSE; non-finite input/output/gradient/learning rate; aggregate byte or retained-capacity budget exceeded; node/value/op count exceeded; operation count/work budget exceeded; checked arithmetic overflow; failed reservation. Error values identify the offending operation/value/index and observed/limit amounts without embedding machine-specific details. Propagate `TensorError` rather than converting it to panics or generic strings.
