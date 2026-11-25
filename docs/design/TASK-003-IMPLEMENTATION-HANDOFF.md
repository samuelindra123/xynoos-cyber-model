# TASK 003 — Session B Implementation Handoff

**Recipient:** Future Session B, after explicit project-owner implementation authorization.
**State:** Planning only. This document and its PR do not authorize production code.

## Selected architecture

Implement one `xynoos-autograd` crate over the accepted `xynoos-tensor` crate. Use a bounded, explicit reverse-mode tape with private checked `ValueId`/`ParameterId`, a closed enum of operation records, owned forward values and saved inputs, deterministic reverse insertion-order traversal, lazy adjoints, and explicit accumulation. Every tape runs backward once and is dropped after gradient transfer. Parameters are owned by `ParameterSet`; forward uses immutable snapshots; mutation happens only through transactional SGD. All production Rust remains safe, standard-library-only, and first-order.

## Rejected alternatives

- Standalone manual backward functions do not compose/shared-accumulate robustly as layers grow.
- Dynamic reference-linked graphs add lifetime, cycle, retention and repeat-backward complexity unnecessary for this milestone.
- No generalized plugin/backend/trait architecture, arbitrary closures, global graph, optimizer crate, or higher-order derivatives.
- Adam is deferred; momentum state is not required to establish a native learning path.

## Exact operation contracts

Use the forward and backward equations in [the differentiation specification](TASK-003-DIFFERENTIATION-SPEC.md), including:

- Add: `z=a+b`; `da+=g`, `db+=g`.
- Subtract: `z=a-b`; `da+=g`, `db-=g`.
- Elementwise multiply: `z=a⊙b`; `da+=g⊙b`, `db+=g⊙a`.
- Scale: `z=c*a`; `da+=c*g`; scale is a finite constant.
- Sum: scalar `z=Σa`; each `da_i+=g`; empty sum is zero.
- Mean: scalar `z=Σa/n`; each `da_i+=g/n`; reject empty.
- Dot: rank-1 equal shape; `da_i+=g*b_i`, `db_i+=g*a_i`; empty dot returns zero.
- Matmul: `[m,k]@[k,n]`; `dA=G@Bᵀ`, `dB=Aᵀ@G`, with sequential f64 inner accumulation and exact output shapes.
- Transpose: `Y=Aᵀ`; `dA=(dY)ᵀ`.
- MSE: nonempty equal shapes, `L=Σ(p-t)^2/n`; `dp=2(p-t)/n`; target is constant.

All elementwise shapes match exactly; no broadcast. Upstream gradient must match output shape; backward root is scalar. Contributions add by edge, including repeated IDs (`x*x`) and shared subgraphs. Tape boundary values and all derivatives must be finite. Existing tensor non-finite behavior remains unchanged.

## Lifecycle and error semantics

- Inputs are immutable tape leaves; parameters have stable IDs and owned tensors/gradients.
- Tape snapshots parameter values. No parameter mutation while a tape borrow exists.
- `backward` runs once; second call returns `BackwardAlreadyRun`. A bad/stale ID, shape/rank error, empty mean/MSE, non-finite value, allocation failure, or budget excess is typed and never panics.
- Use temporary tape adjoints and validate before publishing them. Parameter gradient accumulation validates all candidates before committing. Failure leaves parameters and their existing gradients unchanged.
- `zero_grad` initializes explicit zero buffers. A successfully committed SGD step clears gradients; failed steps preserve them. Disconnected parameters with zero gradients remain unchanged.
- Drop the tape to release all graph and gradient storage. No reused IDs or stale handles.

## Public API expectations

Provide `ParameterSet::insert/value/zero_grad/accumulate_from`, `Tape::input/parameter` plus one method per supported operation, `Tape::backward/gradient`, `ResourceLimits`, `Sgd::new/step`, and `AutogradError`. All mutating/allocation methods return `Result`. Keep IDs opaque. Add only justified tensor support: fallible clone and checked retained-capacity accounting; do not change numerical formulas or public behavior from TASK 002.

## Resource budgets

Enforce existing per-tensor 64 MiB and proposed aggregate active-session 128 MiB retained tensor-buffer plus metadata ceiling; maximum 100,000 values, 100,000 ops, 100,000,000 estimated scalar work units per pass, and 10,000 learning steps. The session total includes parameter values and gradients, active tape snapshots/outputs/adjoints, metadata, scratch, and transactional SGD staging. SGD has no persistent optimizer state. All arithmetic is checked. Charge leaf snapshots, forward results, gradients, op/value metadata, scratch, and optimizer staging. Preflight shape/work/estimated bytes, use fallible reserve, reconcile actual retained capacities, and discard local candidates on rejection. State clearly that allocator metadata, OS overhead, unrelated allocations, and RSS are excluded; this is not an OOM guarantee.

## Optimizer requirements

SGD only: finite `lr > 0`; update `p_new=p-lr*g`; validate complete inventory/shapes/finiteness/work and budget; build and validate every staged tensor; commit all parameter replacements atomically for expected errors; clear gradients only after commit. No momentum, clipping, Adam, scheduler, or partial commit. Empty parameter set is a no-op.

## Test matrix

Implement all cases in [the validation and resource plan](TASK-003-VALIDATION-AND-RESOURCE-PLAN.md): hand and analytic derivatives; chain rule; repeated/shared dependencies; vector/scalar and rectangular matmul gradients; MSE including invalid/empty/non-finite cases; accumulation and clearing; disconnected values; repeated backward; invalid IDs/shapes; transactional SGD and invalid hyperparameters; finite differences with three incorrect-gradient negative controls; tiny synthetic resource limits; deterministic debug/release results; independent Python reference; unchanged 295-case TASK 002 oracle and all existing checks.

## Learning proof

Use fixed data `x=[-2,-1,0,1,2]`, `y=2x+3`, initial `(w,b)=(0,0)`, mean MSE, full-batch SGD at `lr=0.05`, 100 steps. Record all losses and final values. Reference recurrence: `w_(t+1)-2=0.8(w_t-2)`, `b_(t+1)-3=0.9(b_t-3)`. Success: finite monotonically decreasing loss, final loss `<1e-8` and `< initial*1e-6`, `|w-2|<=1e-5`, `|b-3|<=1e-4`, each step agrees with independent standard-library Python reference under calibrated f32 tolerances. No experiment is implemented in this planning milestone.

## CI and acceptance criteria

Preserve TASK 002 validation. Require pinned Rust, locked offline Cargo, formatting, Clippy warnings denied, docs, debug/release tests, all current oracle/privacy/artifact checks, plus deterministic analytical/finite-difference/optimizer/budget/learning tests and Python reference. Keep one bounded job, read-only permissions, exact-head check, pinned actions, <=15-minute timeout, two build jobs, and no downloads.

TASK 003-B is ready to propose for owner review when:

1. All specified derivatives and shapes are implemented and independently validated.
2. Shared and repeated graph edges accumulate correctly; second backward is rejected.
3. Resource ceilings and transactional failure behavior are tested with small synthetic limits.
4. SGD passes exact and invalid-input tests without partial updates.
5. Learning experiment passes on exact CI head in debug and release and matches independent reference.
6. Existing TASK 002 results and behavior remain unchanged.
7. Safe Rust, zero unjustified runtime dependencies, bounded CI, and clean privacy checks hold.
8. Documentation names actual limits and makes no RSS or language-model-training claim.
9. Owner reviews and authorizes implementation before Session B begins.

## Explicit non-goals

No Transformer, attention, tokenizer, language-model corpus or training, checkpoint format, inference, GPU, distributed execution, SIMD, quantization, FFI, unsafe Rust, higher-order derivatives, serving runtime, custom allocator, generalized views/broadcasting, or Adam.

## Risks and unresolved decisions

- Confirm aggregate default caps (proposed 128 MiB active-session retention, 100k values/ops, 100M scalar work) against target use before coding; tests can lower but not raise project ceilings.
- Decide whether the minimal tensor capacity-introspection extension should be public or exposed through an internal workspace interface; it must be safe and not claim exact RSS.
- Calibrate finite-difference tolerance on f32 operations and preserve a higher-precision independent oracle.
- Confirm exact linear experiment thresholds after running the independently implemented bounded reference; do not loosen thresholds without recorded numerical evidence.
- Review behavior for zero-dimensional matmul gradients and whether empty dot should be accepted as TASK 002 allows.

No code, benchmark, model, or training result is authorized by this handoff.
