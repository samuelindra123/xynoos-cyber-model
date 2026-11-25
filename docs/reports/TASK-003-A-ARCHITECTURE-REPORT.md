# TASK 003-A — Architecture Report

**Status:** Independent planning review; pending project-owner review.
**Verdict:** READY WITH CONDITIONS

## Executive summary

Recommend a small, safe-Rust `xynoos-autograd` crate with an explicitly bounded reverse-mode tape, stable private IDs, closed hand-written backward kernels, deterministic reverse traversal, and a single backward pass per tape. Parameters stay in an owned `ParameterSet`; tapes snapshot values and release all history when dropped. Implement first-order differentiation, MSE, and transactional SGD. Defer Adam and all language-model architecture/training work.

This design composes layer-sized operations while making graph lifetime, storage, errors, and shared-edge accumulation explicit. It is more suitable than either isolated model-specific backward functions or an unbounded reference-linked graph for this small CPU-first engine. TASK 003-A changes no production tensor code.

## Verified baseline

The canonical checkout was clean on `main`; local and fetched `origin/main` both resolved to `c5250543e5a39b087b7daab3d86d69a2af65a703`. The source is an owned contiguous row-major f32 tensor with checked shape/index arithmetic, rank <=32, 64 MiB per-tensor payload and retained numeric-vector capacity checks, fallible reservations, typed errors, elementwise operations, rank-1 dot, rank-2 matmul/transpose, sum, and f64 sequential reduction/matmul accumulation. Scalar shape is rank zero with one element; any zero dimension makes an empty tensor. Current workspace contains only `xynoos-tensor`; there is no gradient, optimizer, or autodiff API.

The acceptance report records 32 tests per profile, 295 deterministic oracle cases/2,670 expected values, six malformed-input negative controls, and zero runtime dependencies. The exact final post-merge CI run is [45543055437](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45543055437), successful on the accepted SHA. No historical numerical values were changed.

## Documentation reconciliation

On this planning branch, the TASK 002 final report closure checklist now marks merge, exact-head CI, post-merge CI, recorded SHA/evidence, and canonical checkout verification complete. It names the final main SHA and links exact-head, merge-commit, and final post-merge CI evidence. The roadmap now states TASK 002 is accepted at the final SHA and TASK 003 requires planning/owner review before implementation. No result or accepted production code was changed.

## Architecture and trade-offs

Selected: explicit tape plus manually implemented kernels, one operation enum, one owned arena per pass. Tape values own snapshots and outputs. Operations refer to earlier checked `ValueId`s. Backward traverses nodes in reverse insertion order, accumulates all edge contributions, and executes once. This avoids references/cycles and makes whole-pass release automatic. Dynamic graphs' arbitrary control flow and retained graphs are unnecessary here; practical systems document graph release/repeated-backward choices as nontrivial user-facing behavior. Standalone explicit backward APIs have low overhead but put composition and repeated/shared contribution correctness on every model author. The minimal tape reuses explicit auditable kernels and supports matrix-layer composition without a general framework.

Primary references: the [JAX autodiff guide](https://docs.jax.dev/en/latest/jacobian-vector-products.html) explains reverse-mode vector-Jacobian products and their relation to Jacobian computation; the [PyTorch autograd mechanics guide](https://docs.pytorch.org/docs/stable/notes/autograd.html) documents saved forward tensors and graph recreation/release behavior. These establish common mathematical and lifecycle patterns, not a requirement to copy framework architecture. The selected resource limits and lifecycle are XYNOOS engineering proposals that require implementation validation.

## Mathematical contracts

The full equations, shapes, empty/non-finite rules, ownership and cost contract are in [the differentiation specification](../design/TASK-003-DIFFERENTIATION-SPEC.md). In brief: add propagates `g` to both inputs; subtract propagates `g` and `-g`; multiply propagates `g*b` and `g*a`; scale propagates `c*g`; sum broadcasts the scalar upstream over the input shape; mean divides that contribution by element count; dot gives `g*b` and `g*a`; transpose transposes upstream; matmul uses `dA=G Bᵀ` and `dB=AᵀG`; MSE over `n>0` values gives `2(p-t)/n`. Matmul formulas are validated for rectangular inputs. All contributions accumulate, including both edges for `x*x` and direct plus shared paths for `x*x+x`.

## Gradient and parameter lifecycle

Inputs are immutable leaves; `ParameterSet` owns trainable values and gradient buffers. A tape snapshots a parameter by fallible copy. Parameter IDs remain stable while the set is structurally fixed. Rust borrows prevent optimizer mutation during a tape's parameter-set borrow. Backward is once per tape; a second attempt errors. Gradients are temporary until backward succeeds, then transferred/accumulated into parameter buffers only after all candidates pass validation. `zero_grad` is explicit; successful SGD clears gradients after atomic commit. Disconnected values return no gradient, and parameters explicitly zeroed remain unchanged. Dropping a tape releases all values/adjoints. Typed errors cover bad IDs, shapes, numerical values, repeated backward, allocation and budget exhaustion.

## Optimizer

Select plain SGD. Require finite positive learning rate. Candidate update is `p-lr*g`. Preflight every parameter and gradient, stage every output, verify all are finite and within resource limits, then replace all parameters and clear gradients. Expected errors leave all parameters and gradients unchanged. Adam is deferred because its moment arrays, bias correction, counter overflow, epsilon, and additional state doubles storage and test surface without helping this learning proof.

## Aggregate resource and compute budget

The tensor-level 64 MiB contract does not bound an autodiff pass. Proposed caps: 128 MiB aggregate simultaneously retained tensor-buffer bytes and metadata per active session; 100,000 values; 100,000 ops; 100,000,000 estimated scalar work units per pass; and 10,000 experiment updates. Charge parameter values and gradients, tape snapshots/forward tensors/adjoints, metadata, scratch, and optimizer staging using checked byte arithmetic; SGD has no persistent state. Count matrix multiply-add as two work units; preflight shape/work before allocation; use fallible reserve and reconcile actual Vec capacities. Drop candidate allocations and preserve state on failure. A minimal tensor capacity query is needed for meaningful retained-capacity accounting.

These are logical/app-owned buffer estimates. They exclude allocator bookkeeping, stack/static memory, OS overhead, other process allocations, and RSS; no exact OOM/process-memory guarantee is claimed. No custom allocator or OS manager is proposed.

## Independent verification and learning experiment

Require hand calculations, polynomial and chain-rule examples, repeated/shared edges, scalar/vector cases, rectangular matrix gradients, gradient accumulation/clearing, invalid shapes/IDs/hyperparameters, non-finite rejection, repeated backward, transactional optimizer failures, synthetic small resource limits, and deterministic debug/release tests. Central finite differences are test-only with scale-aware `h`, f32-calibrated absolute/relative tolerances, and independent analytic/standard-library Python comparisons. Sign, zero, and swapped-matrix negative controls prove checker sensitivity. Finite differences are never the production algorithm.

The bounded learning proof uses fixed five-point data `x=[-2,-1,0,1,2]`, `y=2x+3`, initial `(w,b)=(0,0)`, mean squared error, batch SGD `lr=0.05`, 100 steps. Reference recurrences are `w[t+1]-2=0.8(w[t]-2)` and `b[t+1]-3=0.9(b[t]-3)`. Acceptance targets: finite monotone loss, final loss `<1e-8` and `<initial*1e-6`, `|w-2|<=1e-5`, `|b-3|<=1e-4`, and per-step agreement with an independent standard-library Python implementation. Run full validation in bounded GitHub Actions; no fabricated result is recorded here.

## Proposed crate structure

```text
crates/
  xynoos-tensor/       # accepted; only minimal accounting/copy API extension if required
  xynoos-autograd/     # tape, reverse kernels, MSE, ParameterSet, SGD, errors, tests
 tools/
  gradient_oracle.py   # standard-library deterministic gradient/learning reference
 docs/design/          # spec, resource plan, implementation handoff
 docs/reports/         # this independent planning report
```

Do not create a separate optimizer/loss crate for the initial single loss and SGD. Keep the Python oracle independent and test-only.

## CI, risks, and implementation conditions

Preserve the existing exact-head, pinned Rust, locked/offline Cargo, fmt, Clippy, debug/release, docs, oracle, privacy/artifact checks. Add bounded analytical/finite-difference, accumulation/lifecycle, budget/failure, MSE/SGD, and learning-reference tests in the single existing PR job with read-only permission, <=15 minute timeout, two build jobs, pinned actions and no downloads.

Conditions before implementation authorization: owner confirms aggregate ceilings; implementation calibrates gradient-check tolerances; exact tensor retained-capacity interface is chosen; deterministic Python reference confirms experiment thresholds on the pinned runner; zero-dimension matmul backward edge cases have explicit tests. These are implementation-review conditions, not blockers to owner review of this planning package.

Explicitly out of scope: Transformer, attention, tokenizer, language-model data/training, checkpointing, inference, GPU, distributed, SIMD, quantization, FFI, unsafe Rust, higher-order derivatives, production serving, custom allocator, broadcasting/views, and Adam.

## Session B handoff

The implementation-ready contract and acceptance checklist are in [the handoff](../design/TASK-003-IMPLEMENTATION-HANDOFF.md). Session B should begin only after owner approval of this planning PR and separate implementation authorization. TASK 003-A did not implement gradients, optimizers, a learning experiment, or change numerical engine internals.
