# TASK 006-B — Native Training Implementation Handoff

**Precondition:** owner reviews the TASK 006-A design and explicitly authorizes implementation. This handoff does not authorize training, external data acquisition, Colab, or GPU use.

## 1. Crate ownership and dependency direction

Keep the current crates and contracts:

- xynoos-tensor continues to own bounded FP32 tensors and shape/numeric limits.
- xynoos-autograd continues to own ParameterSet, parameter identities/revisions, GradientBatch, the shared reference ledger, deterministic backward, and existing Sgd.
- xynoos-transformer continues to own the accepted architecture, parameter inventory/order, initialization, forward/loss, and scoped candidate evaluation.
- xynoos-tokenizer remains unchanged for this milestone.
- xynoos-data continues to own manifests, provenance, immutable split validation, and bounded batch streaming.

Add one crate, xynoos-training, for run orchestration, schedule, metrics, resource/run budget coordination, checkpoint codec, run identity, stop control, and detached runtime. It depends on xynoos-transformer, xynoos-data, and xynoos-autograd. Those crates must not depend on xynoos-training. No separate optimizer crate is needed.

Put AdamW and token-weighted gradient accumulation in xynoos-autograd, not the training crate. ParameterSet already owns optimizer-compatible parameter and gradient state, revision identity, work checks, resource ledger, and transactional SGD. The AdamW transaction must commit parameter values and moment state together under that owner. Keep current Sgd API, behavior, fixtures, and errors unchanged.

## 2. Proposed public interfaces

The following describes ownership and method shape, not finalized Rust syntax:

- xynoos-autograd: AdamWConfig, ParameterGroupPolicy, AdamW, AdamWStateView, and WeightedGradientAccumulator.
- AdamW::new binds empty moments to a complete ParameterSet inventory and current owner/revision.
- WeightedGradientAccumulator::add consumes a GradientBatch plus its supervised-target count; finish consumes the accumulator and returns one token-normalized GradientBatch only when total count is positive.
- AdamW::step consumes the complete persistent gradient inventory at an explicit finite learning rate and atomically commits parameters, moments, step, bias powers, and revision.
- Immutable ParameterSet and Transformer inventory/value iteration is the only checkpoint read surface. Restore creates a private model from validated values and binds restored optimizer state after inventory verification.
- xynoos-training: TrainingConfig, TrainingLimits, LinearWarmupDecay, ExperimentIdentity, TrainingMetrics, StopReason, Trainer, CheckpointCodec, and detached-runtime report/handoff records.
- Trainer::run_until_boundary accepts a bounded batch source, monotonic clock, and stop signal. It returns a final report and does not keep itself alive to supervise remote work.
- CheckpointCodec::write_atomic streams a committed trainer state to an artifact root; load constructs a new private trainer or returns a typed error without mutating the caller's trainer.

Do not expose mutable raw optimizer vectors or a general ParameterSet mutation bypass. Add named, checked APIs for candidate restore and optimizer attachment. Candidate-only backward authorization remains an unforgeable evaluation policy; the future training policy is separate and includes a whole-run budget.

## 3. Internal module responsibilities

Within xynoos-autograd:

- adamw: hyperparameter/group validation, FP32 moment storage, FP64 equations, transaction staging, owner/revision binding, finite checks.
- weighted_accumulator: supervised-token weighting, FP64 ordered sums, identity/revision checks, resource charges, private window lifecycle.
- parameter_set integration: inventory fingerprint, optimizer-state accounting, atomic commit, existing Sgd regression protection.

Within xynoos-training:

- config: model/data/tokenizer identity, finite run caps, policy validation, checked budget calculations.
- schedule: one-based linear warmup/decay and exact resume derivation.
- engine: finite lifecycle state machine and safe boundary stop handling.
- metrics: token-level counters/loss/perplexity, gradient/update norms, finite and clipping statistics.
- checkpoint: canonical bounded format, checksums, compatibility checks, atomic write and staged load.
- runtime: structured logs, cancellation, max-runtime enforcement, failure report, artifact finalization, and handoff JSON.

Data and tokenizer contracts should be extended only where needed to expose stable identity/cursor metrics. Any such additions require compatibility tests and no weakening of validation.

## 4. Ordered implementation work

1. Freeze public error types and state diagrams; add types without changing current behavior.
2. Add AdamW scalar/reference unit behavior and parameter-group inventory mapping. Run the Python oracle fixture before wiring updates to Transformer.
3. Add resource-ledger accounting for moments, bias state, staging arrays, weighted accumulator, and all detached overlap. Preserve the current caps.
4. Add all-or-nothing AdamW commit and fault injection at each allocation/staging boundary. Prove Sgd and TASK 001–005 behavior is unchanged.
5. Add weighted token-gradient accumulator with unequal-count, zero-count, batch-owner, stale-revision, order, overflow, and discard tests.
6. Add schedule implementation and boundary/resume trajectory tests; reject invalid configs before model allocation.
7. Add a separate candidate training policy bound to exact model/inventory/shape and finite microbatch/update/token/work/memory/runtime caps. Do not reuse the candidate evaluation token as authorization.
8. Add the training state machine and small project-authored Objective A fixtures. Verify genuine Transformer parameter changes and loss behavior without changing the architecture.
9. Implement checkpoint v1 writer/loader with corruption, size, duplicate-field, path, atomic-write, and failed-load coverage.
10. Add uninterrupted-versus-resumed integration and deterministic reports.
11. Add runtime-owned JSONL progress, cancellation, final report, clean destruction, and a bounded startup check path.
12. Measure complete CPU steps, memory, data, validation, checkpoint, and shutdown overhead. Only then propose experiment budgets/environment.

## 5. Required tests and acceptance gates

### Unit and independent oracle

- Rust AdamW scalar fixtures match every frozen Python result and boundary exactly on the reference build.
- First and multi-step moments, positive/negative/zero gradients, both beta endpoints that are valid, epsilon outside sqrt, decoupled decay, rank-one exclusion, shared embedding single decay, and one global step.
- Negative controls reject coupled L2, missing bias correction, epsilon inside sqrt, reversed accumulation weighting, and applying decay twice to tied E.
- Invalid learning rate/betas/epsilon/decay, rate-decay guard, wrong owner, stale revision, missing gradient, shape mismatch, non-finite inputs, moment/candidate overflow, and u64 counter exhaustion.
- Fail each staged allocation and final precommit check. Assert parameter, moment, revision, step, bias-power and gradient bits are unchanged and temporary ledger charges return.
- Weighted accumulation covers unequal supervised counts, masked/padded targets, empty microbatches, all-empty windows, deterministic order, zero and extreme gradients, normalization before clip, exact/below/above clip threshold, non-finite rejection, and abort cleanup.

### Resource and repeated-step

- Keep per-tensor 64 MiB, aggregate reference 128 MiB, ordinary 100M, candidate 160M, and value/operation limits unchanged.
- Prove every allocation is preflighted/charged, retained capacity is included, staged candidates overlap correctly, and repeated evaluations return to the expected persistent memory after drop.
- Prove candidate training authorization rejects wrong model, inventory, shape, missing explicit limits, oversized cumulative work/tokens/steps/runtime, and a lower caller cap.
- Measure application ledger and process high-water separately; include dataset, tokenizer, validation, checkpoint, and temporary overlap.

### Integration, checkpoint, science

- Objective A uses project-authored deterministic content, source/license provenance, template/source-held-out split, exact/near-duplicate checks, and train-only tokenizer fit.
- Record pre-update/train/held-out losses, per-token perplexity, gradient norms, clipping fraction, parameter/moment changes, all finite checks, and exact token counts.
- Demonstrate at least one nonzero genuine Transformer update and compare a held-out learning objective against frozen baseline/acceptance thresholds set before the run.
- Checkpoint parsing is bounded and rejects all malformed versions, identities, fields, lengths, checksums, shapes and non-finite states.
- Failed checkpoint write/load preserves the prior artifact/live trainer. Uninterrupted and resumed traces agree bitwise under the same reference profile.
- Detached runtime completes under configured step/token/work/memory/time limits, emits parseable structured logs and one final report, responds to cancellation at safe boundaries, and destroys resources.
- Existing Foundation CI passes on the exact PR head. Do not merge until owner approval.

## 6. Decisions reserved for owner review or TASK 006-B/C evidence

- Approve the architecture and numerical conventions in the TASK 006-A specs.
- Choose the Objective A generator and its predeclared held-out metric thresholds.
- Select Objective B sources only after source-specific permission and redistribution review.
- Choose numeric learning-rate sweep, warmup, final rate, weight decay, clipping threshold, microbatch count, token caps, cumulative work cap, memory cap, runtime cap, and validation/checkpoint cadence from bounded measurements.
- Decide whether and when authorized external compute is justified. Colab provisioning, paid resources, and any GPU work remain separately authorized.
- Do not expand model vocabulary/context or issue a new model ID without a separate architecture decision.
