# TASK 006 — Native Training Engine Specification

**Status:** TASK 006-A research design. This is not a training authorization or a production implementation.

## 1. Fixed target and boundary

The target remains XYNOOS Cyber 0.1, model ID xynoos-cyber-0.1: 914,400 parameters in 74 tensors, eight decoder blocks, width 96, four heads, FFN width 256, vocabulary 292, context capacity 128, tied token/output embeddings, pre-RMSNorm, RoPE, causal attention, and SwiGLU. It uses the existing native Rust reference and random initialization. The accepted backward case is exactly B=1, T=32. No model, tokenizer, parameter count, or version change is proposed.

TASK 005 accepts bounded reference execution, not repeated training. Its exact-candidate backward permit is not a training permit. TASK 006-B must introduce and review an explicit bounded training policy before repeated optimizer steps are enabled.

## 2. Training-run contract

A run is a finite state machine. Every transition is checked before the next state is published:

1. **Validate configuration.** Reject unknown model/tokenizer versions, missing finite resource limits, invalid schedule or optimizer values, dirty or unavailable source identity, unauthorized data, invalid splits, incompatible shapes, and bounds that exceed project caps.
2. **Initialize or resume.** Initialization uses the accepted seeded initializer. Resume validates the complete checkpoint before exposing a new trainer. Do not reinitialize and then silently continue from a different seed.
3. **Acquire a batch.** The data stream verifies the immutable dataset identity and split. Compute checked consumed-token and supervised-token additions before forwarding the batch.
4. **Forward and loss.** Validate every ID and mask under the accepted batch contract. Compute masked token cross-entropy using the TASK 005 stable reference arithmetic. Training skips an all-unsupervised batch before evaluation; the public model evaluation continues to reject an empty loss mask as it does today.
5. **Backward and transfer.** Each microbatch owns a fresh, bounded evaluation. Reverse failure seals and consumes that evaluation. A successful transfer creates one owner- and revision-bound gradient batch.
6. **Accumulate.** Weight each microbatch mean gradient by its supervised-token count in deterministic microbatch, parameter-inventory, and row-major order. Keep the accumulation private until the whole optimizer window is valid.
7. **Normalize and clip.** Divide the accumulated sum by the total supervised-target count, narrow once per gradient coordinate to FP32, then compute and apply global-norm clipping. Empty windows never produce an optimizer update.
8. **Update transaction.** AdamW stages all candidate moments and all candidate parameter values. On any error it publishes none of them. A successful update increments the model revision and optimizer step exactly once and clears accumulated gradients.
9. **Metrics, checkpoint, validation.** Publish a metric record only for a committed update. Write checkpoints at configured completed-update boundaries. Run fixed validation after the due checkpoint so a validation failure cannot destroy the last committed training state.
10. **Terminate.** Stop at an explicit finite boundary, write a final safe checkpoint when possible, report one stable stop reason, and destroy tape, batches, staging, readers, and runtime-owned resources.

No loop may run without positive finite limits for optimizer steps, consumed token occurrences, supervised target tokens, cumulative work, owned memory, and monotonic wall time. The effective maximum optimizer-step count is at most the existing 10,000 experiment-step cap. A later cap may be lower. Token and work limits are checked before accepting a batch or microbatch; the run does not silently overshoot or partially truncate it.

## 3. Gradient lifecycle and clipping

The cross-entropy kernel returns a mean over that microbatch's supervised targets. Let microbatch j have m_j supervised targets and mean gradient g_j. The optimizer-window objective is the mean over all M = sum(m_j) targets, so the correct gradient is:

    g = (sum_j m_j * g_j) / M

Do not average microbatch means equally. For example, mean gradients [2, 0] over two targets and [0, 3] over one target produce the combined mean [4/3, 1]. Averaging the two microbatch means gives the wrong [1, 1.5].

TASK 006-B should add a weighted accumulator owned by xynoos-autograd, bound to the same parameter owner, inventory, and revision as GradientBatch. Accumulate each FP32 microbatch mean multiplied by its integer target count into an FP64 row-major sum. At the window boundary divide once by total targets and narrow once to FP32. This avoids repeated FP32 rounding and preserves deterministic reduction order. The accumulator consumes every incoming batch on success or failure, like current accumulation. Do not publish partial-window gradients into persistent ParameterSet state.

The clipping input is the token-normalized global gradient, not an individual microbatch mean. Compute squared norm in FP64, in parameter-inventory then tensor-row-major order, using separate multiply and add. If the norm is finite and greater than threshold C, multiply every FP32 gradient by C/norm in FP64 and narrow once. If it is equal to or below C, leave its bits unchanged. Zero norm remains exact zero. A non-finite gradient or norm rejects the window before optimizer mutation. An extreme but finite gradient may be clipped if the FP64 norm is finite; overflow during normalization or clipping is a numerical failure.

When clipping is enabled, C must be finite and strictly positive. Disabled clipping is an explicit configuration value, not a zero threshold. The reported norm is pre-clip; also report post-clip norm and whether clipping occurred.

For a small independent illustration, [4/3, 1] has norm 5/3 and clips at C=1 to [0.8, 0.6]. The equal-microbatch negative control [1, 1.5] instead clips to approximately [0.554700, 0.832050]. The latter is incorrect for unequal target counts.

If a stop arrives or a resource/data/numerical error occurs during a window, discard its private accumulator and retain the last committed model and optimizer state. Do not apply a partial window. An all-unsupervised batch is counted as acquired input but contributes zero supervised targets and causes no forward, backward, or update. If an entire run reaches its boundary without a supervised target, stop with no_supervision.

## 4. Learning-rate schedule

The first reference schedule is **linear warmup followed by linear decay**. It needs no trigonometric library, is easy to audit, and is sufficient to establish deterministic scheduling before experiments compare schedule families. Constant learning rate is simpler but omits warmup; a purely linear decay has no warmup; cosine decay is a reasonable later comparison, not required for the first engine acceptance. The original Transformer paper used warmup with inverse-square-root decay, while SGDR studies cosine cycles; neither establishes an optimal schedule for this model or corpus.

Let S be the configured total optimizer updates, W the warmup update count, p the FP32 peak rate, and f the FP32 final rate, with 0 <= W < S, S >= 2, 0 < f <= p, and p > 0. Update index k is one-based and identifies the update about to occur. For W > 0:

    k <= W:  lr(k) = p * k / W
    k > W:   lr(k) = f + (p - f) * (S - k) / (S - W)

For W = 0:

    lr(k) = f + (p - f) * (S - k) / (S - 1)

The final update k=S uses positive f, so every AdamW update has a strictly positive rate. W=S, zero or negative S, S above the configured/project step cap, non-finite rates, f<=0, f>p, or a rate/decay product that violates the AdamW guard is invalid. Evaluate the formula in FP64 from widened FP32 configuration values, in the written order, then narrow the scheduled rate once to FP32. Store S, W, p, f and completed update count in the checkpoint; derive the next rate from that state rather than trusting a cached value.

Do not name a guaranteed learning rate. TASK 006-B/C should first run a bounded logarithmic sweep on the correctness objective, hold data order and seed fixed, record gradient norms, update-to-parameter norm, finite status, and train/validation loss, then select only a rate supported by those observations. See [Attention Is All You Need](https://arxiv.org/abs/1706.03762) and [SGDR](https://arxiv.org/abs/1608.03983) for historical schedule examples.

## 5. Stop conditions and transaction boundaries

Optimizer update is the central transaction. Parameter values, first/second moments, bias-power scalars, optimizer step, and ParameterSet revision commit together only after all staged values pass shape, finite, representability, and resource checks. Before commit, failures leave all those fields bitwise unchanged. Successful commit clears persistent gradients. Existing SGD behavior and tests remain unchanged.

Batch acquisition and gradient accumulation are separate private transactions. A failed evaluation, backward, transfer, weighted accumulation, normalization, or clip publishes no parameter or optimizer change. Failed optimizer staging leaves any already-published persistent gradient inventory available for diagnosis; run termination then drops it. The last checkpoint always describes a completed optimizer transaction, never a half-window.

Count consumed tokens as valid input-token occurrences in a yielded batch accepted for processing; overlapping windows count repeated occurrences. Count supervised tokens as loss-mask ones. The two counters are not interchangeable. A batch that would exceed either remaining cap is not forwarded; the run ends below or exactly at the cap. Validation tokens have separate counters and never count toward training limits.

Stable stop reasons are max_optimizer_steps, max_consumed_tokens, max_supervised_tokens, max_runtime, dataset_exhausted, stop_requested, no_supervision, resource_limit, numerical_failure, gradient_instability, checkpoint_failure, validation_failure, and configuration_failure. At one safe boundary, hard errors take precedence, followed by an accepted stop request, then optimizer-step, supervised-token, consumed-token, runtime, and dataset-exhaustion limits in that order. Diagnostics include a stable stage code and parameter ordinal where applicable, not input text, credentials, or host identifiers.

Every non-finite loss, gradient, norm, optimizer moment, or staged parameter is an immediate numerical failure. A finite but unstable run is detected by predeclared, configuration-bound guards such as a hard maximum pre-clip norm or a consecutive clipped-update limit. Their values must be established from bounded Objective A measurements and recorded in the run identity; do not pick a post hoc threshold after observing a run.

Poll stop requests between bounded microbatches and before optimizer commit. A request before commit discards the private window. A request after commit writes the configured safe checkpoint and exits. The maximum runtime uses a monotonic clock and is checked between bounded operations. The provider job timeout must also leave a separately measured shutdown/checkpoint grace period; wall-time limits cannot make a blocked filesystem write cancellable.

## 6. Candidate training resource policy

Preserve the accepted limits: 64 MiB per tensor payload and retained capacity; 128 MiB aggregate reference ledger; ordinary 100,000,000 work units; exact candidate backward 160,000,000 work units; current value, operation, and insertion limits; and the existing 10,000-step experiment cap. No limit is raised here.

TASK 005's candidate policy applies only to its exact candidate inventory and one B=1,T=32 reference backward. It does not authorize repeating this operation for training. TASK 006-B must define a separate unforgeable CandidateTrainingPolicy bound to model ID, architecture/inventory digest, batch shape, maximum microbatches per update, maximum updates, token caps, cumulative work, memory budget, and runtime. It may select the already accepted 160,000,000 per-pass backward ceiling only for that exact microbatch and only when the new policy is explicitly supplied. It must not alter ResourceLimits::MAX_WORK, grant a generic bypass, or cover other shapes/configurations.

The policy preflights the full run arithmetic with checked operations, charges forward/backward, transfer, weighted accumulation, clipping, optimizer staging, validation, and checkpoint overlap, and enforces cumulative counters. TASK 006-B/C must choose numeric budgets from measured timings and owner approval. Until then, no repeated candidate training is permitted by this specification.

## 7. Memory and repeated-step design

The following separates measured payloads from modeled overlap. Values marked accepted are from the candidate reference; an optimizer-enabled peak has not been measured.

| Component | Bytes | Basis and lifetime |
|---|---:|---|
| Model parameters | 3,657,600 | Exact 914,400 FP32 values; persistent |
| AdamW first moments | 3,657,600 | Exact FP32 payload; persistent after initialization |
| AdamW second moments | 3,657,600 | Exact FP32 payload; persistent after initialization |
| Candidate parameter snapshot | 3,657,600 | Accepted forward reference payload; live with evaluation tape |
| Candidate activations, logits and loss | 2,028,036 | Accepted TASK 005 forward payload |
| Saved attention probabilities | 262,144 | 8 layers * 4 heads * 32 * 32 FP64 cells |
| Copied batch IDs and masks | 320 | Accepted B=1,T=32 input |
| Forward metadata/capacity | 40,864 | Accepted measured net forward charge |
| Reverse additional ledger high-water | 6,379,100 | Difference between accepted forward 9,658,268 and forward/backward 16,037,368; includes reverse overlap net, not a pure payload |
| Weighted accumulator | 7,315,200 | Modeled P FP64 sums; temporary for one optimizer window |
| One transferred FP32 gradient inventory | 3,657,600 | May overlap while adding a microbatch to the accumulator |
| AdamW staged candidates | 10,972,800 | One P FP32 value table plus two P FP32 moment tables; update transaction only |
| Checkpoint numeric content | 10,972,800 | Serialized parameters plus two FP32 moments; streamed, not a RAM staging buffer |
| Checkpoint I/O buffer | 65,536 | Proposed fixed 64 KiB; bounded and reusable |
| Dataset metadata | Up to 67,108,864 | Existing 64 MiB cap; separate from autograd ledger |
| Dataset token/batch buffers | Up to 16,777,216 | Existing 16 MiB cap |
| Document and reader | Up to 1,114,112 | Existing 1 MiB document plus 64 KiB reader |
| Tokenizer/encoding buffer | Up to 16,777,216 | Existing 16 MiB buffer policy; actual BPE retained bytes are data-dependent |

The conservative modeled backward overlap is 16,037,368 accepted bytes + 7,315,200 persistent moment bytes + 7,315,200 FP64 accumulation bytes + 3,657,600 transferred gradient bytes = 34,325,368 bytes (about 32.74 MiB). This intentionally assumes transfer can overlap the measured candidate peak; it is not a prediction of the eventual ledger after optimization.

The independent maximum data-side allowances sum to 101,777,408 bytes (97.06 MiB). Adding the modeled optimizer-enabled peak gives 136,102,776 bytes (about 129.80 MiB), above 128 MiB by approximately 1.80 MiB before process/allocator overhead. This does not change the accepted 128 MiB reference-ledger ceiling: dataset/tokenizer buffers are outside that ledger today. It does show that the independent subsystem maxima cannot all be selected for one training run.

Propose a stricter training-run owned-allocation cap of 128 MiB across the measured autograd ledger, optimizer/accumulator staging, configured dataset/tokenizer buffers, and checkpoint staging. Keep the per-ledger, per-tensor, data and tokenizer caps as upper limits. The trainer must reserve each child budget within this run cap and reject an over-budget combination before allocation. For the first small objective, use a dataset/tokenizer allowance no larger than 64 MiB pending actual measurements; reduce it further if measured backward/optimizer overlap requires it. This is a narrow additional policy for training, not an increase to existing limits.

Every evaluation is created fresh and destroyed after transfer. No tape, activation, adjoint, or probability buffer survives to the next microbatch. Persistent memory per completed update is parameters plus first/second moments and counters. Persistent FP64 gradient accumulation exists only within one optimizer window and is dropped after update or abort. Checkpoint output streams through a fixed bounded buffer. Validation runs after training tape and gradient-window buffers are released. The runtime records ledger peak and process high-water separately when the platform exposes it.

## 8. Acceptance evidence

TASK 006-B/C must keep four claims separate:

1. **Numerical correctness:** all optimizer, accumulation, clipping, finite-value, transaction, and resume fixtures match independent equations.
2. **Feasibility:** bounded end-to-end CPU timing, peak application ledger and process memory, data/validation/checkpoint overhead, and resource limits are measured.
3. **Learning behavior:** model weights and optimizer state change and a predeclared learning objective improves under a held-out split.
4. **Domain capability:** a separate licensed corpus and evaluation demonstrate transferable cybersecurity usefulness. No optimizer test or falling training loss establishes this.

For each experiment record pre-update training loss, final training loss, held-out validation loss, token-level perplexity, pre/post-clip gradient norms, clipped fraction, parameter and optimizer-state changes, finite checks, consumed and supervised tokens, and train/validation overlap checks. Perplexity is exp(mean token negative log likelihood) over supervised targets under the named tokenizer; it is not byte-normalized. Report overflow instead of clamping a non-finite perplexity. Do not invent universal quality thresholds: freeze objective-specific thresholds before the run, and include baseline, tokenization, data split, seed, and negative controls.

Same source, target, tokenizer, dataset, config, seed, and execution profile must reproduce the same reference trace. Uninterrupted and checkpoint-resumed runs must agree bitwise on the same platform/toolchain for parameters, moments, scheduler state, next batch identity, counters, and emitted scalar metrics. Cross-platform or accelerated paths require separately justified tolerances and CPU-reference parity.

## 9. Detached runtime and handoff

The future runtime, not an AI agent, owns a versioned JSONL log with start, progress, validation, checkpoint, stop, and failure record types. Every record carries experiment ID, source SHA, configuration digest, model/dataset/tokenizer identities, completed updates, consumed and supervised token counts, current learning rate, train loss, validation loss when due, pre/post-clip norm, parameter-update norm, finite/instability status, checkpoint path and digest when written, and monotonic elapsed time. It never writes source text, secrets, personal paths, or machine identifiers.

The runtime also owns signal handling, safe-boundary cancellation, step/token/work/memory/time caps, instability detection, checkpoint checksums, a final machine-readable report, artifact finalization, and resource destruction. A checkpoint or metrics write failure is reported as a failure reason without corrupting the previous committed model/checkpoint.

The dispatch handoff contains: experiment ID; source SHA; configuration digest; dataset and tokenizer identity; local run URL or remote job identity; checkpoint location; metrics location; stop instructions; and the explicitly authorized resource budget (step/token/work/memory/time limits and provider authorization reference). The preparing agent validates immutable inputs and budget before dispatch, confirms startup with bounded checks, reports this handoff, then ends active supervision. It does not continuously poll or supervise a detached job.

No detached Colab execution, persistence across CLI disconnection, file transfer, or cleanup behavior is claimed as supported or verified. Any future exercise needs separate owner authorization and bounded verification before relying on it.

## References

- Kingma and Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), ICLR 2015.
- Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://openreview.net/forum?id=Bkg6RiCqY7), ICLR 2019.
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
- Loshchilov and Hutter, [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983), 2016.
- Existing contracts: [TASK 005 numerical/resource amendment](TASK-005-NORMATIVE-NUMERICAL-AND-RESOURCE-AMENDMENT.md), [compute policy](../COMPUTE_POLICY.md), and [implementation handoff](TASK-006-IMPLEMENTATION-HANDOFF.md).
