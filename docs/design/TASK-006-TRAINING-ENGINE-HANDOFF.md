# TASK 006-A — Training Engine Research Handoff

## Starting point

The first training target is **XYNOOS Cyber 0.1** (`xynoos-cyber-0.1`), the randomly initialized native decoder-only Transformer accepted in TASK 005. Its reference candidate has 914,400 parameters, eight blocks, four attention heads, hidden dimension 96, feed-forward dimension 256, vocabulary size 292, context capacity 128, and an independently verified candidate run at `B=1`, `T=32`.

TASK 005 establishes reference forward and candidate-scoped backward execution. It does not establish a sustained training engine, AdamW, convergence, language-model intelligence, domain knowledge, text-generation quality, production throughput, quantized inference, GPU acceleration, or deployment. No training is authorized by this handoff.

Before specifying a run, TASK 006-A must explicitly evaluate:

- Training-data quality, provenance, licensing, safety, coverage, and suitability for the intended evaluation.
- Whether the current tokenizer and vocabulary are suitable for the selected data and objectives; the synthetic tokenizer corpus is not presumed sufficient for useful cybersecurity intelligence.
- Architecture constraints, context capacity, batch shape, initialization, numerical precision, and any design changes required by evidence.
- Measured CPU throughput, memory use, expected duration, required token coverage, quality objectives, provider constraints, and authorized compute availability.
- Whether the proposed experiment can answer a defined research question and whether an honest quality evaluation is possible.

Do not choose an arbitrary token budget or lower a meaningful quality objective solely to fit a constrained runner. Do not infer a scale-up or model version from parameter count.

## Research questions

1. What data, tokenizer, and evaluation set can support a specific, measurable first learning objective without overstating domain coverage?
2. What optimizer behavior and numerical tolerances are required for the native FP32 reference path?
3. What schedule and gradient policy preserve stability and make experiment comparisons reproducible?
4. How should the stream produce bounded batches without loading the complete corpus into memory?
5. What token, step, runtime, and memory bounds are justified by measured throughput and available authorized compute?
6. What checkpoint schema captures model, optimizer, scheduler, data position, RNG state, source revision, and configuration without ambiguity?
7. What evidence distinguishes true loss improvement from data leakage, memorization, or a broken validation split?
8. Which numerical failures should stop a run, and what state is required to diagnose them safely?
9. If CPU measurements materially block the objective, what optional acceleration path can preserve an independently verifiable CPU reference and numerical parity?
10. Can a future detached runtime survive CLI disconnection, persist checkpoints, produce failure reports, and clean up without an AI-agent monitoring loop?

## Required TASK 006-A design and acceptance gates

TASK 006-A must define and independently review contracts and bounded acceptance evidence for each item below before a sustained training run is proposed:

- Native AdamW: equations, bias correction, decoupled weight decay, parameter groups if needed, finite checks, transactional update behavior, and independent scalar reference cases.
- Learning-rate scheduling: explicit schedule semantics, warmup/decay boundaries, resume behavior, and deterministic tests.
- Gradient clipping: norm definition, accumulation order, non-finite handling, and tests at below/equal/above thresholds.
- Bounded training loop: explicit maximum steps, maximum runtime, work and memory limits, cancellation, cleanup, and deterministic stop reasons.
- Streaming dataset batches: bounded memory, reproducible ordering, train/validation separation, resume cursor, and data-provenance preservation.
- Token-level cross-entropy: agreement with the accepted stable reference equations, masked-target semantics, and finite-difference coverage.
- Gradient accumulation: sample/token weighting, no premature optimizer update, bounded temporary storage, and rollback on failure.
- Training/validation separation: immutable split identity, no fitting on validation data, leakage controls, and evaluation data provenance.
- Deterministic checkpoints: schema/version, architecture and tokenizer identities, weight and optimizer state, scheduler state, RNG state, data cursor, source/config identity, and checksums.
- Optimizer-state persistence and resume correctness: uninterrupted versus save/resume equivalence under defined tolerances, including scheduler and data position.
- Loss convergence verification: a defined, sufficiently meaningful learning task and held-out evidence; a falling training loss alone is insufficient.
- Training-token accounting: exact definition and counts for consumed, supervised, discarded, repeated, and validation tokens as applicable.
- Numerical failure handling: non-finite loss/gradient/parameter detection, gradient instability policy, atomic failure reports, checkpoint safety, and cleanup.
- Detached execution: runtime-owned logs, counters, checkpointing, resume, resource and time bounds, graceful termination, failure reports, artifact finalization, and cleanup; bounded startup verification and no continuous AI-agent supervision.
- CPU reference training: a bounded, reproducible native CPU path with independently checked equations and measured resource usage.
- Measured compute feasibility: throughput, peak memory, expected duration, data coverage, quality target, provider constraints, and explicitly authorized budget before choosing an execution environment.
- Optional future GPU acceleration: separate design/review, preserved CPU oracle, backend-specific failure/resource controls, and precision-appropriate numerical parity. GPU acceleration is optional and not a prerequisite for supported CPU inference profiles.

## Execution boundary

This document is a starting handoff, not TASK 006 implementation or approval to train. Any later long-running job must follow the detached execution policy in [the compute policy](../COMPUTE_POLICY.md) and [repository instructions](../../AGENTS.md). Colab provisioning, compute-unit consumption, and training require separate authorization. Detached Colab execution, checkpoint persistence across CLI disconnection, and cleanup have not been verified.
