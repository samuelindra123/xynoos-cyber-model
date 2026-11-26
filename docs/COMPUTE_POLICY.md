# Compute Policy

The project is CPU-first, hardware-agnostic, and designed for safe resource use. Documentation must not publish maintainers' personal hardware specifications, machine identifiers, usernames, or local absolute paths. Use generic terms such as commodity CPUs, CPU-only environments, low-resource computing environments, and memory-constrained devices.

## Local development

Keep local workloads lightweight: coding, documentation, formatting, static analysis, small tests, small numerical experiments, minimal tokenizer and inference checks, smoke tests, and inspection of remote results. Do not automatically run large training, download large datasets, saturate CPUs for extended periods, allocate unbounded memory, or create large binaries.

## Tensor library budgets

Allocating calls enforce a 64 MiB ceiling per tensor on both logical f32 payload and the retained f32 buffer capacity reported by `Vec::capacity()`. Callers can supply smaller budgets. An adopted buffer must fit the limit, and newly reserved buffers are checked before values are written. Metadata, allocator overhead, capacity invisible to `Vec`, all other live allocations, process RSS and OS overcommit are outside this per-tensor contract. Callers bound matrix computation separately. Keep local tests small and use the bounded routine CI for full profile validation. [Tensor resource contract](../crates/xynoos-tensor/README.md)

## Default research environment

GitHub Actions remains the default for CI and build validation, mathematical verification, independent oracles, regression testing, bounded CPU experiments, and initial small-scale model training when suitable for its resource limits. Keep jobs bounded, deterministic where appropriate, and transparent about measured resource use.

For TASK 006 experiments, compute-budget fit is not scientific or dispatch
approval. First pass the frozen-data, split/leakage, unchanged-weight baseline,
source/config identity, and private-recovery gates. Count training, validation,
checkpoint, and recovery work together. Under the current CPU Actions profile,
the hard software ceilings are 24 updates, 768 consumed/supervised tokens,
8 billion cumulative work units, 64 MiB owned memory, 64 MiB autograd
allowance, 4 MiB data allowance, 120 seconds active runtime, 90 seconds per
attempt, and 16 MiB per checkpoint. These maxima do not imply that a proposed
experiment is meaningful or fits its wall-time budget. Before admission,
measure the exact workload at least three times in the candidate provider's
bounded CPU environment; record throughput, wall time, work, memory, and
checkpoint results, and use the slowest run against the hard limits. GitHub
Actions remains the default. Colab is considered only when measured CPU
constraints prevent the scientific objective; an accelerator requires a
native backend, independent numerical parity, a measured benefit, and an
authorized resource budget. See the [TASK 006-D3 compute readiness report](reports/TASK-006-D3-COMPUTE-READINESS.md)
for the current measurements and remaining gates.

Do not weaken a meaningful model-quality objective solely to fit GitHub Actions. First measure whether the limiting factor is throughput, duration, peak memory, architecture, dataset coverage, or another constraint. Investigate numerical, architectural, optimizer, and dataset defects independently; moving a workload does not fix those defects.

## Adaptive compute and remote training

Google Colab is the preferred candidate for authorized remote GPU training when measured CPU limitations materially prevent the intended research objective. Decide using measured throughput, estimated training-token count, projected duration, peak memory, architecture constraints, dataset coverage, quality objectives, provider restrictions, and an explicitly authorized compute budget.

Use the official Google Colab CLI for runtime management and remote execution where supported. Authentication alone does not establish runtime, GPU, transfer, or cleanup capability. Provisioning, remote GPU execution, file transfer, and runtime cleanup remain unverified until a separately authorized, documented exercise establishes them.

Use the official Colab CLI for runtime management and remote execution where supported. Do not provision a runtime, consume compute units, change subscription settings, purchase resources, or use billable infrastructure without owner authorization. Never publish private account balances or authentication information. Keep private credentials out of the repository.

## Detached long-running training

Authorized long-running training must operate independently of continuous AI-agent supervision. The agent's bounded responsibilities are to:

1. Prepare the experiment and its immutable identity.
2. Validate configuration, data, dependencies, and resource limits before dispatch.
3. Dispatch only the explicitly authorized job.
4. Confirm startup with bounded checks.
5. Report the experiment identity, status, and artifact locations.
6. End active supervision.

The training process must manage its own structured logs, loss and step counters, checkpoint creation, optimizer and resume state, non-finite loss detection, gradient-instability detection, resource limits, maximum steps, maximum runtime, graceful termination, failure reports, final artifact generation, and runtime cleanup. It must not depend on an agent to poll logs, keep a session alive, or decide when to stop.

Do not continuously poll training logs using AI reasoning or use open-ended agent monitoring loops. Deterministic CI checks, runtime-owned telemetry, and bounded startup or completion checks are permitted. An agent may return after the job completes or for a separately authorized investigation. Ending an agent session does not imply that the job or its compute use has stopped.

Before relying on detached Colab execution, verify in a separately authorized exercise that the job survives CLI disconnection where supported, checkpoints persist across the relevant session boundary, and runtime cleanup succeeds. These capabilities remain unverified; this policy does not authorize provisioning Colab or consuming compute units.

## GitHub Actions and future experiments

Use bounded workflows with job timeouts, concurrency controls, redundant-run cancellation, deterministic seeds where suitable, constrained inputs, and explicit artifact retention. Heavy training and benchmark workloads must be manually dispatched unless they become demonstrably lightweight and necessary for normal CI. Report resource usage where practical. Follow GitHub Actions usage policies, repository-purpose limits, resource limits, and available billing allowances. Do not configure paid infrastructure without explicit authorization.

If future work exceeds these constraints, migrate only the affected workload to suitable, authorized research compute and document the evidence and limits. GitHub Actions must not be treated as unlimited general-purpose compute.
