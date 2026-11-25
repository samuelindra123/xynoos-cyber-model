# Compute Policy

The project is CPU-first, hardware-agnostic, and designed for safe resource use. Documentation must not publish maintainers' personal hardware specifications, machine identifiers, usernames, or local absolute paths. Use generic terms such as commodity CPUs, CPU-only environments, low-resource computing environments, and memory-constrained devices.

## Local development

Keep local workloads lightweight: coding, documentation, formatting, static analysis, small tests, small numerical experiments, minimal tokenizer and inference checks, smoke tests, and inspection of remote results. Do not automatically run large training, download large datasets, saturate CPUs for extended periods, allocate unbounded memory, or create large binaries.

## Tensor library budgets

Allocating calls enforce a 64 MiB ceiling per tensor on both logical f32 payload and the retained f32 buffer capacity reported by `Vec::capacity()`. Callers can supply smaller budgets. An adopted buffer must fit the limit, and newly reserved buffers are checked before values are written. Metadata, allocator overhead, capacity invisible to `Vec`, all other live allocations, process RSS and OS overcommit are outside this per-tensor contract. Callers bound matrix computation separately. Keep local tests small and use the bounded routine CI for full profile validation. [Tensor resource contract](../crates/xynoos-tensor/README.md)

## Default research environment

GitHub Actions remains the default for CI and build validation, mathematical verification, independent oracles, regression testing, bounded CPU experiments, and initial small-scale model training when suitable for its resource limits. Keep jobs bounded, deterministic where appropriate, and transparent about measured resource use.

Do not weaken a meaningful model-quality objective solely to fit GitHub Actions. First measure whether the limiting factor is throughput, duration, peak memory, architecture, dataset coverage, or another constraint. Investigate numerical, architectural, optimizer, and dataset defects independently; moving a workload does not fix those defects.

## Adaptive compute and remote training

Google Colab is the preferred candidate for authorized remote GPU training when measured CPU limitations materially prevent the intended research objective. Decide using measured throughput, estimated training-token count, projected duration, peak memory, architecture constraints, dataset coverage, quality objectives, provider restrictions, and an explicitly authorized compute budget.

The official Google Colab CLI is installed and authenticated in the development environment. Authentication has been verified, but remote GPU provisioning, execution, file transfer, and runtime cleanup have not been verified. Treat those capabilities as untested until a separately authorized, documented exercise establishes them.

Use the official Colab CLI for runtime management and remote execution where supported. Do not provision a runtime, consume compute units, change subscription settings, purchase resources, or use billable infrastructure without owner authorization. Never publish private account balances or authentication information. Keep private credentials out of the repository.

## GitHub Actions and future experiments

Use bounded workflows with job timeouts, concurrency controls, redundant-run cancellation, deterministic seeds where suitable, constrained inputs, and explicit artifact retention. Heavy training and benchmark workloads must be manually dispatched unless they become demonstrably lightweight and necessary for normal CI. Report resource usage where practical. Follow GitHub Actions usage policies, repository-purpose limits, resource limits, and available billing allowances. Do not configure paid infrastructure without explicit authorization.

If future work exceeds these constraints, migrate only the affected workload to suitable, authorized research compute and document the evidence and limits. GitHub Actions must not be treated as unlimited general-purpose compute.
