# Compute Policy

The project is CPU-first, hardware-agnostic, and designed for safe resource use. Documentation must not publish maintainers' personal hardware specifications, machine identifiers, usernames, or local absolute paths. Use generic terms such as commodity CPUs, CPU-only environments, low-resource computing environments, and memory-constrained devices.

## Local development

Keep local workloads lightweight: coding, documentation, formatting, static analysis, small tests, small numerical experiments, minimal tokenizer and inference checks, smoke tests, and inspection of remote results. Do not automatically run large training, download large datasets, saturate CPUs for extended periods, allocate unbounded memory, or create large binaries.

## Tensor library budgets

Allocating calls enforce a 64 MiB logical payload ceiling per tensor and allow smaller caller budgets. Shape metadata and allocator capacity add to process memory. Matrix products can require considerable computation within the payload budget. Keep local tests small and use the bounded routine CI for full profile validation. [Tensor resource contract](../crates/xynoos-tensor/README.md)

## GitHub Actions and future experiments

Use bounded workflows with job timeouts, concurrency controls, redundant-run cancellation, deterministic seeds where suitable, constrained inputs, and explicit artifact retention. Heavy training and benchmark workloads must be manually dispatched unless they become demonstrably lightweight and necessary for normal CI. Report resource usage where practical. Follow GitHub Actions usage policies, repository-purpose limits, resource limits, and available billing allowances. Do not configure paid infrastructure without explicit authorization.

If future work exceeds these constraints, document the need for dedicated research compute or external cloud infrastructure and obtain authorization before configuring paid services. GitHub Actions must not be treated as unlimited general-purpose compute.
