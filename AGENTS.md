# Repository instructions

All agents must use the canonical repository working directory. Do not create additional permanent Git worktrees or repository clones. Session A and Session B operate sequentially using separate Git branches. Preserve all existing changes before switching branches.

- Maintain one permanent working tree. Use branches in the same checkout to isolate planning, implementation, remediation, and independent verification.
- Check `git status` before switching branches. Switch only after the working tree is clean or its changes have been safely preserved.
- Coordinate through branch commits, pull requests, and documented handoffs. Finish one session's modifications before the next session begins modifying the workspace.
- Keep local builds and tests small. Remove regenerable build output when it is no longer needed, and use bounded GitHub Actions workflows for heavier validation.
- Use repository-relative paths in tracked material. Keep personal paths, machine details, and credentials out of the repository.

## Compute and model engineering direction

- Use bounded GitHub Actions by default for CI, builds, mathematical oracles, regression checks, CPU experiments, and initial small-scale training when suitable.
- Base compute decisions on measured throughput, required training tokens, duration, peak memory, architecture constraints, dataset coverage, quality objectives, provider restrictions, and authorized budgets.
- When measured CPU limits materially block a research objective, consider authorized Google Colab GPU resources with the official CLI where supported. Authentication alone does not prove runtime, GPU, transfer, or cleanup capability. Do not start a runtime, consume units, or buy resources without owner authorization.
- Keep the architecture, parameters, tokenizer, tensor engine, autograd, optimizer, training pipeline, and inference runtime native to this project. Initialize language-model weights randomly; do not substitute external pretrained models or external ML frameworks.
- Treat CUDA interfaces, cuBLAS, native GPU kernels, CPU SIMD, and optimized matrix operations as future low-level acceleration options. Preserve an independently verifiable CPU reference and require precision-appropriate numerical parity before accepting acceleration.
- Prefer CPU-first inference and verify compatibility on physical devices before making resource or speed claims. Public documentation must describe generic hardware profiles and must not disclose personal machine specifications.
- Read [the compute policy](docs/COMPUTE_POLICY.md) and [efficiency-first model engineering direction](docs/design/EFFICIENCY-FIRST-MODEL-ENGINEERING.md) before planning training, acceleration, or deployment work.
