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

## Adaptive AI engineering workflow

- Primary engineering model: GPT-6 Luna. Use medium effort for routine documentation, Git operations, and maintenance; high for ordinary implementation, testing, and debugging; and max for complex milestones, mathematical verification, architecture research, and autonomous end-to-end engineering.
- Escalate genuinely difficult correctness issues to GPT-6.2 Sol, preferably at high effort. Escalation scope includes complex numerical defects, autograd lifecycle violations, optimizer correctness failures, resource-policy bypass, complex GPU backend problems, security-critical architecture, and unresolved mathematical contradictions.
- Do not escalate routine engineering only because work remains. Prepare a focused escalation package with the failing behavior, evidence, relevant code, and specific question; return routine follow-up work to the primary agent.
- Keep independent verification for critical production changes. An implementation agent must not self-certify critical changes.
- Model routing is an owner-controlled operational preference, not a software dependency. Do not put subscription pricing, private account information, usage statistics, or unsupported performance claims in repository documentation.

## Long-running training execution

Authorized long-running training must not depend on continuous AI-agent supervision. The agent prepares the experiment, validates configuration and prerequisites, dispatches the authorized job, performs bounded startup checks, reports the experiment identity and artifact locations, and then ends active supervision.

The training runtime independently owns structured logs, loss and step measurements, checkpoint and resume state, non-finite loss and gradient-instability detection, resource and step limits, maximum runtime, graceful termination, failure reports, final artifact creation, and runtime cleanup. Do not continuously poll training logs with AI reasoning or create open-ended agent monitoring loops. Deterministic CI and runtime monitoring are allowed. An agent may return after completion or for a separately authorized investigation.

Stopping an agent does not stop compute consumption. Before relying on detached Colab execution, verify that work survives CLI disconnection where supported, checkpoints persist, and runtime cleanup works. Do not claim that detached Colab execution has been verified until a separately authorized exercise establishes it. Do not provision Colab or consume compute units without owner authorization.
