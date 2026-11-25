# TASK 005-B — Native Transformer Implementation Handoff

**State:** planning handoff only. TASK 005-A does not authorize production implementation; follow project milestone review and authorization sequence.

## Required reading and fixed decisions

Read the selected architecture in [TASK-005-TRANSFORMER-SPEC.md](TASK-005-TRANSFORMER-SPEC.md), the [native autograd proposal](TASK-005-NATIVE-AUTOGRAD-EXTENSIONS.md), [resource/scaling plan](TASK-005-RESOURCE-AND-SCALING-PLAN.md), [validation plan](TASK-005-VALIDATION-PLAN.md), accepted tokenizer/dataset specifications, and the repository compute/dataset policies. Selected reference: tied E, RoPE, MHA, pre-RMSNorm, SwiGLU, no biases, explicit stable causal attention, prefix-valid PAD masking, independent samples, f32 parameters/values with f64 reductions. Do not silently add a deferred option or alter equations. Record a decision amendment before changing them.

## Scope

1. Define validated model/batch configuration and typed errors, without changing tokenizer or dataset contracts.
2. Add only checked tensor/autograd APIs required by the spec; preserve every existing numerical tolerance and accepted test/oracle fixture. Keep current tensor/autograd crate behavior source-compatible where practical.
3. Build the Python standard-library oracle and frozen deterministic fixtures before accepting Rust output as expected.
4. Implement the tiny configuration first; prove every forward and backward component, finite differences, negative controls, repeated IDs, and tied E gradients independently.
5. Implement small integration configuration and resource/work accounting. Report observed charged peak plus conservative activation/saved-state accounting, not RSS guarantees.
6. Test native optimizer integration only with current transactional SGD or do not update parameters. AdamW belongs to TASK 006 design.
7. Run exact-head bounded Foundation CI, including all preserved checks and 90 accepted Rust checks per profile, and produce a traceability report. No training corpus, model training, GPU dependency, inference runtime, quantization, checkpoints, or weight publication.

## Resource gate

The current hard caps are 64 MiB per tensor, 128 MiB aggregate parameter-set/tape ledger, 100,000 tape values, 100,000 operations, and 100,000,000 work units per phase. Prove explicit fixtures fit before selecting API defaults. If a cap blocks the candidate, make a narrow separately reviewable policy change with exact accounting and preserve old default/test contracts; do not bypass budgets or report FLOPs as work units. Preflight `B*T*V`, `B*H*T*T`, activations, adjoints, gradients, parameter snapshots, and transactional stages using checked arithmetic.

## Required deliverables and evidence

Native Rust design/API docs, tests and oracle fixture schema; executable independent oracle; reproducible tiny/small forward+backward; parameter-count assertion; explicit resource report; validation CI URL/job and exact SHA; changelog/roadmap update only after acceptance. Every negative control must be executed. Final report must state unresolved issues and actual tolerance/resource data. A green CI run alone does not prove architecture quality, training capability, or model intelligence.

## Out of scope

No GPU/CUDA/cuBLAS dependencies or kernels; no external ML framework/pretrained weights; no real model training; no external corpus acquisition; no packing; no AdamW; no FlashAttention implementation; no quantization/inference/checkpoint format; no Colab provisioning or compute spend; no release. Revisit these only in their separately scoped future milestones.
