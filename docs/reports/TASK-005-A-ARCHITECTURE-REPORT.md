# TASK 005-A — Architecture Research and Feasibility Report

**State:** planning evidence on the exact TASK 004 accepted baseline. No production Transformer or training pipeline is implemented.

## A–B. Summary and verified baseline

The selected reference is a decoder-only pre-RMSNorm Transformer using tied token/output embeddings, RoPE, MHA, RMSNorm, SwiGLU, no biases, explicit stable causal attention, and f32 reference values with f64 reductions. This is chosen for precise reference semantics, parameter sharing, standard causal behavior, modest KV state and a direct CPU implementation path. It is a design selection, not a claim of greater intelligence or quality.

Verified local `main` was clean at exact accepted SHA `334beb088f116240cdd0346a70d9527c26f08e51`, matching user-specified main. Remote is the canonical GitHub repository. Roadmap is `docs/ROADMAP.md`. Accepted TASK 004 final acceptance and TASK 002/003 reports, architecture specs, compute and dataset policy, and efficiency-first policy were reviewed. The tokenizer contract is byte vocabulary 260, BPE vocabulary 292 in accepted fixture, current BPE cap 772; IDs are u32, PAD 258 and DOC_SEP 259. Dataset batches use exact next-token shift, optional BOS/EOS, per-document boundaries, and explicit validity/loss masks. No attention mask exists. No packing is allowed here.

## C–D. Native engine and candidate research

Tensor is contiguous row-major f32, rank <=32, 64 MiB per tensor for payload and reported Vec capacity, with checked shape/allocation. Implemented ops: exact-shape elementwise add/subtract/multiply, scalar scale, rank-1 dot, rank-2 matmul, transpose, sum, checked indexing. Dot/matmul accumulate sequentially in f64 then cast to f32. Autograd is first-order, finite-only, explicit tape, snapshot parameters, scalar-root backward, gradient accumulation and transactional SGD. Parameter IDs/revisions are stable; repeated snapshots merge. Limits: aggregate 128 MiB; 100,000 values, 100,000 operations and 100,000,000 work units per phase; 10,000 experiment steps. Missing: gather/scatter, rank-3 batched matmul, reshape/permutation, RMSNorm, softmax, causal mask, RoPE, activations, cross-entropy, attention, tied embedding path, and model-level batch validation.

| Candidate | Pros | Costs/risks | Decision |
|---|---|---|---|
| Learned absolute positions | Simple lookup/add and gradient | Adds `Tmax*D`; fixed learned range; position gradient scatter | Deferred |
| RoPE | No learned position weights; relative phase in Q/K; defined formula | Trig, pair layout/sign and length behavior need exact tests | Selected; fixed theta and no extrapolation claim |
| MHA | One Q/K/V head layout; direct standard oracle; easiest baseline | KV bytes `2NBCD`; all heads store K/V | Selected as easiest semantically complete reference |
| GQA | Reduces K/V projection parameters and KV cache by query/KV head ratio | Head mapping and gradient aggregation add semantics | Deferred until MHA parity and memory evidence |
| MQA | Minimum K/V cache | More shared K/V gradient traffic; possible quality trade-off | Deferred |
| LayerNorm | Centers and scales features | Mean subtraction and variance VJP; extra reduction | Deferred |
| RMSNorm | One square reduction; scale only; simple VJP | No centering; quality is empirical | Selected for simpler, explicit reference—not assumed better |
| Post-norm | Familiar alternative ordering | Different optimization/stability behavior | Deferred |
| GELU/SiLU FFN | Simpler one/two projection structure | May require more width/params for comparable quality | Deferred |
| SwiGLU | Gated nonlinear capacity with explicit differentiable branches | Three matrices and activation memory; more parameters at same F | Selected as testable hypothesis, quality unproven |
| Untied output | Independent output matrix | Additional `V*D` weights/gradient | Deferred |
| Tied output | Saves `V*D`; tests shared gradients; lower model storage | Input/output gradients interact; target distribution may favor untied | Selected for lower storage and directly verified sharing |
| Explicit full attention | Simple CPU oracle and exact semantics | `O(BHT²)` activation memory and quadratic compute | Selected reference only |
| Tiled/FlashAttention-compatible | Reduces materialized attention state and often memory traffic | Specialized kernel and complex backward; backend-specific evidence needed | Deferred; must preserve semantics |

Primary research references: Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762); Su et al. [RoFormer](https://arxiv.org/abs/2104.09864); Zhang & Sennrich [RMSNorm](https://arxiv.org/abs/1910.07467); Shazeer [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202); Ainslie et al. [GQA](https://arxiv.org/abs/2305.13245); Dao et al. [FlashAttention](https://arxiv.org/abs/2205.14135). Papers motivate candidates and complexity, not guaranteed benefit for XYNOOS's vocabulary, data or scale.

## E–I. Selected model and contracts

Complete input/output shapes, every forward equation, VJP, initialization and rejection contract are in [Transformer specification](../design/TASK-005-TRANSFORMER-SPEC.md). Batch/sample isolation, right padding, PAD/loss handling, causal masks, fully masked query behavior and no-cross-document semantics are explicit there. Loss is stable masked mean next-token cross-entropy; zero supervised positions are a typed error. Target IDs are validated even when their loss mask is false. PAD is excluded from loss. No `-inf` is injected. Parameters/activations are f32, f64 reduction intermediates; failures on nonfinite values abort transactionally.

Tokenizer integration consumes actual `Batch` IDs and masks, checks exact model/tokenizer vocabulary equality, all shape/range/mask invariants, and shifted input/target semantics. Because existing batches do not define an attention mask, model computes key-valid + causal attention internally. The right-padding prefix constraint prevents holes. It does not construct boundaries across documents or infer them from DOC_SEP.

## G. Native autograd strategy

Minimal additions: checked reshape/permutation and `[B,M,K]x[B,K,N]` matmul; selective tape ops for embedding scatter-add, RoPE, fused masked causal attention, RMSNorm, SiLU/SwiGLU, and masked next-token CE. Keep dense projections, residual math and simple elementwise ops compositional. Charge saved state and retained capacities. Full contract and error behavior: [autograd extension design](../design/TASK-005-NATIVE-AUTOGRAD-EXTENSIONS.md). No silent gradient approximations. Existing SGD may verify one tiny update, or TASK 005-B can stop at verified gradient production. AdamW deferred to TASK 006.

## H–J. Initialization and exact parameter counts

SplitMix64 + open-interval top-53 uniform and Box-Muller normal stream; seed/config/order frozen. Embedding normal stddev .02; matrices Xavier uniform, residual output matrices scaled by `1/sqrt(2N)`; RMS gains 1; no biases. Reproducibility is bitwise on same backend/order, tolerance-based across future hardware.

For vocab V, width D, N blocks, FF width F: `P=V*D + D + N*(4D²+3DF+2D)`. Includes tied E once, final RMS gain, Q/K/V/O, gate/value/down, two per-block RMS gains. No RoPE parameters.

| Fixture | Configuration | Exact parameters | Role |
|---|---|---:|---|
| Tiny | V260,D16,N1,H2,F48,Tmax8 | 7,536 | Math checks only |
| Small | V292,D32,N4,H4,F96,Tmax64 | 62,880 | Integration correctness |
| First-training candidate | V292,D96,N8,H4,F256,Tmax128 | 914,400 | TASK 006 research candidate only |

## K–L. Memory and work feasibility

Detailed conservative accounting and formulae are in [resource/scaling plan](../design/TASK-005-RESOURCE-AND-SCALING-PLAN.md). At first-candidate f32, parameters are 3.49 MiB, gradients 3.49 MiB, transactional SGD candidate values another 3.49 MiB. RMS forward intermediates at B1,T32 are estimated ~2.4 MiB, with similar adjoints plus saved states/tables/staging; plausible peak is tens of MiB but must be measured/accounted by actual implementation. The 128 MiB ledger may fit this fixture but has not been demonstrated. One tensor can still exceed 64 MiB: logits at large B/T/V require preflight. Existing work ceiling may fail for N8,D96,F256,B1,T32: approximately 60M dense forward arithmetic and about 180M forward+backward before elementwise work; arithmetic FLOPs do not equal `ResourceLimits` units. Session B must prove exact op-accounted work or submit narrowly bounded model-specific policy change. Do not weaken old budgets or infer training feasibility.

## N–O. Future acceleration and CPU inference

Keep this algorithm hardware-agnostic with an independently executable Rust CPU path. Future boundary dispatches matmul, normalization, attention and pointwise kernels to optimized CPU then optional CUDA runtime/driver, cuBLAS or custom native kernels; persistent device buffers, mixed precision and fused/tiled attention all need independent parity against the CPU oracle at precision-specific tolerances. Dense projections and attention dominate compute. No GPU dependency, kernel, runtime or Colab execution is part of this task.

Weights for the first-training candidate occupy 3.49 MiB at f32. MHA KV cache is `2*N*B*C*D*bytes_per_element`; at N8,B1,C128,D96,f32 it is 768 KiB (excluding overhead). MHA uses more cache than GQA/MQA. Future INT8/INT4 requires scale/zero-point and accumulation contract; direct quantized compute may trade accuracy/latency, and no speed claim is made. CPU scalar fallback, runtime feature detection, bounded threads, mmap loading and activation reuse are future runtime concerns. No physical device measurements or low-resource performance claims exist.

## P–Q. Independent validation

[Validation plan](../design/TASK-005-VALIDATION-PLAN.md) specifies Python stdlib independent oracle, binary32 bit fixture serialization, tolerances, finite differences across all parameter families, a hand-derived 2-token attention example, stable loss, negative controls, invariance, resource and error cleanup. These are proposed, not run in TASK 005-A.

Negative controls include wrong causal mask/future leakage, scale, QKV reshape/head order, RoPE sign/frequency, normalization axis, activation derivative, omitted residual, shifted target error, PAD in loss, missing tied E path, incorrect CE gradient, unstable extreme logits and nonfinite/budget failures. They count only after execution demonstrates detection.

## R–S. Session B and deferred scope

Session B handoff: [TASK-005-IMPLEMENTATION-HANDOFF.md](../design/TASK-005-IMPLEMENTATION-HANDOFF.md). Build independent oracle first; implement tiny then small; validate exact VJPs/finite differences and resource use; retain all 90 accepted Rust checks/profile and earlier fixtures. No production training, AdamW, GPU, external data, packing, inference, quantization, checkpoints or model publication. Edits are subject to milestone review and must not treat this design as automatic production authorization.

Scaling direction remains Experimental 100K–50M, Nano 125M, Micro 350M, Edge 1B, Lite 3B, Core 8B, Advanced 14B as conditional research ambition. f32 14B weights alone are ~52.2 GiB; current tensor/autograd resource contracts do not support such models. Larger editions require redesigned parameter/gradient storage, compute, memory, checkpoint and inference contracts. The architecture equations scale coherently, but the current implementation engine does not.

## T. Planning branch and evidence status

Branch `research/task-005-a-transformer-architecture` was created from accepted SHA `334beb088f116240cdd0346a70d9527c26f08e51`. Draft PR [#12](https://github.com/samuelindra123/xynoos-cyber-model/pull/12) targets `main` and remains unmerged. Initial planning head `6af97920f0610156916c1e1e8d97e36e90ae1998` passed Foundation CI run [45573318266](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45573318266), job `139065024111`. The final specification head and its exact-head Foundation CI result are recorded in the PR check and the TASK 005-A completion report; the initial run passed exact checkout, hygiene/privacy, locked metadata, formatting, Clippy, all three prior Python oracles, debug/release tests and doctests, deterministic comparisons and Rust docs.

## U. Readiness verdict

**DESIGN CONDITIONAL — SPECIFIC ISSUES REMAIN.** Mathematical and interface choices are fixed sufficiently for implementation review. The precise operation-to-work-unit mapping, measured autograd peak/capacity accounting, and executed independent/negative-control validation remain outstanding for Session B before architecture correctness can be accepted. This report does not authorize production implementation or establish model intelligence.
