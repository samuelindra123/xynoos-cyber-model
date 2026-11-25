# TASK 005-C independent verification report

## Scope and verdict

This audit is pinned to accepted main `334beb088f116240cdd0346a70d9527c26f08e51`, planning head `f56af8fccbd4d1248ce3919504211c05ae9a3c42`, and implementation head `97179bcb1af400e56c8b1983590dcad0d4dba946`. PRs #12 and #13 are open; the implementation is the requested head. Foundation run `45580159102`, job `139087889051`, succeeded at that exact SHA. Its hosted steps include revision verification, locked dependencies, lint, oracle regenerations, debug/release tests, resource experiments, and rustdoc.

**Verdict: CONDITIONAL PASS — SPECIFIC REMEDIATION REQUIRED.** The source and hosted record support the Gate A/B reference implementation and the completed Gate C forward pass. Gate C backward is deliberately rejected by the unchanged 100,000,000-unit work cap before reverse allocation. No evidence supports TASK 006 training readiness for the 914,400-parameter candidate.

This is an independent source and evidence review, not an independent re-execution of every hosted experiment. The hosted run is the reproduction evidence for the bounded workloads; this audit did not alter production code, raise limits, rerun the entire suite locally, or execute candidate backward. No independent numerical mutation harness was added. The conclusions below distinguish inspected implementation and exact-head CI from independently derived checks.

## Repository and architecture

Remote refs resolve to the three pinned SHAs. The audit branch starts at the implementation head in the canonical checkout. PR #13 changes the autograd budget/error/kernel/parameter/tape code and adds `xynoos-transformer`, fixtures, oracle, and CI changes. Tensor crate is absent from the changed-file list. Tokenizer and dataset production code is absent from the changed-file list; the workflow retains their fixtures and regression checks.

The model construction order is embedding, each block's attention gain and Q/K/V/O, FFN gain and gate/value/down, then final gain. Forward source follows pre-RMSNorm, Q/K/V projections, RoPE on Q/K, causal attention, O projection and masked residual, then pre-RMSNorm, SwiGLU projections and masked residual; final RMSNorm and tied E projection feed masked shifted cross-entropy. There are no linear biases. Shapes and operation composition match the pinned decoder reference. Reference kernels are separate from legacy kernel arithmetic. Source review found no evidence of an undocumented architectural substitution.

Parameter count follows `VD + D + N(4D²+3DF+2D)`: 7,536; 62,880; and 914,400 for the specified configurations. E is a single registered parameter snapshot used by gather and output projection; the test suite explicitly compares shared-path gradients. Config arithmetic is checked and invalid dimensions/count overflow return typed errors. The full Gate C inventory is source- and CI-reported; this audit did not independently enumerate all runtime parameter IDs.

## Mathematical review and interpretation decisions

The reviewed local contracts are consistent with the standard derivatives: gather scatter-adds repeated rows; dense activation and weight gradients are `dY Wᵀ` and `Xᵀ dY`; RMSNorm uses `gγ/r - x sum(gγx)/(D r³)` and gain reduction `sum(gx/r)`; RoPE VJP uses the transposed rotation; stable softmax VJP is `p*(dp-sum(p dp))`; attention scales QK scores and dQ/dK by `1/sqrt(d)`; SwiGLU differentiates `silu(gate)*value`; residual fan-out sums both paths; and masked mean CE divides by supervised count. The source uses contiguous head slices and interleaved coordinate pairs. These equations and the operation ordering inspected are consistent with the pinned reference.

| Interpretation | Decision | Basis / boundary |
|---|---|---|
| A. PAD target boundary | ACCEPT | Valid final input may have PAD target only at zero loss; invalid rows require PAD on both IDs and zero loss. This is compatible with TASK 004 paired input/target windowing and padded rows. Interior PAD-target rejection enforces shifted stream semantics. |
| B. Stable CE rearrangement | ACCEPT | `log(sum(exp(z-max))) + (max-z_t)` is algebraically equivalent in exact arithmetic and avoids cancellation of the log term when equal large logits make `max + log(sum)` round back to max in FP64. Exponentials are on nonpositive shifts; near-FP32-max equal logits preserve `log(V)`. The implementation rejects nonfinite inputs/results. |
| C. Forward failure lifecycle | ACCEPT | Model owns the tape for evaluation and drops it on error, while individual legacy tape calls retain retry behavior; reverse failures seal. This preserves the legacy distinction and avoids publishing partial model results. |
| D. Initialization details | REQUIRES DOCUMENTED AMENDMENT | Sequence and distributions are deterministic and the exact-head fixture agrees with the independent Python generator. The top-53 midpoint endpoint correction and narrowing the Gaussian before scaling are specific choices not stated by the pinned contract; record them as normative initialization semantics before treating seeded bits as architecture contract. |
| E. Numerical rounding | REQUIRES DOCUMENTED AMENDMENT | The FP32 trig coefficients, FP64 pair transform / probabilities / weighted sum, SiLU narrowing, local VJP narrowing, and FP64 fan-in are explicit and internally coherent, with reference separated from legacy. They are evaluation-boundary choices rather than implied by generic FP32 parameter/activation wording; amend the spec to pin these semantics and parity tolerances. |

## Numerical, data, oracle, and resource evidence

The three fixture digests in the implementation report match the required values as recorded by the exact-head workflow's regeneration step. The Python oracle uses independent standard-library arithmetic and frozen JSON fixtures, but it shares the selected interpretation choices and similar operation decomposition; exact fixture agreement therefore does not establish correctness outside those fixtures. CI reports 2,601 Gate A and 2,921 Gate B logit/loss values, 7,536 Gate A gradients, 76 Gate B gradient coordinates, 6,281 forward and adjoint operation values, and 4,160 tied-E coordinates with zero fixture disagreement. Tolerance remains `2e-5 + 2e-5*abs(expected)`.

Finite-difference evidence is hosted evidence: 23 Gate A and four Gate B coordinates at each of `h=.001` and `.003`, under `2e-3 + 2e-2*abs(fd)`. Reported maximum absolute errors are `1.640543342e-4`, `7.334165275e-5`, `1.380359754e-4`, and `4.443409853e-5`; relative maxima can be large for small derivatives. Coverage is insufficient to independently establish every coordinate in every candidate layer.

The source validates ID bounds, shapes, binary prefix masks, PAD/DOC_SEP rules, shift semantics, and zero-supervision failure; tests cover tokenizer-derived vocab and partial versus padded final batches. Hosted CI preserves the nine accepted tokenizer/data controls and digests. The requested malformed-batch edge matrix was not independently expanded in this audit.

Resource accounting explicitly charges owned tensor payloads/shapes, tape/reference Vec capacities, snapshots, probability tables, batch copies, FP64 pending adjoints, narrowed adjoints, scratch, transfers, and transactional replacements. It excludes allocator internals, stacks, process image and OS overhead; VmHWM is separately process-level. This distinction is appropriate. Work charges are project accounting units, not FLOPs/time. The audited report documents per-kernel formulas and preflight. The exact Gate A/B deterministic totals and Gate C forward totals below are recorded by successful exact-head Foundation measurements; RSS is runner-dependent.

| Gate | Forward work | Backward work | Transfer work | Forward / combined peak | Loss |
|---|---:|---:|---:|---:|---:|
| A, B=2,T=5 | 208,911 | 547,964 | 30,215 | 94,520 / 203,876 B | 5.449348450 |
| B, B=3,T=8 | 3,728,632 | 8,794,371 | 251,762 | 846,572 / 1,516,720 B | hosted report |
| C, B=1,T=32 | 66,894,756 | 152,581,015 requested | not reached | 9,658,268 / reverse not allocated | 5.736882210 |

Gate C initialization is 20,118,280 units. Planner returns typed `WorkLimit` against 100,000,000 before reverse buffers; failure cleanup returns ledger to 3,669,304 parameter-only bytes and publishes neither gradient batch nor parameter mutation. The dense reverse contribution floor independently evaluates to `4*32*(8*(4*96²+3*96*256)+292*96) = 116,834,304`. Attention memory optimization cannot remove that dense work floor. The blocker is software policy under current work accounting, not evidence of a physical CPU/GPU shortfall. No Colab or other external compute is warranted by this result.

## Findings and remaining limits

No confirmed production defect was identified in the source paths reviewed. Confirmed acceptance limitations are: (1) candidate backward unsupported under current policy (MEDIUM, documented policy blocker); (2) initialization and rounding interpretation need normative amendment (LOW, specification ambiguity); (3) candidate-wide backward gradients and peak memory remain unverified (HIGH acceptance gap, not a demonstrated defect). Minimal reproduction for (1) is the exact Gate C config and `backward` call, producing `WorkLimit { requested: 152581015, limit: 100000000 }`; expected current behavior is rejection before reverse allocation. Remediation requires an explicitly authorized scoped policy and full hosted candidate backward verification, as detailed in the proposal.

Hosted CI reports 105 debug and 105 release checks with zero failures/ignored, preserving the 90 accepted checks and 15 Transformer checks; it also records the legacy numerical, gradient, learning trajectory, tokenizer and dataset evidence. CI workflow is pinned, locked, bounded, read-only, and performs debug/release tests, clippy, docs and fixture regeneration. This report does not claim independent reproduction beyond the checked remote records and source-level derivations.

**Candidate status:** not ready for TASK 006 training. Gate A/B reference acceptance and Gate C forward success do not satisfy the missing full-candidate backward, gradient, transfer, accumulation, and reverse-memory evidence.
