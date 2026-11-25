# TASK 005-B native Transformer implementation

## Baseline and architecture

The draft implementation is PR #13 on `feat/task-005-b-native-transformer`,
based on accepted main `334beb088f116240cdd0346a70d9527c26f08e51`.
The contract is the exact documentation at planning head
`f56af8fccbd4d1248ce3919504211c05ae9a3c42`, in separate draft PR #12.
Planning Foundation CI run 45573907506, job 139066987645, passed on that head.

The native decoder uses a learned token embedding, pre-RMSNorm with ε=1e-5,
ordinary causal multi-head attention, interleaved RoPE, SwiGLU, two masked
residuals per block, a final RMSNorm, a tied vocabulary projection, and
masked mean next-token cross-entropy. Shapes are `[B,T]` u32 IDs,
`[B,T,D]` activations, and `[B,T,V]` logits. There are no linear biases.
One actual E parameter snapshot serves the input and output paths.
`P=VD+D+N*(4D²+3DF+2D)` matches the registered inventory of each configuration.

| Reference | V | D | N | H | d | F | Context cap | P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mathematical | 260 | 16 | 1 | 2 | 8 | 48 | 8 | 7,536 |
| Integration | 292 | 32 | 4 | 4 | 8 | 96 | 64 | 62,880 |
| Research candidate | 292 | 96 | 8 | 4 | 24 | 256 | 128 | 914,400 |

## Shape, numerical and reverse contracts

The [crate API](../../crates/xynoos-transformer/README.md) registers parameters
in deterministic order and composes the existing native tape. Projection
treats `B*T` as leading rows. Head indexing splits each activation row into
contiguous segments, so the implementation needs neither a generic reshape
nor a permutation/batched matrix primitive. The tensor crate is unchanged.

- Gather returns the selected embedding row on valid positions and +0 on
  padding. Its VJP scatter-adds each row's adjoint in increasing batch/time
  order. The projection VJP adds the transpose/weight contributions to that
  same parameter. Dense sums are increasing-index binary64 reductions.
- Per feature row, RMSNorm uses `r=sqrt(mean(x²)+ε)` and `y=γ*x/r`.
  Its input VJP is `g*γ/r-x*sum(g*γ*x)/(D*r³)`. The gain VJP reduces
  `g*x/r` across valid rows. Masked rows are zero in both directions.
- RoPE applies the fixed 2×2 rotation to each interleaved pair at
  `t*10000^(-2j/d)` and multiplies the VJP by the transposed rotation.
  Trig rounds to FP32 and the paired transforms narrow from FP64 at output.
- For a valid query, attention visits only keys in the same sample up to
  the query. `p=exp(z-max)/sum(exp(z-max))` with
  `z=dot(q,k)/sqrt(d)`. It saves FP64 probabilities, including exact zeros
  for exclusions, and returns the narrowed weighted V sum.
  `dP=dA·V`, `dZ=P*(dP-sum(P*dP))`, `dQ=dZ*K/sqrt(d)`,
  `dK=dZ*Q/sqrt(d)`, and `dV=P*dA`, with the corresponding sequence
  reductions. A padded query contributes zero.
- The stable sigmoid branches by sign and never exponentiates a positive
  magnitude. SwiGLU uses `silu(gate)*value`. It sends
  `upstream*value*(σ+gate*σ*(1-σ))` to the gate and
  `upstream*silu(gate)` to the value projection. A masked residual
  supplies the same masked upstream adjoint to both edges.
- Loss validates ID and mask shape, then reduces
  `log(sum(exp(z-max))) + (max-z[target])` over supervised rows.
  `(P-one_hot)/M` is the logit VJP for a unit seed and produces exact
  zero on unsupervised rows. At equal FP32 maximum logits, the shifted form
  retains the logarithm. An unrepresentable final FP32 result is a typed
  numerical error.

The reference reverse pass checks each edge contribution, sums each node's
fan-in sequentially in FP64, then narrows once before that node's VJP.
The parameter set retains identity and revision checks and transactional
gradient batches. The model drops an evaluation on failure, and reverse
failure seals the underlying tape. A unit-test injection exercises allocation
failure in both directions. Legacy operations retain their original arithmetic.

## Dataset and initialization

The input and target IDs are independently checked, including masked
positions. Only 0/1 masks and a valid prefix are accepted. PAD targets
use loss 0 and can follow the final valid input. The accepted TASK 004
stream pads both sides and emits exact shifts over one document at a time.
The integration test fits the accepted BPE fixture on its train split,
uses the resulting 292-token vocabulary, and reads the validation split
with both actual and padded final batch sizes. Those layouts agree exactly
on shared logits and loss. The added rows are exact zero in the logits.

Seed `0x58594e4f4f530005` reconstructs all 7,536 seeded binary32 values
against an independent implementation and a second native initialization.
The sequence is E, then attention gain and Q/K/V/O, then FFN gain and
gate/value/down per layer, then final gain. Gains are ones, embedding uses
Box–Muller scaled to standard deviation 0.02, matrices are Xavier uniform,
and O/down apply the `1/sqrt(2N)` factor. See [scope decisions](TASK-005-B-REQUIREMENT-TRACEABILITY.md)
for the binary64 endpoint and rounding choices requiring independent review.

## Oracle evidence

[The standard-library Python program](../../tools/transformer_oracle.py)
generates canonical sorted JSON, little-endian four-byte binary32 hex
strings, explicit parameter arrays, seeded initialization, batch metadata,
forward and reverse operation tables, tied gradient paths, and two central
difference steps. The full mathematical topology and four-block logits
come from Python arithmetic. The hand attention reference uses two
tokens with Q=K equal to the basis vectors and V rows (2,0) and (0,4).

| Frozen fixture | SHA-256 |
|---|---|
| `reference.json` | `8c92aa8bf96a2c69b53440026f265e54609c4d4ab874bdf9fe4cfb8dbba2e71d` |
| `parameters.json` | `7fd60d86c5fb1a81197410fe971d8bd5bbd9018fd500f9dff2e48049eaf7e88d` |
| `operations.json` | `15ed8f5d77a576639e3c39d460713830b06e62381570d8bb4eae01c83a9f81b3` |

The mathematical loss is 5.449348450. The 2,601 mathematical logit/loss
values, all 7,536 parameter gradients, and all 4,160 summed E contributions
match with zero maximum absolute and relative error in the reference
environment. The four-block fixture compares 2,921 logit/loss values and
76 gradient coordinates across all four blocks and all parameter families
with zero maximum absolute and relative error. Both directions of the
19 operation tables match across 6,281 values each with zero maximum absolute
and relative error. The comparison tolerance is
`2e-5 + 2e-5*abs(expected)`.

| Mathematical central step | Coordinates | Max absolute error | Max relative error for nonzero FD |
|---|---:|---:|---:|
| 0.001 | 23 | 1.640543342e-4 | 1.621079102e-1 |
| 0.003 | 23 | 7.334165275e-5 | 2.568380964e-1 |

| Four-block central step | Coordinates | Max absolute error | Max relative error for nonzero FD |
|---|---:|---:|---:|
| 0.001 | 4 | 1.380359754e-4 | 1.600988770e-1 |
| 0.003 | 4 | 4.443409853e-5 | 7.987291015e-2 |

Both native and independent central differences produce the listed errors.
The finite-difference tolerance is `2e-3 + 2e-2*abs(fd)`. Relative maxima
can occur on small derivatives. The mathematical coordinates cover every
parameter family. The four-block case also selects one additional coordinate
per block at both steps.

## Executed controls

| Mutation | Maximum absolute disagreement |
|---|---:|
| Attention scale set to 1 | 0.335858703 |
| Future keys allowed | 0.423488975 |
| Head / feature layout interchanged | 0.109036461 |
| RoPE rotation sign reversed | 0.284113064 |
| RMSNorm applied along sequence | 0.478551537 |
| SiLU derivative replaced with sigmoid | 0.00245086849 |
| Residual input omitted | 0.924205214 |
| Targets at offset 0 | 0.567856729 |
| PAD target included in supervision | 0.29706338 |
| Output contribution omitted from tied E | 0.314389564 |
| Loss VJP mean divisor omitted | 2.1656847 |

Naive exponentiation on the three extreme cases raises `OverflowError`
in the oracle. The native stable cases return 0.313261688, 5.560681820,
and 0.000000000 with finite gradients and exact unsupervised zeros. Rust
also executes typed input, numerical, node, operation, tensor, aggregate,
allocation, forward work and reverse work controls. Invariance checks
cover future mutation, batch isolation, separate-versus-batched evaluation,
padding equality, padding gradients, and exact masked zeros.

The aggregate-inventory control returns `MemoryLimit` at 31,880 requested
bytes against 1,024. The tensor control fails at 68,157,440 requested bytes
against 67,108,864. Reverse controls return `MemoryLimit` at 95,432 against
94,520 bytes and `WorkLimit` at 547,964 against 300,000 units.
A forward work control returns `WorkLimit` for 29 against 28 units.
Forward finite-input overflow and NaN injection return `NonFinite`. The
injected allocation fails exactly as `AllocationFailed` on both forward
and reverse, with no published partial results.

## Validation and remaining review

Foundation CI runs locked acquisition followed by offline workspace check,
formatting, Clippy, debug and release tests, doctests, legacy and new
oracles, deterministic comparisons, privacy and Markdown checks, and
Rustdoc with warnings denied. Results from the exact-head PR runs and
numbers for the two reference gates and candidate appear in the
[resource report](TASK-005-B-RESOURCE-MEASUREMENT-REPORT.md).
Foundation CI preserves 90 accepted Rust checks per profile, plus 15 new
checks per profile and a separate hosted-only candidate check.
Legacy regression evidence includes 295 numerical cases, 2,670 expected
outputs, six numerical controls, 25 gradient coordinates, three gradient
controls, transactional SGD and the 100-step learning experiment with
101 observed states, and nine tokenizer/data controls. The accepted
trajectory SHA-256 is
`e50d201a133838fe64158e0dcf94df894efcf17241f1f16e52232a4e75d41c6a`.
The tokenizer experiment digest is
`3cd12b220e38e423e5dab918b30d86708d64c8fa5fb9c8c1c8168d51bfb8b5d1`.

The initial run (45578967619) caught a test binding compilation error.
Run 45579141607 completed the numerical suites and caught unescaped shape
references in API Rustdoc. Both defects were fixed before the successful
implementation run 45579536215, job 139085784834, on
`1f5066c1f9d3ba0f3d7f5c5ee4216df5963ce07f`. PR #13 records the final
documentation and test head together with its exact-head Foundation check.

The candidate completes forward and hits the legacy backward work cap at
152,581,015 requested units. The resource report preserves the target and
derives the dense contribution floor. No weight updates, model training,
external corpus acquisition, or compute provisioning occurred. The
reference has no claim of production throughput or model quality.
