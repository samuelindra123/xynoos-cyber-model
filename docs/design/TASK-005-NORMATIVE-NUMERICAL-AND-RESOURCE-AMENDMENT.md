# TASK 005 normative numerical and resource amendment

Version 1. This clarification applies to the TASK 005 reference Transformer.
The architecture basis is `f56af8fccbd4d1248ce3919504211c05ae9a3c42`.
The numerical boundaries below describe the seeded implementation at
`97179bcb1af400e56c8b1983590dcad0d4dba946`, as reviewed in the TASK 005-C
audit. Independent Session A review is required for remediation acceptance.

## Initialization stream v1

Use one unsigned 64-bit seed per model. Assign it directly to the SplitMix64
state and start with an empty normal spare. Each raw draw advances the state
by wrapping addition of `0x9e3779b97f4a7c15`. For the updated state `z`,
apply the following transformations in order:

```text
z = (z XOR (z >> 30)) * 0xbf58476d1ce4e5b9
z = (z XOR (z >> 27)) * 0x94d049bb133111eb
z = z XOR (z >> 31)
```

The products wrap at 64 bits and shifts are unsigned. Let `k = z >> 11`.
Convert `k` exactly to binary64, add binary64 `0.5`, cap the rounded
numerator at `2^53 - 1`, then divide by `2^53`. The upper correction
replaces the binary64 rounding of the last midpoint to 1 with the adjacent
representable value below 1. The minimum is `2^-54`. Every uniform result
lies strictly inside the unit interval.

Register E first. Proceed in block execution order, registering attention
RMS gain, Q, K, V, O, FFN RMS gain, gate, value, and down, followed by final
RMS gain. Within each tensor use row-major order. There is one E registration.

- E takes Box–Muller pairs. Draw `u1` then `u2`, compute
  `r = sqrt(-2 * ln(u1))` and `a = (2 * pi) * u2` in binary64, and form
  `r*cos(a)` and `r*sin(a)` in binary64. Narrow both to binary32.
  Consume cosine first and consume the saved sine on the next normal request
  without any new draw. Widen the chosen normal to binary64, multiply by
  binary64 `0.02`, and narrow to the stored binary32 embedding entry.
- Each matrix coefficient consumes one uniform draw. Compute the Xavier
  bound `sqrt(6 / (fan_in + fan_out))` in binary64. Multiply the bound for O
  and down by the binary64 `1/sqrt(2N)`. Compute `(2u - 1) * bound` in
  binary64 and narrow once to the stored binary32 coefficient.
- All RMS gains are exactly binary32 1 and consume zero draws. A saved
  normal belongs only to the normal stream. Matrix draws advance SplitMix64
  directly.

The seed records the entire stream for this version. Same-reference fixture
verification checks the initialized binary32 bits. Cross-backend arithmetic
is evaluated with the specified parity tolerances.

## Reference rounding v1

Parameters, activations, logits and scalar loss are stored as binary32.
Kernel conversions to binary32 use IEEE rounding to nearest, ties to even.
Binary64 products and additions in reductions follow the given order with
separate rounding for multiply and add.

| Operation | Reference boundary |
|---|---|
| Gather and residual | Gather copies a binary32 embedding value. Residual adds binary32 inputs in binary32. Invalid rows are exact zero. |
| Dense and tied projection | Widen inputs and weights. Multiply and sum in binary64 in increasing contraction-index order. Narrow each result once. The activation VJP reduces over output features and the weight VJP over flattened batch and time in increasing order. |
| RMSNorm | Sum feature squares, divide by width, add binary64 epsilon `1e-5`, and take square root in binary64. Compute gain times input divided by the RMS in binary64 and narrow each output. The VJPs perform their feature and valid-row reductions in binary64. |
| RoPE | Compute frequency and phase in binary64, evaluate cosine and sine in binary64, and narrow each coefficient to binary32. Widen those coefficients and the pair inputs for the binary64 products and signed sums. Narrow each pair result independently. Reverse negates the widened sine and uses the same product and narrowing boundaries. |
| Attention | Use sequential binary64 QK dot products over increasing feature indices and multiply by binary64 `1/sqrt(head_width)`. Scores, max shift, exponentials, sequential softmax denominator and probabilities are binary64. Visit valid causal keys in sequence order. Excluded positions have exact zero probability. Apply probabilities to binary32 V using a sequential binary64 sum over keys, then narrow each attention activation once. The Q, K and V VJPs reduce in binary64 over query rows, heads and causal keys in increasing order, and narrow once per local edge element. |
| SwiGLU | Compute binary64 `e = exp(-abs(z))`. Use `1/(1+e)` when `z >= 0` and `e/(1+e)` otherwise. Multiply binary64 `z` by that sigmoid and narrow SiLU to binary32 before multiplying by the binary32 value activation. The gate VJP uses the binary64 sigmoid and derivative before final local narrowing. The value VJP uses the narrowed forward SiLU. |
| Cross-entropy | For each supervised row compute binary64 `log(sum(exp(z - max))) + (max - z[target])`. Sum terms in vocabulary order and losses in flattened batch/time order. Divide by the supervised count in binary64 and narrow the scalar once. The loss VJP recomputes the stable probability in binary64, subtracts the target indicator, multiplies by the upstream seed and divides by the count before local narrowing. |
| Local edges | The VJP takes the node's already narrowed upstream adjoint. Every edge publishes its local element as binary32, after the local reduction where applicable. A scatter-add first sums repeated IDs in flattened batch/time order in binary64 and narrows each embedding coordinate once. |
| Node fan-in | Widen each finite local contribution to binary64 and add it to the node's pending array in reverse node traversal and operation-edge order. Narrow each accumulated element when visiting the node and check finiteness before use and final adjoint publication. The one E snapshot receives the output projection edge first and the lookup edge later. |

Forward model outputs and analytic comparisons use absolute plus relative
tolerance `2e-5 + 2e-5*abs(expected)`. Central finite-difference comparisons
retain `2e-3 + 2e-2*abs(fd)`. Existing legacy tape arithmetic follows its
own rounding path. The amendment changes no seeded fixture, kernel equation
or legacy arithmetic.

## Batch and failure lifecycle

Both ID arrays and supplied masks have shape `[B,T]`. Missing masks
imply all ones. Validate all IDs and binary masks, including unsupervised
positions. A valid prefix may end with a PAD target only at loss mask 0.
Between adjacent valid inputs the target must equal the next input even
with loss mask 0. An invalid input position requires PAD input, PAD target
and loss mask 0. DOC_SEP is invalid in ordinary content. Zero supervised
positions are a typed error. The loss arrangement in the table preserves
the log term when maximum logits are equal and near the binary32 limit.

A model evaluation owns the tape and releases it on a forward error.
Individual legacy tape operations retain their existing forward retry
semantics. Reverse failure seals the tape, releases reverse temporaries
and publishes no adjoints. A failed model-level reverse or transfer
consumes the evaluation and releases its tape. Accumulation consumes its
batch on success or failure, checks origin and revision, and commits only
after all staging succeeds.

## Candidate resource policy v1

`Transformer::evaluate_candidate` explicitly selects
`CandidateReferenceExperimentPolicy` for the validated configuration
`V=292, D=96, N=8, H=4, d=24, F=256, context=128` and `[B,T]=[1,32]`.
The inventory contains 74 entries and 914,400 scalar parameters. Autograd
requires the immutable candidate inventory designation and checks every
registered shape before it records the fixed graph. The resulting
evaluation privately owns its tape. The loss is its only reverse root.
No caller can append operations or transfer the policy to an ordinary tape.

The policy sets a ceiling of 160,000,000 for this complete backward pass
when the parameter set uses the default work ceiling. A caller's lower
work ceiling takes precedence. Default backward and forward work are
100,000,000. Parameter insertion, transfer, accumulation, optimizer and
experiment-step limits retain their original contracts. The 64 MiB tensor
limit, 128 MiB aggregate ledger, and 100,000 tape values and operations
remain in force.

The checked reachable-graph planner is the pre-allocation gate for reverse.
It retains the original scan, pending-adjoint, local kernel and fan-in
charges. Backward work equal to the applicable limit is permitted and any
greater requirement returns a typed work-limit error. Forward state stays
charged while reverse uses it. The operation formulas and ledger inclusions
remain as specified in the [TASK 005-B resource report](../reports/TASK-005-B-RESOURCE-MEASUREMENT-REPORT.md).
