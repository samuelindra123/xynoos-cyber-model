# TASK 005-C2 independent candidate re-verification

## Scope and decision

This audit resumes PR #16 from `3f70303efb3da23c77a10cd573f2c0f48ff23217`
on `audit/task-005-c2-candidate-reverification`. It adds auditor-authored
controls and does not modify production behavior. It does not claim exhaustive
verification of every one of the 914,400 gradient coordinates.

**Gate A — resource-policy confinement: ACCEPT.** Explicit opt-in, exact model
configuration, candidate inventory identity, batch shape, ordinary work cap,
caller cap, fixed graph and rejection cleanup were checked.

**Gate B — independent Transformer gradients: ACCEPT.** Twelve additional
fixed coordinates were selected in the oracle before comparison and cover the
shared embedding, attention projections, RMSNorm gains, SwiGLU projections,
multiple blocks and final gain. Twenty-four native central differences pass
at the two required steps. A Transformer-level tied-edge omission control
detected a `0.063402506057` disagreement.

**Gate C — numerical amendments: ACCEPT.** Initialization stream v1 and
reference rounding v1 agree across the normative amendment, Rust source and
the independent standard-library Python oracle. Small independent fixtures
and the candidate exercise support these decisions; details follow below.

**Gate D — gradient transactions: ACCEPT.** Parameter-set and revision
binding, complete gradient transfer, successful accumulation, foreign and
stale rejection, bit preservation, injected rollback and resource cleanup
were exercised with the candidate helpers.

The final overall verdict is pending successful Foundation CI on the exact
committed audit head. No owner acceptance, merge or TASK 006 work is authorized
by this report.

## Gate A — resource-policy confinement

Source review covered `Transformer::evaluate_candidate`,
`ReferenceTransformerEvaluation::record_candidate`, the private fixed graph,
the candidate inventory marker, and the reachable backward-work planner.
The upper-level entry point requires exact `V=292, D=96, N=8, H=4, F=256,
context=128` and `[B,T]=[1,32]`. The lower-level entry point requires the
candidate inventory designation, 74 entries and every exact ordered shape;
matching tensor shapes in an ordinary inventory do not confer the designation.
The fixed graph has no caller mutation API and `backward` roots only at its
recorded loss.

Executed controls include:

- Existing wrong-configuration controls for V, D, N, H, F and context, plus
  wrong B/T rejection. Configuration and batch failures preserve the parameter
  ledger, revision and empty gradients.
- New `c2_matching_shapes_without_candidate_inventory_identity_are_rejected`:
  an ordinary, unmarked set with the exact 74 candidate shapes and 914,400
  values is rejected as `InvalidCandidateInventory`; values, revision,
  gradients and ledger peak remain unchanged.
- The ordinary candidate evaluation path requests 152,581,015 units and is
  rejected at the 100,000,000 default cap, with evaluation allocations
  released and no gradient published.
- The scoped evaluation succeeds at 152,581,015 under the 160,000,000 policy
  ceiling. The stricter caller limit of 99,999,999 remains effective and
  rejects before reverse allocation. `ResourceLimits` clamps public work
  requests at 100,000,000, so a caller cannot request an intermediate cap
  above that ordinary ceiling.
- Injected reverse, transfer and staged-accumulation failures preserve state
  and release temporary charges. Lower-level inventory and shape rejections
  happen before graph recording.

The candidate ceiling is selected only for an explicitly supplied policy and
the complete candidate graph. Ordinary tapes do not inherit it.

## Gate B — independent gradient checks

The original fixture retains its 225 direct comparisons, 222 aggregate
comparisons, 12-coordinate finite-difference set, shared-E path fixture and
scalar tied-edge check. The oracle now also fixes these twelve additional
coordinates in `tools/transformer_oracle.py` before observing errors:

| Family | Selected coordinates |
|---|---|
| Shared embedding | E[5095] |
| Attention RMS gain | block 6, index 71 |
| Q / K / V / O | blocks 2/6/4/7, indices 3011/811/5377/1300 |
| FFN RMS gain | block 3, index 82 |
| Gate / value / down | blocks 1/5/6, indices 22033/20123/17003 |
| Additional Q | block 0, index 809 |
| Final RMS gain | index 49 |

The Python oracle independently differentiates the full Transformer equations
and freezes each selected gradient. All 12 native gradients matched with
zero error. Native central differences use the actual representable FP32
high/low perturbation distances and were compared with both native and
independent analytic gradients:

| Step | Coordinates | Maximum absolute error | Maximum relative error for nonzero FD |
|---:|---:|---:|---:|
| 0.001 | 12 | `1.419536587e-4` | `3.108333221e-1` |
| 0.003 | 12 | `4.438590993e-5` | `2.261671338e-1` |

All 24 checks pass the unchanged tolerance `2e-3 + 2e-2*abs(fd)`. Relative
error is sensitive to derivatives near zero. The native Transformer negative
control drops the lookup contribution from tied E at E[1]; the resulting
gradient disagrees with the actual shared adjoint by `0.063402506057`, above
its `0.01` detection threshold. The independent small one-block operation
fixture separately checks forward values and adjoints for 6,281 operation
cells each, including RMSNorm, RoPE, attention, SwiGLU, loss and local
backpropagation.

## Gate C — normative numerical amendments

### Initialization stream v1 — ACCEPT

The normative specification, `init.rs`, and the standard-library Python
oracle agree on SplitMix64 state transition and seed, top-53 conversion,
midpoint endpoint correction, Box–Muller draw order and saved sine use,
registration order, FP32 storage narrowing, E scale, Xavier bounds, residual
scaling and zero-draw unit RMS gains. The oracle asserts the first seed
transition/output (`state=0xf690c808ce9d7c1a`, `z=0x5dfd1045191ba975`) and
both uniform endpoints (`2^-54` and the representable value below 1).
The exact candidate initialization checks every one of the 74 per-tensor
FP32 hashes against the independently generated fixture. The same fixture
preserves all earlier parameter values and identities.

### Reference rounding v1 — ACCEPT

The independent small operation fixture matches Rust forward values and
adjoints for 6,281 cells in each direction with zero maximum error. The
candidate's 9,345 logits/loss values match exactly; its original 225 gradient
coordinates and 222 full-inventory aggregates remain within the recorded
tolerances, and the additional 12 coordinates match exactly. Candidate
finite differences pass at both established steps. Together with the
independent Python equations and the normative operation table, these checks
cover sequential FP64 dense reductions; RMSNorm; RoPE coefficients and pair
transforms; attention probabilities, weighted sums and VJPs; narrowed
SwiGLU; stable cross-entropy; locally narrowed VJPs; sequential FP64 fan-in;
both shared-E contributions; and FP32 adjoint publication. Small hand
attention, extreme-logit, mutation and omission controls distinguish the
specified equations from the relevant incorrect alternatives. No tolerance
was changed.

## Gate D — transaction verification

The exact candidate completes reverse and publishes all 914,400 finite
gradient entries for the 74 registered tensors at revision 74. Transfer
returns a complete origin-bound batch; accumulation into the empty persistent
inventory retains the transferred buffers and preserves all parameter bits
and revision. A batch from a different parameter set returns
`WrongParameterSet`; the receiving parameter and gradient bits, revision and
ledger remain unchanged, and the source batch allocation is released.

The candidate transaction control also verifies stale-revision rejection,
allocation failure during reverse and transfer, and failure after four staged
accumulation replacements. Existing gradient bits, parameter bits and
revision remain unchanged on failures; temporary charges are released. After
destroying the isolated candidate set the observed ledger is zero. The
original successful and failure-path controls remain intact.

## Preserved candidate and regression evidence

The exact candidate still has 914,400 parameters, 74 parameter tensors and
eight blocks. It publishes a complete finite gradient inventory. The
measurements remain:

| Contract | Observed |
|---|---:|
| Forward work | 66,894,756 |
| Backward work | 152,581,015 |
| Scoped / default backward limits | 160,000,000 / 100,000,000 |
| Combined charged peak | 16,037,368 bytes |
| Gradient transfer work | 3,658,070 |
| First accumulation work | 914,548 |

The hosted baseline preserved 109 checks in each profile plus the existing
auditor-authored scalar control. The four bounded candidate experiments,
all earlier numerical, gradient, tokenizer, data and Transformer fixtures,
resource contracts, and failure-injection tests remain in CI. The new
identity control is ignored in the default check count and invoked explicitly
as a bounded release test. `candidate.json` now includes the C2 fixed
coordinates (SHA-256 `927628202d321b61576092db3a2f6a7289b85c0badb8f369af88f41702bddde5`);
the tied-path fixture remains unchanged (SHA-256
`1fdaa1553ae3415a88240c20a858d19a94fd8c1d8776cabda8e5e692c4af9f21`). The
learning and tokenizer-data digests remain those recorded in the R report.

The accounting is the application's resource ledger, not process RSS. Timing
samples are bounded reference-test observations, not production throughput
claims. Coordinate sampling is not exhaustive gradient verification.
Sustained training, model quality, broader configurations, hardware
compatibility and TASK 006 planning remain outside this audit. Future
authorized long-running jobs must run independently of continuous agent
supervision where supported, with deterministic monitoring, checkpointing,
failure detection, resource limits and termination. This audit did not
provision Colab or run training.

## Exact-head hosted verification

Pending final audit commit and Foundation CI. The CI URL, job, exact SHA and
hosted log findings will be recorded here before the report is accepted as
complete.
