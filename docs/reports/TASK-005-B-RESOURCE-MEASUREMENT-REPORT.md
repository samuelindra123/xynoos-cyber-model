# TASK 005-B resource and feasibility measurements

## Boundaries and preflight

The legacy limits remain 64 MiB per FP32 tensor, 128 MiB aggregate ledger,
100,000 values, 100,000 operations, and 100,000,000 units per phase.
The reference also applies the 64 MiB capacity limit to each binary64 attention
probability table. Parameter construction preflights dimensions, all parameter
tensor sizes, inventory bytes, and initialization work. Batch planning checks
shapes, the exact value and operation counts, and lower bounds for forward
dense work and simultaneously retained data. The tape checks cumulative
forward work before recording each operation and the exact charged work for
the reachable reverse graph before allocating reverse buffers. The aggregate
ledger checks and reconciles each allocation separately. Capacities and
temporary overlap are observable as allocations occur.

Checked preflight formulas for the fixed model use `R=B*T`,
`C=R*H*T`, and the inventory parameter count `P`. A model records
`15N+4` operations and `24N+6` values. The forward dense work floor is
`2R*(N*(4D²+3DF)+VD)`. The lower bound above currently charged bytes is
`4*(P+R*(N*(12D+3F)+2D+V)+1) + 8NC + 10R` bytes. This counts
snapshots, retained FP32 outputs, the FP64 probability matrices, and copied
batch constants.

Each owned tensor includes its FP32 data and retained shape capacity. Tape
nodes, reference-operation records, parameter inventory, adjoint and
gradient-batch slots, and scratch vectors charge their reported Vec capacities.
The batch copies own two u32 arrays and two u8 arrays. Attention owns one
FP64 probability table per layer and reuses that table for the score and
softmax passes. Reverse mode keeps a charged FP64 pending array per reached
node until that node is visited and narrowed. The FP32 adjoints remain live
through gradient transfer. Attention reverse additionally has a charged
FP64 contribution array and, for Q or K, one FP64 sequence-wide row for dP.
Embedding scatter and RMS gain reduction have charged temporary arrays.
Detached replacement tables, transfer, persistent accumulation, and the
legacy optimizer retain overlapping charges during their transactions.

The ledger excludes caller-owned tokenizer and dataset buffers, allocator
internals, fixed stack values, the process image, and OS memory overhead.
The `VmHWM` reading is separately reported in KiB for the isolated Linux
test process, including its fixture parser and test harness.

## Exact conservative work charges

Let `X` be the number of cells in a selected input or output as indicated,
`M` the count of supervised positions, `d=D/H`, and `C=BHT²`.
For a projection `K` is the input width and `J` the output width.
Each forward formula is the kernel charge. Each reverse formula is the extra
charge on top of `6X+1` for each differentiable edge with `X` input cells.
These units count arithmetic, nonlinear calls, scans, or updates per the
operation's loop. Index traversal and scalar control are included in the
conservative loop and row allowances. The charges are project accounting units.

| Kernel | Forward | Extra reverse per edge | Charge coverage |
|---|---:|---:|---|
| Embedding gather | `3R+3RD` | `VD+R*(2D+3)` | Mask/ID access, writes/checks, zero and scatter-add |
| Dense or tied vocabulary projection | `RJ*(2K+3)` | `2X*J` for activation, `2X*R` for weights | Sequential products and sums, conversions, output writes/checks |
| Masked feature RMSNorm | `R*(7D+6)` | `R*(13D+10)` for input, `R*(7D+10)` for gains | Row mask, square/reduce, root, scale, input contraction and gain reduction |
| Interleaved RoPE | `18X`, output cells | `18X` | Frequency, phase, trig, pairwise transform, output handling |
| Explicit MHA | `C*(4d+12)+4RD+4` | `RD+C*(8d+10)+4` for Q or K, `RD+C*(2d+10)+4` for V | Full-square bound on permitted keys, scores, stable softmax, PV, dP and weighted row mean, Q/K/V contributions |
| Masked residual | `5X`, output cells | `X` | Row mask, addition and checks, masked edge copies |
| SwiGLU | `10X`, output cells | `14X` | Stable sigmoid, narrowed SiLU, gate multiply and its derivative |
| Masked cross-entropy | `3R+6MV+4` | `3R+9MV+4` | Mask/target scan, max, shifted exp, normalization, mean and seeded VJP |

Forward also charges `24R` for validating and copying the batch. Each
snapshot charges two units per parameter cell for finite checking and copying,
plus the node overhead. A published leaf or operation adds one node unit and
the number of node entries moved if the table grows. Each reference operation
also adds one record unit and the number of reference records moved if that
table grows. Kernel charges cover numerical outputs and output finite checks.

Backward adds `5A+2` for a tape of `A` values, plus five units per cell of
each reachable value for its binary64 pending buffer and checked narrowing.
The per-edge base covers checks and accumulation. Scratch allocated inside
kernels is covered by their additional terms. The legacy gradient transfer
charge is `Kp+2A+4G`, where `Kp` is the parameter entry count and `G`
the summed cells of reached parameter snapshots. Initialization preflight
charges `22P+20Kp` for inventory scans, generated writes, checks, RNG calls,
and transcendental operations. The parameter snapshot charge is a
subset of forward work in the measurements below.

First accumulation into an empty persistent gradient inventory charges
`2Kp+P` when all parameters have gradients. Later accumulation with all
gradients present charges `2Kp+6P` and stages all checked additions before
commit. TASK 005-B applies no optimizer update. Legacy SGD work and staging
retain the [previous contract](../../crates/xynoos-autograd/README.md).

## Hosted measurements

| Measurement | Mathematical B=2,T=5 | TASK 004 integration B=3,T=8 | Candidate B=1,T=32 |
|---|---:|---:|---:|
| Parameters | 7,536 | 62,880 | 914,400 |
| Parameter-only charged bytes | 31,880 | 257,528 | 3,669,304 |
| Completed forward charged bytes | 94,520 | 846,572 | 9,658,268 |
| Forward charged peak bytes | 94,520 | 846,572 | 9,658,268 |
| Peak bytes with requested backward | 203,876 | 1,516,720 | 9,658,268 |
| Charged bytes after gradient transfer | 62,968 | 512,320 | 3,669,304 after failure cleanup |
| Charged bytes after first accumulation | 62,176 | 509,584 | Unreached |
| Values / operations | 30 / 19 | 102 / 64 | 198 / 124 |
| Initialization work | 166,012 | 1,384,120 | 20,118,280 |
| Forward work | 208,911 | 3,728,632 | 66,894,756 |
| Snapshot work included in forward | 15,072 | 125,760 | 1,828,800 |
| Backward charged or preflight demand | 547,964 | 8,794,371 | 152,581,015 demand |
| Gradient transfer work | 30,215 | 251,762 | Unreached |
| First accumulation work | 7,558 | 62,956 | Unreached |
| Isolated test-process VmHWM, KiB | 7,000 | 6,000 | 13,480 |

Measurements above were read from Foundation CI run 45579536215, job
139085784834, on implementation SHA
`1f5066c1f9d3ba0f3d7f5c5ee4216df5963ce07f`.
The charged bytes and work totals are deterministic for the reference
environment. Process RSS varies across runner invocations.

The completed-forward totals include the following logical payloads. The
remainder above the starting ledger plus these components is the net
metadata and capacity charge, including tensor shapes and both tape tables.

| Completed-forward increment | Mathematical | TASK 004 integration | Candidate |
|---|---:|---:|---:|
| FP32 parameter snapshot payload | 30,144 | 251,520 | 3,657,600 |
| FP32 activations, logits, and loss | 25,124 | 292,228 | 2,028,036 |
| Saved FP64 attention probabilities | 800 | 24,576 | 262,144 |
| Owned ID and mask payloads | 100 | 240 | 320 |
| Net metadata and capacity | 6,472 | 20,480 | 40,864 |

## Candidate blocker and equivalent changes

The hosted release fixture initializes the exact candidate, evaluates
`[1,32,292]` logits, and obtains loss 5.736882210. The reverse planner
returns `WorkLimit { pass: Backward, requested: 152581015, limit: 100000000 }`
before allocating reverse adjoints. Disposal restores 3,669,304 charged
bytes and leaves parameter gradients absent. The candidate reverse memory
peak remains unmeasured because reverse arithmetic did not run.

For this shape, the two reverse dense edges alone cost
`4*32*(8*(4*96²+3*96*256)+292*96) = 116,834,304` units.
Eliminating duplicate dP reductions between Q and K would reduce attention
work. Releasing adjoints at their last use or recomputing attention
probabilities could reduce live memory. Neither addresses the dense
contribution floor. Further work needs an independently reviewed,
model-scoped reverse phase policy or a provably equivalent dense schedule
with a separately defined work model. One review proposal is a dedicated
reference experiment policy restricted to these exact model and batch
dimensions, with a 160,000,000-unit complete backward ceiling. That
adds 7,418,985 units of headroom over the measured planner demand and
keeps memory, tensor, and node limits in force. Acceptance would require
a hosted analytic/numerical cross-check of the full candidate reverse pass
and an observed reverse ledger peak. TASK 005-B keeps the existing caps
and the B=1,T=32 research target.
