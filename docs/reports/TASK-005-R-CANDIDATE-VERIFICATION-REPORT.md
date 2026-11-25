# TASK 005-R candidate verification

## Hosted evidence

The initial complete candidate measurement was recorded by Foundation CI
run **45584705085**, job **139103232033**, on source
`1ba32510e3b96811f839cdbf126c02351b179c87`. The hosted job confirmed its
checkout revision and concluded successfully. The draft PR records the
exact-head rerun after the separate native tied-edge check and documentation
were added.

The oracle uses the Python standard library and independently reconstructs
seed `0x58594e4f4f530005` with frozen configuration and token streams.
It evaluates the full decoder and independently differentiates its
equations. The initial build has token IDs 0 through 31 and targets 1
through 32, with all positions valid and supervised. An auxiliary batch
substitutes token 3 for token 17 and retains valid next-token shifts.
An explicit-weight auxiliary evaluation uses the same repeated-token batch
and gives a test seam for independently checking the two native E edges.
Fixture serialization is deterministic.

| New fixture | Bytes | SHA-256 |
|---|---:|---|
| `candidate.json` | 123,168 | `46cdcf67b819fcc6658938331728d588c21dbcdf14f047b3938268ea2ee2df59` |
| `candidate-tied-paths.txt` | 254 | `1fdaa1553ae3415a88240c20a858d19a94fd8c1d8776cabda8e5e692c4af9f21` |

## Forward and backward evidence

The model registered 74 tensors with 914,400 FP32 parameter elements.
All initialized values were hashed as little-endian binary32 with FNV-1a
and checked against the independent per-tensor hashes. Input and target
shapes were `[1,32]`, tokenizer vocabulary was 292, and logits had shape
`[1,32,292]`. All 9,344 logits and the masked mean loss of 5.736882210
agreed with the oracle. The maximum absolute and nonzero-reference
relative errors were both zero.

The native reverse pass completed through eight blocks. All 914,400
published gradient elements were finite. Each registered parameter had one
corresponding gradient with the expected shape. The comparison below covers
each gradient family and each block. Three coordinates per block tensor
sample the lower, middle and upper regions of its flattened row-major
storage, offset by parameter ordinal. E includes positions used by
lookup, a repeated-token coordinate, and positions outside the lookup set.

| Family | Shape | Native entries | Direct comparisons on the original batch |
|---|---|---:|---:|
| Shared embedding E | `[292,96]` | 1 | 6 |
| Attention RMS gain | `[96]` | 8 | 24 |
| Wq | `[96,96]` | 8 | 24 |
| Wk | `[96,96]` | 8 | 24 |
| Wv | `[96,96]` | 8 | 24 |
| Wo | `[96,96]` | 8 | 24 |
| FFN RMS gain | `[96]` | 8 | 24 |
| Gate | `[96,256]` | 8 | 24 |
| Value | `[96,256]` | 8 | 24 |
| Down | `[256,96]` | 8 | 24 |
| Final RMS gain | `[96]` | 1 | 3 |
| Total | | 74 | 225 |

Each block from 0 through 7 contributes nine entries and 27 comparisons.
The 225 comparisons had zero maximum absolute or relative error, zero
tolerance violations and zero nonfinite results. Independently checking
the L1 sum, squared L2 sum and maximum magnitude of all 74 tensors gave
222 aggregate comparisons, with maximum absolute error
`7.105427358e-15` and relative error `2.114338218e-16`. The aggregate
tolerance is `1e-10 + 2e-5*abs(expected)`. Aggregates summarize all gradient
values. Direct coordinate comparisons cover the 225 listed samples.

## Central differences

One fixed coordinate from each of the nine block entries was chosen across
blocks 0 through 7, with block 0 contributing both attention gain and down.
Two E coordinates and one final gain coordinate complete the selection.
Both steps used 12 coordinates and actual representable FP32 perturbation
distances. Native central differences use 48 forward-only evaluations and
are checked against both native and independent analytic gradients.
The table reports error versus the Python analytic reference.

| Step | Compared | Maximum absolute error | Maximum relative error for nonzero FD |
|---:|---:|---:|---:|
| 0.001 | 12 | `1.480514240e-4` | `2.304521028e-1` |
| 0.003 | 12 | `4.906307896e-5` | `3.031127241e-1` |

All 24 differences passed `2e-3 + 2e-2*abs(fd)`. Relative error is
sensitive to a small derivative in the denominator. Binary32 scalar
rounding in the two loss evaluations also affects the loss difference.

## Tied embedding and gradient transactions

The gather and vocabulary projection point to the same tape parameter
snapshot. The independent computation narrows the lookup scatter-add and
the projection VJP separately, then sums the two paths in binary64 and
narrows once. The original and repeated-token native adjoints each match
the six selected combined coordinates and three E aggregates. The Python
omission controls for the original batch change the maximum gradient
coordinate by 0.354800334 when lookup is omitted and 0.0946526825 when
projection is omitted. The repeated-token controls give 0.366682491 and
0.123294633 respectively. The explicit-weight native seam compares six
lookup, six projection, and six combined values to independent
expectations and checks that the narrowed sum matches the one E snapshot.

Transfer yields a native batch at revision 74. Origin-bound parameter
handles resolve all gradients. The tape's allocations are released
and the batch retains the parameter gradient buffers and batch slots.
Accumulating into the initially empty persistent inventory retains those
same buffers and preserves parameter bits and revision. The case that
uses the repeated-token model's batch at the original model returns
`WrongParameterSet`, preserves the original state and cleans up its
origin's batch allocation. An isolated autograd case exercises a stale
candidate batch by advancing the revision with inventory insertion. It
also injects allocation failure during reverse, transfer and after four
staged replacements during accumulation. The failed operations release
their temporary charges and leave the existing state intact. Cleanup
of this isolated candidate set leaves zero charged bytes.

The original measurement has one successful seeded reverse and one
successful auxiliary seeded reverse. The isolated control runs four
complete explicit-weight reverses for failed transfer, two batches for
the accumulation rollback, and stale rejection. It also stops one reverse
with an allocation fault. These are bounded gradient experiments.

## Resource measurements

Measured ledger bytes refer to the original candidate's parameter set.
Work units are the configured accounting definition, and elapsed phases
are wall-clock samples from run 45584705085.

| Phase or state | Work units | Live or peak ledger bytes | Elapsed ms |
|---|---:|---:|---:|
| Initialization, retained | 20,118,280 | 3,669,304 | 2.812 |
| Forward, peak and retained | 66,894,756 | 9,658,268 | 60.796 |
| Backward, requested and completed | 152,581,015 | 16,037,368 peak | 55.503 |
| Combined forward/backward peak | | 16,037,368 | |
| Gradient transfer, retained | 3,658,070 | 7,333,280 | 0.034 |
| First accumulation, retained | 914,548 | 7,327,952 | 0.323 |
| Auxiliary set after tape and rejected batch cleanup | | 3,669,304 | |
| Isolated control set after destruction | | 0 | |
| Central differences including 48 forwards | | | 2,893.188 |

There were 198 recorded values and 124 operations. Backward had
7,418,985 units of headroom against 160,000,000. The combined peak was
within the 134,217,728-byte aggregate limit. Buffers passed their existing
allocation-level limits. Gradient payload was 3,657,600 bytes, with 1,048
bytes of shape storage per complete gradient inventory. Transfer also
retained 5,328 bytes for its batch slots. The accounted peak includes the
retained forward graph and saved state. The native test process reported
28,832 KiB VmHWM before finite differences, including its fixture and
auxiliary evaluation. Process memory includes allocations outside the
resource ledger. The original release measurement comprises one phase
sample per operation and provides no production throughput estimate.

## Regressions and limits of evidence

The hosted run passed 109 checks in each of the debug and release
profiles, including the 105 existing checks and four ordinary remediation
controls. Four bounded candidate experiments are invoked separately,
including the default-policy rejection. Foundation CI also verifies exact
checkout, locked acquisition, offline Cargo, formatting, Clippy with
warnings denied, Python oracles, Rustdoc, privacy and artifact policy.

| Preserved fixture or experiment | SHA-256 |
|---|---|
| Learning trajectory | `e50d201a133838fe64158e0dcf94df894efcf17241f1f16e52232a4e75d41c6a` |
| Tokenizer and data experiment | `3cd12b220e38e423e5dab918b30d86708d64c8fa5fb9c8c1c8168d51bfb8b5d1` |
| `reference.json` | `8c92aa8bf96a2c69b53440026f265e54609c4d4ab874bdf9fe4cfb8dbba2e71d` |
| `parameters.json` | `7fd60d86c5fb1a81197410fe971d8bd5bbd9018fd500f9dff2e48049eaf7e88d` |
| `operations.json` | `15ed8f5d77a576639e3c39d460713830b06e62381570d8bb4eae01c83a9f81b3` |

The evidence establishes the bounded candidate forward, reverse and
gradient handling at the listed shapes. Other batch sizes and model
configurations use the existing resource contracts. Numerical verification
covers the frozen inputs and coordinate/aggregate sets. Extrapolation
to production training, larger models or model competence requires
separate experiments and authorizations, after independent re-verification.
