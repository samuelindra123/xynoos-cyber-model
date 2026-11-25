# XYNOOS reverse-mode differentiation

## Lifecycle and numerical contract

A parameter set fixes the resource policy for its entire lifetime. Every tape borrows the set and inherits its limits. A tape accepts forward records until its first backward attempt. Success makes adjoints readable and seals the tape. Any backward error discards its adjoints and permanently seals the tape. Forward failures publish no value. An owned gradient batch consumes a successful tape and ends the parameter borrow.

An input is a constant leaf. MSE accepts only a constant input leaf as its target. Its prediction and target have identical nonempty shapes. Differences, squares, sequential row-major accumulation, division, and the backward expression `((g * 2) * (p - target)) / n` use binary64. Each output or edge contribution converts once to binary32 and must be finite.

Disconnected values and constants have missing adjoints. A connected zero contribution produces an explicit gradient. Disconnected parameters retain their previous persistent gradient, if any. Call `zero_grad` to install explicit zeros across the inventory when an update should include disconnected parameters. SGD requires a gradient for each parameter and clears all gradients after commit.

Matmul work includes output initialization and a finite-value scan even for zero inner dimension. An empty output skips the cell kernels. Each derivative product has its own output cell count and inner dimension, so empty forward outputs can still yield nonempty zero derivatives.

## Resource contract

Charge simultaneously retained tensor data and shape capacities, metadata vector capacities, and staging buffers across all tapes and owned batches associated with the same set. External inputs, allocator bookkeeping, fixed-size handles, stack, and process RSS are outside the accounting boundary. Vector growth stages a replacement before moving entries and charges both old and new capacity. A detached allocation with excess reported capacity is released on reconciliation failure. Reservations preflight the requested capacity, and any capacity reported above the request can exist transiently during reconciliation. Fallible vector reservations handle expected allocation errors. The fixed shared accounting handle uses the standard allocator.

## Public API

Create a `ParameterSet` with `ResourceLimits`, then use `insert`, `value`,
`gradient`, `revision`, and `accounted_bytes` to manage or inspect the inventory.
IDs identify their owning set or tape as well as a slot. Stable parameter slots
survive insertion and successful SGD. Neither owner identity nor revision wraps.
An insertion invalidates outstanding batches through a revision advance while
preserving existing persistent gradients for unchanged parameters. `zero_grad`
and gradient accumulation leave the revision unchanged.

```rust
use xynoos_autograd::{ParameterSet, ResourceLimits, Sgd, Tape};
use xynoos_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut parameters = ParameterSet::new(ResourceLimits::default())?;
    let p = parameters.insert(Tensor::from_data(&[], &[2.0], Default::default())?)?;
    let mut tape = Tape::new(&parameters)?;
    let x = tape.parameter(p)?;
    let loss = tape.multiply(x, x)?;
    tape.backward(loss)?;
    let gradients = tape.into_parameter_gradients()?;
    parameters.accumulate_gradients(gradients)?;
    Sgd::new(0.25)?.step(&mut parameters)?;
    assert_eq!(parameters.value(p)?.data(), &[1.0]);
    assert!(parameters.gradient(p)?.is_none());
    Ok(())
}
```

`Tape::backward_with_seed` takes an explicit finite binary32 scalar seed.
`Tape::gradient` exposes successful adjoints until the tape is consumed or
dropped. `Tape::value` can inspect a recorded forward result after sealing.
A `GradientBatch` has no public constructor or clone. Accumulation consumes it on
either outcome. Several batches computed at one revision may be accumulated.

## Operation rules

| Operation | Input requirements | Reverse edge contributions given upstream `g` |
| --- | --- | --- |
| `add` | Equal shapes | `g`, `g` |
| `subtract` | Equal shapes | `g`, `-g` |
| `multiply` | Equal shapes | `g*b`, `g*a` |
| `scale` | Finite constant scale `c` | `c*g` |
| `sum` | Any shape | `g` per element |
| `mean` | At least one element | `g/n` per element |
| `dot` | Equal rank-one shapes | `g*b`, `g*a` |
| `matmul` | Rank two and matching inner dimensions | `G Bᵀ`, `Aᵀ G` |
| `transpose` | Rank two | `gᵀ` |
| `mse` | Equal nonempty shapes and constant target input leaf | `g*2*(p-t)/n` for prediction |

Sum and dot of empty tensors give positive zero. Each matrix derivative cell
uses sequential binary64 products and sums in increasing inner-axis order and
converts once to binary32. The kernel reads the saved row-major operands directly.
Mean uses a sequential binary64 sum and division, and its derivative divides
in binary64 before converting. Add, subtract and the ordinary elementwise
products use binary32. The reverse visit order is descending tape index with
left edge before right edge. The first contribution owns an adjoint buffer.
Subsequent edge contributions and parameter-gradient additions sum in binary64
and convert once to binary32. Distinct snapshots of a parameter merge in reverse
insertion order. NaN, infinity and any finite-input operation that converts to
an infinite output or gradient are rejected at the relevant tape or optimizer
boundary.

SGD evaluates a binary64 product and subtraction, in that order, then converts
once to binary32 and tests finiteness. Every gradient must exist and match its
parameter. All candidates are staged before values are replaced. Success advances
the revision and clears gradients. An empty set is a no-op. Any error preserves
all existing values, gradients, and the revision.

## Budget limits and work units

| Limit | Ceiling |
| --- | ---: |
| Tensor numerical payload and retained numerical capacity | 64 MiB each |
| Simultaneous accounted session memory | 128 MiB |
| Tape values | 100,000 |
| Tape operations | 100,000 |
| Estimated scalar work per pass | 100,000,000 |
| Synthetic experiment steps | 10,000 |

Logical payload is the number of values times four bytes. Aggregate accounting
uses retained capacity for data and shape vectors, so logical payload is charged
once. The session belongs to a parameter set. Its shared allocation ledger uses
single-threaded bookkeeping and cannot mutate parameter or tape values. Multiple
sets have independent budgets. The ledger charges live stages even when several
tapes or batches coexist. Batch metadata overlaps all live tape buffers at the
point of transfer. Each moved adjoint carries its original charge. A gradient
sum stages a new tensor with both source tensors still live. A successful pure
move into a missing gradient requires no additional numerical buffer.

Forward work accumulates on a tape and covers snapshot scans and copies,
operation arithmetic, result initialization and finite scans, and metadata moves.
Work is retained for late forward failures after the attempt has passed the
metadata and work preflight. Backward preflights reachable edges and traverses
records in reverse, charging for reachable gradient writes, products, casts
and finite checks. Matrix multiplication estimates two arithmetic units per
multiply-add and also charges output work per cell. Two derivative products
have independently checked cell counts and inner dimensions. Transfer,
accumulation, zeroing, insertion, and SGD have separate work checks. Estimates
are conservative application-level units. Instruction counts and execution time
depend on the compiler and runtime environment.

See the implementation report for the independent gradient fixtures, synthetic
learning experiment, and verification commands.
