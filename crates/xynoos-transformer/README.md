# Native Transformer reference

`Config` validates the dimensions and parameter-count arithmetic.
`Transformer::new` binds the model vocabulary to a `Tokenizer` and generates
parameters in the ledger from a recorded SplitMix64 seed. `inventory` exposes
the parameter names, row-major shapes, and revisioned parameter identities.
`Transformer::from_coordinates` is the explicit-coordinate constructor used
by the mathematical fixture.

`Transformer::evaluate_batch` accepts a TASK 004 `Batch`.
`Transformer::evaluate` accepts the same shapes as borrowed IDs and masks.
The model checks all IDs, including masked targets, as well as shift consistency
within each input-valid prefix. A final valid input may target PAD with loss
mask 0. Raw batch callers supply the per-document sample boundaries.
`Evaluation` exposes borrowed `[B,T,V]` logits and a scalar supervised mean loss.
`Evaluation::gradients` consumes the forward pass to return the native
single-use `GradientBatch`. Dropping an evaluation releases its tape charges.
`Transformer::accumulate_gradients` applies the native transactional gradient
accumulation API.

The model runs FP32 values and parameters with sequential FP64 reductions.
One embedding parameter snapshot is shared by gather and transposed vocabulary
projection. The explicit attention kernel indexes contiguous heads within
`[B,T,D]` and saves finite binary64 probabilities. A masked residual kernel
and RMSNorm both return exact zero for invalid rows.

See the [implementation report](../../docs/reports/TASK-005-B-IMPLEMENTATION-REPORT.md)
for equations and validation, the
[resource report](../../docs/reports/TASK-005-B-RESOURCE-MEASUREMENT-REPORT.md)
for phase budgets, and [traceability](../../docs/reports/TASK-005-B-REQUIREMENT-TRACEABILITY.md)
for the scoped specification interpretations.
