# TASK 006-D1 — Scientific Evaluation Infrastructure and Baseline

**Outcome:** evaluation infrastructure and bounded Objective A baseline implemented. The baseline is an unchanged-weight software control only; it is not a TASK 006-D training experiment or evidence of cybersecurity capability.

## Evaluator behavior

The training crate now provides configurable multi-batch evaluation with
deterministic token-weighted loss, token/batch/work/source-memory/position
limits, strict stage/split checks, source order checks for training and
validation, finite-loss checks, and explicit empty-supervision behavior.
Reports include data, source, model, tokenizer, order, evaluation-policy,
work, and memory identities. A bounded prefix is tagged with
`coverage_complete: false`; only a report with complete coverage represents
the entire split. Periodic trainer validation is bound to the validation split
and emits its coverage status. The training constructor rejects validation or
test data before model construction.

`evaluate_standard_dataset_split` is the preferred path for manifest-backed
datasets. It verifies the admitted dataset and fitted tokenizer identity and
constructs the deterministic split source. Generic `BatchSource` remains
available to software fixtures and makes its caller responsible for matching
the claimed split content. Initial baselines can evaluate train, validation,
and test on one immutable model without an optimizer. The test row is marked
as accessed; trainer admission prevents that data from entering training or
optimizer state.

Validation policy is part of custom run configuration and is persisted in the
versioned checkpoint run-policy record. The reader accepts the prior 125-byte
run-policy encoding and supplies its legacy default. A checkpoint round-trip
test verifies custom multi-batch limits survive save/load.

## Unchanged-weight Objective A baseline

The source was attested clean at `da99cba3016b7fe564c42e65d8d2babe44b3b98f`.
One model was initialized with seed 107 and reused without optimizer updates
for the three split evaluations.

| Identity or metric | Value |
|---|---|
| Dataset fixture | `task006-objective-a-v1` |
| Fixture SHA-256 (used as dataset identity in this software fixture) | `63050baf97f301c5ca6a3ad07a1a632bef77938b69f0317622738b5bbc00fa33` |
| Model | vocab 292, width 96, 8 layers, 4 heads, FFN 256, context 128 |
| Actual parameter count | 914,400 |
| Initial model state SHA-256 | `470fb79f5ec91c7e805115c19b90cb198586ff760b03ed7fafe4742f19f57a44` |
| Tokenizer | `byte-bpe-v1`, artifact SHA-256 `86aa40996d4a9bfe15491131bbe6fb1b04f5e325a50b23a3d7f0fa06bb56011c` |
| Evaluation limits | 2 batches, 64 consumed tokens, 250,000,000 forward work, 4 MiB source, 32 positions |

| Split | Batches | Supervised targets | Mean cross entropy | Perplexity | Forward work | Complete | Test access |
|---|---:|---:|---:|---:|---:|---|---|
| Train | 2 | 64 | 5.710966587 | 302.162994 | 133,789,512 | yes | no |
| Validation | 2 | 64 | 5.695174217 | 297.428606 | 133,789,512 | yes | no |
| Test | 2 | 64 | 5.690947771 | 296.174193 | 133,789,512 | yes | yes |

All losses were finite. The model-state digest was identical across the three
reports, which confirms that evaluation did not change the initial parameter
state. Test was read for this initial baseline and recorded as accessed; it
was not fed into training, optimizer state, or rate selection. Each split
report has its own deterministic order identity and evaluation digest.
The fixture source reservation was 864 bytes and measured peak reservation
was 1,296 bytes per split; these are synthetic-source accounting values, not
process RSS or a manifest-backed dataset memory estimate.

This fixture is not manifest-backed and has only two rows / 64 supervised
targets per split. Train and validation share template families; test has one
held-out family. Its scores validate report generation and deterministic
aggregation only. See the [split and tokenizer leakage audit](TASK-006-D1-SPLIT-LEAKAGE-AUDIT.md)
and the [evaluation principles](../EVALUATION.md).

## Acceptance limits

The result is insufficient to select a learning rate or claim generalization.
Objective A needs a larger frozen authored dataset with reviewed held-out
coverage before experiment readiness. Objective B requires separate data
licensing, provenance, and quality review. No external data was acquired.
No training workflow or compute provider was dispatched.
