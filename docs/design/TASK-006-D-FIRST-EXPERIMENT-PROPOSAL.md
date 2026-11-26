# TASK 006-E — First Training Readiness Assessment

**Status:** preparation only. No training run, workflow, or external compute is authorized or dispatched.

## Intended experiment

The first experiment is an Objective A engineering control: a fixed, randomly
initialized native Transformer, one predeclared optimizer/schedule, and
project-authored deterministic examples. It can establish that this training
path changes model and optimizer state and measure a narrow held-out learning
signal. It cannot establish cybersecurity intelligence, broad technical
generalization, or safe model behavior. Objective B remains a separate future
corpus and evaluation effort.

The earlier three-learning-rate sweep is retired. The current Objective A
fixture has two training rows, two validation rows, and two test rows, only 64
supervised targets per split. The audit found no normalized exact duplicates,
but train and validation reuse the same template families; the maximum
cross-split character-trigram Jaccard is 0.717391. The test set has a single
template family. Those observations make the fixture a useful software
control, not a defensible rate-selection dataset. The full details and limits
are in the [split and tokenizer audit](../reports/TASK-006-D1-SPLIT-LEAKAGE-AUDIT.md).

## Evaluation policy

The native runtime supports configurable deterministic multi-batch evaluation
with token-weighted aggregation, work and source-memory caps, explicit empty
supervision, model-state hashing, and an explicit test-access marker. Periodic
validation accepts only the validation split; trainer construction accepts
only the training split. The manifest-backed evaluator binds evaluation to the
validated dataset and fitted tokenizer. Initial-weight baseline reports must
cover train, validation, and test before training; the locked test must be
evaluated after selection and must not influence training or selection.

The checked-in two-batch validation pass uses 64 supervised targets. The
earlier 24-update work estimate was based on one batch per validation event.
Recalculated with two batches at updates 6, 12, 18, and 24, the current upper
bound is:

| Component | Charged work units |
|---|---:|
| 24 candidate microbatches, reader reservations, and optimizer updates | 6,898,429,152 |
| Four two-batch validation passes, including bounded reader lookahead | 800,000,384 |
| Four periodic plus one final checkpoint at the 16 MiB file limit | 83,886,080 |
| **Total** | **7,782,315,616** |
| **Headroom under the current 8,000,000,000 workflow ceiling** | **217,684,384 (2.72%)** |

One additional validation batch at every cadence event adds 400,000,128 work
units and exceeds that ceiling. Initial baseline and final locked-test passes
are separate evaluation operations and need their own explicit limits. The
existing measured Objective A software fixture used 133,789,512 forward-work
units per two-batch split evaluation. This is a measurement of the tiny fixture,
not a cost estimate for a larger dataset.

The 24-update plan consumes 768 supervised training targets and cannot cover
64 distinct 32-token training windows once. The current 8B work ceiling cannot
be assumed to support a larger coverage target; the plan must be recomputed
from the frozen dataset, validation cadence, and actual memory/work counters.
Do not claim that 24 updates or 768 targets are sufficient. Do not reduce
held-out coverage to fit a workflow ceiling. No learning-rate sweep or
candidate-selection plan is accepted at this stage.

## Readiness gates still open

- Author a larger, manifest-backed Objective A dataset and freeze its
  provenance, splits, deterministic order, tokenizer fit, and leakage report.
- Independently review that dataset and the evaluation policy, including a
  meaningful held-out combination or family target.
- Recompute a single-run work, memory, runtime, and evaluation budget from the
  actual dataset and the supported CPU measurements.
- Select and verify private checkpoint storage. This repository is public, so
  its Actions artifacts are not a private checkpoint destination.
- Independently review the R2/C3 code and evidence; C3 currently documents a
  separate verification pass with the separate-review gate still pending.
- Verify detached execution, persistence, restart, and cleanup for the chosen
  provider. Local process tests do not establish remote runner or Colab
  survival.
- Obtain separate owner authorization before any training dispatch or
  billable compute use.

## Provider decision

Use GitHub Actions for bounded CPU tests, numerical oracles, and regression.
The current public Ubuntu 24.04 runner workflow is bounded to 10 minutes and
does not upload model checkpoints. A training job is not selected because the
scientific data and private persistence gates are open, not because a GPU is
available. The installed Colab CLI was inspected without creating a runtime;
detached execution, artifact persistence, and cleanup remain unverified. See
the [TASK 006-D3 compute assessment](../reports/TASK-006-D3-COMPUTE-ASSESSMENT.md).

This document is not an experiment configuration and does not authorize a run.
