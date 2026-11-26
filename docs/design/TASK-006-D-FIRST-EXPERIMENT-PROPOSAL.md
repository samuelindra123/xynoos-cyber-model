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

The earlier three-learning-rate sweep is retired. The current D1 candidate
is Objective A v3: 60 project-authored rows (24 train, 18 validation, 18
public test) across parity, integer comparison, and modular progression.
Its dataset ID is
`6b8d4b140c8d20636b206cb6259a1789a2734f5a13d7fd234e3d39aad3037c87`, its
manifest SHA-256 is
`db2577c6ed7f0f74c69b643dc9d74683c76084ddf8dc21bddc0ce072c56562b8`, and
the 32-merge train-only tokenizer SHA-256 is
`9ac03af34eaa496a54dc7721f83107ecdd14c26c643fb4c51d34de9afe77d6d0`.

The v3 audit found no normalized exact, task-input, template, or cross-split
character-trigram Jaccard overlap at or above 0.82 (maximum 0.461538). The
corpus still has one authored source/date, a small task inventory, no human
review, and a public test that was read for the unchanged-weight baseline.
It cannot support a hidden-test claim or a broad generalization claim. Ten
answer spans begin inside merged BPE tokens, so answer-bearing loss includes
some prompt bytes. The v3 baseline and limitations are recorded in the
[dataset report](https://github.com/samuelindra123/xynoos-cyber-model/blob/f122f4e283b06d2cd20dd544fc0b18af4c8b6c98/docs/reports/TASK-006-D1-OBJECTIVE-A-V3-DATASET.md); it
establishes evaluation behavior only, not post-update learning.

## Evaluation policy

The native runtime supports configurable deterministic multi-batch evaluation
with token-weighted aggregation, work and source-memory caps, explicit empty
supervision, model-state hashing, and an explicit test-access marker. Periodic
validation accepts only the validation split; trainer construction accepts
only the training split. The manifest-backed evaluator binds evaluation to the
validated dataset and fitted tokenizer.

Initial-weight baseline reports must cover train, validation, and public-test
batches before training. The v3 test was read in the unchanged-weight
baseline; it is not locked and cannot support a hidden post-selection claim.
Test access is tagged and validation/test are rejected from training; keep
test out of training and schedule choice. A future locked evaluation needs
a reviewed test-access policy and a hidden set not used during selection.
A report marks a capped prefix with `coverage_complete: false`; partial-split
reports cannot be used as full-split baselines or selection metrics.

The checked-in two-batch validation pass used 64 targets on the tiny v1
fixture. The following 24-update estimate is a historical fixture ceiling,
not an admitted plan or transferable v3 estimate. It assumes one validation
batch per event, recalculated with two at updates 6, 12, 18, and 24:

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
existing measured Objective A v1 software fixture used 133,789,512 forward-work
units per two-batch split evaluation. This is a measurement of the tiny fixture,
not a cost estimate for a larger dataset. The later v3 unchanged-weight baseline
used 3,124,414,056 forward-work units across its five evaluator passes.

The v3 trainer-facing stream now emits one independently padded `[1,32]` batch
per train document, with 24 batches and 197 supervised targets; no document
is packed across another. A deterministic two-update software control
completed with 19 consumed/supervised train tokens, 18 validation tokens
across two one-batch partial validation events, and no test access. It charged
825,200,872 cumulative work units and took 17.2875 seconds active in local
debug, with per-update checkpoints. A simple twelve-fold projection for 24
such updates is 9,902,410,464 work units, above the 8-billion ceiling. This is
not a measured full run: it omits the initial baseline and complete validation
coverage, so it cannot establish a feasible 24-update budget. The two-update
control is software verification, not a learning experiment.

Do not claim that 24 updates or 768 targets are sufficient. The earlier
three-learning-rate sweep and v2 ceiling calculation are retired. No v3
schedule is selected: the corpus still needs independent review, a locked
held-out evaluation protocol, and a complete work/runtime calculation that
preserves full validation coverage without exceeding the ceiling. Do not
reduce held-out coverage or weaken the ceiling to force admission.

## Readiness gates still open

- Have an independent reviewer assess v3 task validity, tokenizer provenance,
  held-out family coverage, leakage limits, and test-access policy; prepare a
  reviewed alternative if v3 is not adequate for a narrow engineering control.
- Independently review the supported v3 trainer stream and decide whether the
  corpus is adequate for the narrow learning question. Derive a full-coverage
  experiment budget from actual measurements; the current 24-update partial
  cadence projection already exceeds the 8-billion work limit.
- Keep the current public test out of training and schedule selection; create
  and review a locked or otherwise inaccessible held-out evaluation set before
  making any generalization claim.
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
the [TASK 006-D3 compute assessment](../reports/TASK-006-D3-COMPUTE-READINESS.md).

The complete evidence and remaining gates are tracked in the [TASK 006-E
first-training readiness package](../reports/TASK-006-E-FIRST-TRAINING-READINESS.md).

This document is not an experiment configuration and does not authorize a run.
