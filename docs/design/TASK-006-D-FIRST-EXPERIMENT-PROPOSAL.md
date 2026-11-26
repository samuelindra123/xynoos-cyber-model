# TASK 006-D — First Bounded Training Experiment Proposal

**Status:** proposal only; no run is authorized or dispatched.
**Prerequisite:** owner approval of the experiment and separate authorization for its training run. Detached execution must not be used until the TASK 006-C executable/runtime readiness gap is closed and tested.

## Research question

On a small project-authored educational systems-language dataset, do three predeclared learning-rate peaks produce finite native optimization and reduce fixed held-out combination cross-entropy after 24 updates?

The outcome is a narrow optimizer/learning-signal measurement. It does not measure cybersecurity capability, broad language generalization, or useful generated text.

## Dataset and evaluation freeze

Before dispatch, author and freeze a larger fixture than Objective A’s two training and two validation rows. A planning target is 64 distinct training windows, one fixed 32-token held-out combination validation window, and one locked 32-token test-family window. The current trainer evaluates only the validation source’s fixed first batch per event; this proposal defines the online selection metric over that one predeclared batch and does not imply broad validation coverage. Use only project-authored educational material; acquire no external corpus. Retain the existing `byte-bpe-v1` tokenizer and vocabulary 292 without refitting. Commit an immutable manifest, exact split digests, provenance notes, deterministic ordering, generation seed, and tokenizer digest.

Normalize and check exact duplicates across all splits. Record the existing character-trigram Jaccard surface-overlap check and its inability to detect semantic paraphrase. Keep test-family examples inaccessible to learning-rate selection. Measure initial validation CE on unchanged initialized weights; select a rate only on the one fixed validation window; evaluate the selected run on the locked test-family window once after selection. If authoring the fixture does not support the target counts or objective, reduce the experiment and recompute all limits before approval.

## Frozen sweep configuration

Run three independent immutable run identities using identical model seed, initialization, dataset, ordering, validation procedure, and schedule shape. Change only peak learning rate.

| Parameter | Per-run value |
|---|---:|
| Peak learning-rate candidates | `1e-4`, `3e-4`, `1e-3` |
| Final learning rate | `1e-5` |
| Schedule | Linear warmup for 3 one-based updates, then linear decay through update 24 |
| Maximum optimizer updates | 24 |
| Microbatches per update | 1 |
| Input batch | `[1,32]` |
| Consumed-token cap | 768 occurrences |
| Supervised-target cap | 768 targets |
| Cumulative charged-work cap | 8,000,000,000 units |
| Run-owned memory cap | 64 MiB |
| Data/tokenizer allowance | 4 MiB |
| Autograd byte cap | At most 64 MiB, subject to exact `ResourceLimits` constructor and successful preflight |
| Cumulative active-runtime cap | 120 seconds |
| Per-attempt active-runtime cap | 90 seconds |
| Invoking-runner shutdown grace | 30 seconds |
| Checkpoint cadence | Every 6 completed updates, plus finalization |
| Validation cadence | Every 6 completed updates after the due checkpoint |

The work cap is based on current precharges, not the hard maximum: 24 × (260,000,296 candidate microbatch + 32 reader + 27,434,220 update) + 4 × 100,000,000 validation = 7,298,429,152. The proposed 8.0B leaves about 9.6% margin for this explicit bounded configuration. Three independent roots imply a 24.0B sweep-wide ceiling. This should be recalculated if validation batches, work accounting, or update count changes.

The 64 MiB run-owned and autograd caps are below the existing 128 MiB project ceiling and above the observed 42,439,416-byte charged whole-run peak with roughly 23 MiB slack. The 4 MiB data allowance matches the measured fixture setup. The configuration must pass both training memory-plan validation and candidate/autograd preflight before dispatch; no cap is increased to make an invalid plan fit.

At measured bounded throughput near 40 supervised tokens/s, 768 tokens imply roughly 19 seconds of update work/run before setup, validation, checkpointing, and finalization. The 120-second lineage limit allows bounded operational overhead; it is not a sustained-throughput promise. Stop checks are safe-point checks and synchronous operations can overrun. The 30-second runner grace is for final checkpoint/report and process exit, not a hard kill deadline.

## Expected artifacts

Each run replaces one canonical checkpoint of approximately 11 MB, derived from 914,400 FP32 parameters plus two FP32 moment inventories and format metadata; it must stay below the enforced 16 MiB file cap. Metrics JSONL, handoff, final report, and lineage journal are expected below 100 KB/run. Budget under 12 MB per run root and 36 MB for the three-run sweep. These are format-based estimates, not measurements of a future run. Preserve checksums and source/data/config identities for every artifact.

## Predeclared outcomes

**Success requires all of the following:**

- Every run remains finite and within its update, token, work, memory, and runtime limits.
- Parameters and both AdamW moment inventories change from initialization.
- At least one candidate reduces mean validation CE by at least 0.05 from its own unchanged-weight baseline after update 24.
- The selected validation candidate resumes from checkpoint with bitwise model/optimizer/schedule/cursor equivalence.
- The required metrics log, final report, checkpoint, and integrity digests validate.
- Rate selection uses validation only; the held-out test family is evaluated once after selection and does not change the choice.

**Failure includes:** any non-finite loss/gradient/state; exceeded or inconsistent budget; checkpoint, resume, or lineage mismatch; missing required log/report/checkpoint; no candidate meeting the validation criterion; or test-family examples used in selection. Stop, preserve artifacts, document the reason, and do not extend steps or tune against test data to rescue the result.

## Execution and compute decision

A three-run, 24-update CPU experiment is plausibly suitable for a bounded manually dispatched GitHub Actions workflow. The estimate uses the measured two-update Objective A observation and should be checked on the intended runner before owner approval. Keep it off pull-request CI; define an explicit timeout, concurrency group, redundant-run cancellation, bounded artifact retention, and a manual dispatch input that requires the frozen run identity. The workflow must validate all caps and identities before allocation and return a clear nonzero status on any failure.

Optional external compute is justified only if a frozen objective and data coverage require more tokens than measured CPU runtime can deliver and an authorized bounded CPU trial demonstrates the bottleneck. Any external runtime needs separate owner authorization, authorized budget, verified official CLI execution, survival after CLI disconnection, persistent checkpoints, and cleanup evidence. This proposal does not authorize Colab, GPU, paid resources, a long-running job, or a model release.
