# TASK 006-D — First Bounded Training Experiment Proposal

**Status:** proposal only; no run is authorized or dispatched.
**Prerequisite:** owner approval of the experiment and separate authorization for its training run. TASK 006-R adds a native executable and manually dispatched CPU workflow, and local Linux process tests cover detached controller exit, stop, checkpoint, status, resume, and crash recovery. The workflow has not been dispatched; private checkpoint persistence for an actual experiment still needs an owner-approved artifact route.

## Research question

On a small project-authored educational systems-language dataset, do three predeclared learning-rate peaks produce finite native optimization and reduce fixed held-out combination cross-entropy after 24 updates?

The outcome is a narrow optimizer/learning-signal measurement. It does not measure cybersecurity capability, broad language generalization, or useful generated text.

## Dataset and evaluation freeze

Before dispatch, author and freeze a larger fixture than Objective A’s two training and two validation rows. Keep the planning target of 64 distinct training windows and use held-out combinations plus separate held-out template families. Use only project-authored educational material; acquire no external corpus. Retain the existing `byte-bpe-v1` tokenizer and vocabulary 292 without refitting. Commit an immutable manifest, exact split digests, provenance notes, deterministic ordering, generation seed, and tokenizer digest.

Normalize and check exact duplicates across all splits. Record the existing character-trigram Jaccard surface-overlap check and its inability to detect semantic paraphrase. Keep test-family examples inaccessible to learning-rate selection. The earlier one-batch validation plan is too small to select a learning rate reliably. The current `Trainer::validate_once` resets to the validation cursor and evaluates only one batch per event, and `xynoos-train` has no initial-baseline or locked-test evaluation command. A deterministic multi-batch validation path and separately reviewed bounded baseline/test evaluation are dispatch prerequisites. Do not tune on test-family examples. Neither one 32-token batch nor a future small fixture supports broad generalization claims.

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

The corrected training-lineage calculation includes current candidate precharges, periodic and final checkpoint reservations. The 24-update training path is `24 × (260,000,296 candidate microbatch + 32 reader + 27,434,220 update) = 6,898,429,152`. Four one-batch validations at updates 6, 12, 18 and 24 add `400,000,000`; five checkpoint reservations (four periodic plus final), each charged at the 16 MiB file cap, add `83,886,080`. The current one-batch training estimate is `7,382,315,232`, leaving `617,684,768` or 7.72% below the 8.0B cap. The final validation at update 24 is already one of the four cadence validations. The earlier 9.6% estimate omitted checkpoint work.

This arithmetic does not make one 32-token validation batch sufficient for rate selection. Each extra 32-token validation batch repeated at all four validation events adds up to `400,000,000` units under the current work accounting. Recalculate the cap for any multi-batch procedure before approval.

Initial unchanged-weight validation and the locked test-family evaluation are separate from each training lineage. Their current one-batch upper bound is `100,000,000` each, or 200M total. The present `xynoos-train` executable has neither evaluation operation; these are planning bounds and require a reviewed bounded evaluation utility.

Checkpoint equivalence has its own budget, separate from the three training runs. One uninterrupted and one interrupted/resumed 24-update trajectory, each with four one-batch validations, requires up to `14,814,962,112` units: 13,796,858,304 training/update; 800M validation; 11 × 16 MiB checkpoint reservations; and 2 × 16 MiB resume-load reservations. Set a separate 15B ceiling and require separate authorization. The three training lineages have a 24B ceiling; adding the 0.2B initial/test evaluation and 15B equivalence ceilings gives a 39.2B combined maximum. The current single-batch planned arithmetic totals `37,161,907,808` units: 3 × 7,382,315,232 training plus 200M baseline/test evaluation plus 14,814,962,112 equivalence. Do not omit equivalence or evaluation work from a sweep-wide authorization.

The 64 MiB run-owned and autograd caps are below the existing 128 MiB project ceiling and above the observed 42,439,416-byte charged whole-run peak with roughly 23 MiB slack. The native CLI divides its data-buffer and reader reservations between the simultaneously retained train and validation streams and rejects an over-budget source before constructing the model. The 4 MiB data allowance still requires preflight against the frozen fixture. The configuration must pass both training memory-plan validation and candidate/autograd preflight before dispatch; no project limit is increased to make an invalid plan fit.

The fresh bounded release sample measured 40.131 supervised tokens/s end-to-end. A simple 768-token scaling gives about 19.1 seconds for a run of this shape; the sample throughput already includes startup, validation, checkpointing and finalization, and it is based on only two updates. This is a feasibility estimate, not a sustained-throughput promise. The 120-second lineage limit allows bounded operational overhead. Stop checks are safe-point checks and synchronous operations can overrun. The manual workflow uses a 240-second process timeout with a 30-second kill grace inside a 10-minute job timeout; neither is a hard preemption guarantee.

## Expected artifacts

Each run replaces one canonical checkpoint of approximately 11 MB, derived from 914,400 FP32 parameters plus two FP32 moment inventories and format metadata; it must stay below the enforced 16 MiB file cap. Metrics JSONL, handoff, final/failure report, and lineage journal are expected below 100 KB/run. These are format-based estimates, not measurements of a future run. Preserve checksums and source/data/config identities for every artifact. The prepared Actions workflow uploads bounded logs, handoff, reports and integrity metadata for three days, but intentionally does not upload model weights/checkpoints and removes runner-local state. An owner-approved private checkpoint persistence route is required before a real experiment can be resumed after the workflow ends.

## Predeclared outcomes

The one-batch draft is not sufficient for a learning-rate selection study. Before this section can authorize a candidate choice, add and freeze deterministic multi-batch validation plus a separately bounded initial baseline and one locked test evaluation. The current CLI does not provide those evaluation paths. Do not dispatch TASK 006-D until those prerequisites and their work costs are closed.

**After those evaluation prerequisites are closed, success requires all of the following:**

- Every run remains finite and within its update, token, work, memory, and runtime limits.
- Parameters and both AdamW moment inventories change from initialization.
- At least one candidate reduces mean validation CE by at least 0.05 from its own unchanged-weight baseline after update 24.
- The selected validation candidate resumes from checkpoint with bitwise model/optimizer/schedule/cursor equivalence.
- The required metrics log, final report, checkpoint, and integrity digests validate.
- Rate selection uses validation only; the held-out test family is evaluated once after selection and does not change the choice.

**Failure includes:** any non-finite loss/gradient/state; exceeded or inconsistent budget; checkpoint, resume, or lineage mismatch; missing required log/report/checkpoint; no candidate meeting the validation criterion; or test-family examples used in selection. Stop, preserve artifacts, document the reason, and do not extend steps or tune against test data to rescue the result.

## Execution and compute decision

A three-run, 24-update CPU experiment is plausibly suitable for a bounded manually dispatched GitHub Actions workflow after the evaluation and checkpoint-persistence gates are closed. `.github/workflows/train-cpu.yml` is manual-only, validates the immutable configuration and exact dispatched source revision, checks the current pilot ceilings before model allocation, uses a bounded timeout and concurrency group, verifies available artifact integrity, retains only reports for three days, and cleans runner-local state. Foundation CI verifies this workflow file but does not dispatch it. Keep the actual experiment off pull-request CI; every failure must retain a nonzero job result, and a complete filesystem failure may prevent report persistence.

Optional external compute is justified only if a frozen objective and data coverage require more tokens than measured CPU runtime can deliver and an authorized bounded CPU trial demonstrates the bottleneck. Any external runtime needs separate owner authorization, authorized budget, verified official CLI execution, survival after CLI disconnection, persistent checkpoints, and cleanup evidence. This proposal does not authorize Colab, GPU, paid resources, a long-running job, or a model release.
