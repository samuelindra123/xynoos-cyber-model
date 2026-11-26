# TASK 006-C — Independent Native Training Engine Verification

**Verdict: CONDITIONAL PASS — SPECIFIC REMEDIATION REQUIRED**
**Scope:** Independent audit of the TASK 006-B head; no production behavior, architecture, tokenizer, limits, tolerances, or fixtures were changed.

## A. Final verdict

The native optimizer, bounded candidate path, checkpoint codec, resume path, and tiny Objective A learning control pass the inspected contracts and executed checks at the pinned implementation head. Independent native tests added two mathematical controls and one checksum-valid semantic-checkpoint rejection test. Debug and optimized profiles both pass the new optimizer controls; the optimized Objective A control reproduces the frozen loss values and shows nonzero parameter and moment changes.

Owner acceptance remains conditional for two concrete reasons:

1. The crate has a trainer library and test integration, but no independently launchable training binary or dispatch workflow. Process-level detached execution, survival after an agent/CLI disconnect, process signal handling, and cleanup therefore remain unverified.
2. The per-microbatch JSONL write result is discarded in `Trainer::run_updates`; a transient write/flush failure can silently omit that record while training continues. Logging failure injection is not available at the `Trainer` boundary.

The public `BatchSource` memory claim and missing-journal fallback are additional trust and recovery limits documented below. None is hidden behind a PASS verdict. No sustained run was dispatched and no external compute was provisioned.

## B. Pinned evidence and GitHub workflow

- Accepted main baseline: `32eca2b900c4f7a88294f7a24a5dfae0604ae798`; it is an ancestor of the implementation head.
- TASK 006-A design: `26777fa95383600c003ff094216d3d1c310b051c`.
- TASK 006-B implementation: `ae9e038af2b04a2fda4a4c624079a1ec82fd086c`.
- Canonical checkout began clean at the exact implementation head with one physical working tree. Audit branch: `audit/task-006-c-native-training-verification`.
- TASK 006-B PR #19 is open at the pinned head. Its hosted run `45662756847`, job `139376689147`, completed successfully.
- Audit branch: `audit/task-006-c-native-training-verification`.
- Audit PR: [draft PR #20 targeting `feat/task-006-b-native-training-engine`](https://github.com/samuelindra123/xynoos-cyber-model/pull/20). The final exact head and hosted CI job are recorded in the PR checks and final handoff.

Read `AGENTS.md`, `docs/COMPUTE_POLICY.md`, `docs/MODEL_VERSIONING.md`, `docs/design/EFFICIENCY-FIRST-MODEL-ENGINEERING.md`, TASK 006-A specification/numerical/checkpoint/data research, TASK 006-B handoff/report, and the checkpoint v1 format before reaching conclusions.

## C. Independent AdamW mathematics

The frozen fixture has 33 records and SHA-256 `f1839d7c12a0c9b3b0e75b73ec3b7e53b1eb1c67191c64e6053eb9a199244e61`. Removing exactly the two TASK 006-B-added update-order and tied-decay controls reconstructs the original 31 records byte-for-byte at SHA-256 `dc191b568d3134a344d056619422b7fa383b0a7554b934ebcce17771de73bcc2`. `python3 tools/adamw_reference.py --check` verifies all 33 frozen records; the native Rust fixture test checks the frozen state outputs.

Auditor-authored test `crates/xynoos-autograd/tests/task006c_independent.rs` drives native AdamW with actual `Tape`-generated gradients and independently calculated FP64 intermediates. It performs three updates: nonzero gradients, a zero-gradient update after nonzero state, then another nonzero update. It compares FP32 parameter and both moment bits plus FP64 bias-power bits after every update. Both debug and release agree bit-for-bit, including beta values `f32::from_bits(0x3f7fffff)` and epsilon `1e-30`. It separately exercises rank-2 decay and rank-1 gain exclusion.

Negative controls distinguish incorrect behavior: the frozen tied-embedding double-decay value differs from the correct value by `0.0475000143`; the auditor weighted-mean control differs from equal averaging by `0.5` (`2.0` correct versus `2.5` wrong). The additional test derives its expected values independently and never reads native results to make expectations.

The production source performs each widened multiply, addition, correction, square root, decay, and narrowing in the specified order. No explicit FMA, fast-math, `target-cpu`, or `target-feature` setting was found in the repository build configuration. Debug/release oracle bits agree. This verifies source ordering and the configured reference builds; it is not a disassembly audit for every possible downstream compiler/target.

## D. Optimizer transaction and rollback

`AdamW::step` validates owner, inventory fingerprint, expected revision, complete gradients, finite parameters/moments, nonnegative second moments, checked counters, and work before allocating three complete staged tables. It publishes parameters, both moments, gradient clearing, revision, powers, and step only after staging succeeds, with no fallible call after publication begins.

Existing autograd checks cover missing gradients; foreign and stale sets; non-finite input/gradient; invalid learning rate/config/decay product; candidate overflow; step/revision exhaustion; and injected staging allocation failure. They compare committed and gradient state and verify ledger-charge cleanup. One negative second-moment restore control is present. SGD uses its original separate transactional path and existing regression checks remain green.

Audit limit: allocation-fault injection is through internal unit tests, not a public allocation provider. The inspected commit path is all-or-nothing; no production behavior was modified to add fault hooks.

## E. Weighted accumulation and clipping

`WeightedGradientAccumulator` binds owner, inventory, and revision; rejects foreign/stale batches; accumulates supervised-count-weighted FP32 means into FP64 sums in inventory/row-major order; divides once; narrows once; computes one global norm; and clips only after normalization. Empty windows reject `finish`, and all-unsupervised trainer windows do not commit an update. The existing suite covers extreme finite gradients, exact clipping boundaries, empty supervision, stale/foreign inputs, overflow, failed accumulation, and cleanup.

The new native test uses gradients 1 and 4 with supervised counts 2 and 1: it observes 2.0, while equal averaging yields 2.5 (absolute disagreement 0.5). Trainer code clips the finished whole-window gradient; there is no per-microbatch clipping path.

## F. Schedule

The schedule is one-based and computed in FP64 from widened FP32 configuration values, then narrowed once. Existing tests cover zero warmup, first rate, warmup boundary, decay points, final positive rate, completed-step boundary, invalid configurations, and the AdamW decay-product guard. Checkpoint v1 persists total/warmup/rates and completed steps; resume requires exact configuration equality and the completed step to equal optimizer step. The next rate is deterministically derived from that persisted step, so schedule state cannot be silently substituted or reset under the supported resume path.

## G. Candidate training-policy confinement

The fixed model is `914,400` parameters in 74 tensors: 8 blocks, width 96, 4 heads, FFN 256, vocabulary 292, context 128, tied embedding/output, and training microbatch `[1,32]`. `TrainingConfig` fixes `Config::RESEARCH_CANDIDATE`; the transformer path checks the fixed batch and candidate inventory before recording. The backward permit is distinct from repeated-training policy. Ordinary work remains capped at 100M and candidate backward at 160M; tensor max remains 64 MiB, autograd ledger 128 MiB, run-owned cap 128 MiB maximum, and existing operation/value caps remain applied.

Candidate policy construction requires positive finite runtime/token/work/memory/update limits and caps updates at 10,000. Admission checks exact shape, candidate inventory marker and all tensor shapes, owner/inventory binding, caller limits, checked cumulative token/work/runtime state, and a single authorized update window. Lower `ResourceLimits` remain active; candidate backward only selects the separate candidate work ceiling when the candidate marker is present and the ordinary ceiling is the default. Existing tests exercise generic same-shape inventories, wrong batch/config, lower work ceilings, allocation failures, and finite bounded controls. No limit was raised.

The policy is a Rust library authorization boundary, not a security boundary against a caller that can directly construct an explicitly candidate-marked `ParameterSet` and supply arbitrary values. Transformer construction and trainer APIs keep the production run inventory fixed; the lower-level API should continue to be treated as trusted in-process research code.

## H. Whole-run resources and data admission

For the bounded release Objective A run, measured report values were:

| Measure | Bytes | Meaning |
|---|---:|---|
| Whole-run charged peak | 42,439,416 | Plan ledger: persistent reservation plus peak temporary phase reservation |
| Whole-run charged current after finalization | 22,744,448 | Persistent run-owned reservation remaining |
| Autograd ledger peak | 30,682,296 | Application charges for parameters/tapes/batches/staging |
| Autograd ledger current after finalization | 18,314,232 | Persistent parameter/optimizer state and current autograd allocations |
| Process `VmHWM` (isolated release Objective A) | 34,889,728 | OS process high-water; includes runtime/test process overhead |
| Process `VmHWM` (isolated release resume test) | 57,335,808 | OS process high-water across the resume integration process |

These are separate measures; a reservation is not RSS. The memory plan reserves model parameters and optimizer moments, FP64 accumulator, transfer, tape/activations, three-table AdamW staging, checkpoint buffer, resume staging, validation tape, data allowance, and runtime metadata across compatible child phases. Reservations drop at phase exit; repeated-step tests return to persistent baselines. Actual current run charges remain below the 128 MiB run cap and autograd below its 128 MiB cap.

`BatchStream` and `VecBatchSource` calculate retained bytes, and tokenizer retained bytes are checked. `BatchSource` is public and its `reserved_memory_bytes()` value is an implementer-supplied claim. A custom source can lie or omit allocations; the trainer cannot independently inspect its retained capacity. Standard source implementations are checked, arbitrary custom sources are trusted. Dataset/tokenizer objects are also prepared by the caller before trainer construction; only their admitted retained-size claims are covered. RSS, allocator metadata, fragmentation, and OS overcommit are not bounded by these application ledgers.

## I. Checkpoint v1 format and integrity

The codec enforces a 16 MiB file limit, exact 40-byte header, version/endianness/precision markers, 10 ordered sections, strictly increasing required fields, checked payload lengths, terminal SHA-256 section and no trailing data. It validates architecture, ordered 74-entry inventory, one tied embedding identity, model/moment counts and shapes, finite FP32 values, nonnegative second moments, FP64 bias-power progression, schedule/optimizer step agreement, data/tokenizer/source/config identities, cursor/next-batch identity, counters, resource limits, and zero partial accumulation. Decode uses streaming framing/integrity scan followed by private typed staging.

Existing tests cover round-trip, checksum damage, unknown section, duplicate field, truncation, oversize, symlink path, and typed loading. The independent test additionally changes a second moment to `-1.0`, recomputes the correct file checksum, and verifies semantic rejection as `non_finite_or_negative_moment`. Loading returns detached checkpoint data; `Trainer::resume` only constructs a new model after successful codec validation. No mutable caller model is exposed to partial decode.

Coverage limitation: not every malformed semantic combination in the format document has its own test vector (including all alternate/reordered sections, all identity mismatches, non-finite fields, counter mutations, trailing data, and partial accumulation). Static parser checks cover those classes, but a broader independently mutated checkpoint matrix remains useful. No format defect was confirmed.

## J. Checkpoint publication and durability

The writer validates the root and target, creates a new sibling with `create_new`, streams through a bounded 64 KiB buffer, syncs the file, atomically renames in the same directory, then syncs the parent directory. Existing tests inject pre-rename and post-rename failures: before rename the prior checkpoint loads unchanged; after rename `PublicationDurabilityUncertain` is returned and the new sequence is already visible. Temporary sibling cleanup is attempted on failure.

Supported assumptions are a local POSIX filesystem with same-directory atomic rename and file/directory sync support. Symlink roots/components/artifacts are rejected on inspection. TOCTOU races between inspection and use, remote filesystems, controller loss, and power-loss behavior beyond sync requests are explicitly unsupported. A successful rename replaces the old artifact even if the later directory sync fails.

## K. Lineage journal and cumulative budgets

The journal is fixed-size, configuration-bound, SHA-256 checked, atomically replaced and synced before admission. A lock prevents simultaneous live attempts. Work/tokens/attempted updates are precharged before operations; checkpoint counters reconcile by maxima, so restarting with the intact journal or loading an older checkpoint does not refund work. Interrupted partial windows restore the committed cursor and replay; scientific committed counters remain in the checkpoint/model transaction while attempted/discarded input remains in the journal. The journal reserves the whole remaining per-attempt runtime; abrupt exit charges the full outstanding reservation on restart, so unrecorded elapsed time is overcharged rather than refunded.

The journal checksum is not a signature. Deliberately deleting/replacing journal state inside the trusted artifact directory can reset precharges newer than the last checkpoint: `RunJournal::open` accepts a missing file and seeds from checkpoint counters. Replacing bytes and recomputing its unkeyed hash is also not adversarially prevented. This is outside the documented trusted-artifact-directory threat model and is not claimed as tamper resistance, but missing-state recovery is a fail-open limitation to fix before treating the budget as resilient to artifact loss. A prior checkpoint with an intact journal does not refund resources.

Runtime is checked at safe points. A stop cannot interrupt a synchronous forward/backward or blocked filesystem operation; final checkpoint/report work can extend time beyond a configured cap. The runtime guarantee is bounded safe-point stopping plus persisted reservation accounting, not hard real-time preemption.

## L. Independent resume equivalence

The integration suite compares uninterrupted with destroy/load/resume parameters, both Adam moment inventories, bias powers, optimizer step, derived next schedule rate, next batch identity, cursor, committed scientific counters and deterministic loss/clip/update metrics. The partial accumulation interruption test confirms 32 consumed inputs are marked discarded, no scientific update is committed, cursor remains at the prior committed boundary, and replay yields the same final model as uninterrupted execution without double-counting committed scientific tokens. The user-selected additional native independent optimizer case also interrupts the update sequence via a zero-gradient middle update and checks bitwise state after every boundary.

Elapsed runtime and RSS are intentionally excluded from bitwise equivalence. Attempted/replayed input and cumulative work remain charged in the intact lineage journal; those are not expected to equal uninterrupted-path attempted counts when an interruption caused replay.

## M. Objective A scientific evidence

The project-authored fixture digest matches `63050baf97f301c5ca6a3ad07a1a632bef77938b69f0317622738b5bbc00fa33`; deterministic regeneration passes. It uses existing `byte-bpe-v1` artifact and vocabulary 292 without refitting, fixed seed `0x006B`, fixed row order, two training and two validation educational patterns, and two separate test-family rows. Normalized exact duplicate count is zero. Validation combinations and test template families are held out. Character-trigram Jaccard at 0.82 is only surface-overlap screening and misses semantic paraphrases; it proves no reasoning independence.

The isolated release test reproduces held-out mean CE `5.695174217` before and `5.554496050` after two updates (delta `-0.140678167`) over 64 supervised training tokens. It also checks nonzero tied embedding parameter bits and both moment arrays. Validation rows are evaluated at fixed configured points; test-family rows are not used for tuning. The unchanged-weight baseline is the measured initial held-out CE. This is a tiny optimizer/learning signal only; no cybersecurity capability, broad language generalization, or useful text generation is established.

## N. Runtime and detached execution

The trainer library owns JSONL events, experiment/config/data/tokenizer identity, counters, stop-file polling, per-attempt and lineage budgets, numerical failure classification, checkpoint writes, final JSON report, handoff metadata, and cleanup attempts. Stop requests are checked at bounded safe points. Checkpoint and report writes are atomic under the documented filesystem assumptions. Failure handling attempts a failure event and final report but ignores failure-event/log-flush errors; the report write itself propagates an error to the caller when it fails.

There is no `src/bin`, `main.rs`, standalone training executable, or repository dispatch workflow. Tests create a `Trainer` in-process and exercise orderly finish/resume. They do not demonstrate independent process launch, process signal delivery, persistence after agent/CLI disconnect, or process cleanup by a detached runtime. Therefore detached execution readiness is **not established**, and no detached/Colab claim is made.

**Focused handoff:** add a narrowly scoped, manually dispatched executable with immutable run identity/config inputs, startup validation, OS stop handling mapped to the stop token, bounded startup confirmation and process exit status. Add writer-failure injection and propagate all required metric write errors, with deterministic report behavior under log/checkpoint failures. Exercise process termination/restart, checkpoint persistence, runtime cleanup and artifact integrity before any detached dispatch.

## O. CPU feasibility

New isolated optimized-profile measurements on the bounded two-update fixture:

- Full update: 2 samples, mean `412.685 ms`, sample SD `13.035 ms`, range `403.467–421.902 ms`.
- End-to-end supervised throughput: `39.977 tokens/s` for 64 tokens across two updates; not a sustained rate.
- Forward: 2 samples, mean `157.187 ms`; backward: mean `175.303 ms`; optimizer staging/commit: mean `58.337 ms`.
- Validation: 2 samples, mean `150.063 ms`; checkpoint writes: 3 samples, mean `31.391 ms`.
- Charged whole-run peak/current, autograd peak/current and `VmHWM` are listed in section H.
- Isolated resume process `VmHWM`: `57,335,808` bytes; checkpoint codec load 3 samples mean `65.528 ms`, full trainer resume load 1 sample `85.749 ms`.

`Instant` sample collection uses Welford mean/sample variance; throughput divides committed supervised tokens by active attempt runtime, including initialization, validation, checkpointing and finalization. Two update samples are highly bounded observations, not sustained-training estimates. A local debug run showed a process high-water of approximately 122–131 MB; it is a different build profile and not mixed into the optimized-profile table.

## P. Preserved regression coverage

Both Rust profiles completed the repository workspace suite locally. The prior 135 active checks are preserved; three new TASK 006-C Rust checks bring the total to 138 active checks per profile (109 preserved baseline, one earlier independent control, 25 TASK 006-B checks, and these three new controls). Five hosted-only ignored checks remain explicit in CI, which invokes each of the five. Existing oracle fixture bytes, Objective A digest, TASK 001–005 contracts, task-specific trajectory checks and candidate resource policy are unchanged. CI still runs formatting, Clippy, repository hygiene, independent fixture scripts, debug/release suites, hosted-only bounded checks and documentation build.

## Q. Confirmed defects and limitations

| ID | Severity | Evidence | Effect and disposition |
|---|---|---|---|
| F-01 | P2 readiness gap | Training crate exports a library `Trainer`; repository has no binary or dispatch workflow | Detached process execution cannot be verified. Remediate before an authorized detached job. |
| F-02 | P2 logging defect | `trainer.rs` discards the per-microbatch `write_event` result | A failed event write/flush may be silently omitted while training continues. Add fault injection and propagate/stop deterministically. No production fix was made in this audit. |
| F-03 | P2 recovery limitation | Missing journal seeds state from checkpoint counters | Loss/deletion of journal after precharges can reset cumulative budgets to the checkpoint boundary. Trusted artifact directory is assumed; fail closed on missing resume journal or add durable lineage binding. |
| L-01 | Trust boundary | Public `BatchSource::reserved_memory_bytes()` is caller-provided | Built-in sources are counted; arbitrary implementations may underreport. Do not treat API cap as enforcement against untrusted source code. |
| L-02 | Filesystem boundary | Path validation has TOCTOU gaps and local-POSIX-only publication | Remote FS, hostile path races, power-loss beyond sync contract are unsupported. |
| L-03 | Test breadth | One hashed semantic checkpoint mutation was added; not every format semantic has an independent mutation fixture | Static validation inspected; broad malformed matrix remains useful. |
| L-04 | Time boundary | Stop/runtime checks at safe points | Synchronous kernels and blocked I/O may overshoot; not hard real-time interruption. |

No numerical disagreement, state corruption, optimizer rollback failure, old-checkpoint budget refund with intact journal, or resume-equivalence defect was confirmed. No project-specific numerical issue required model escalation.

## R. TASK 006-D first experimental decision

Do not dispatch until separately authorized. A useful first experiment needs a larger frozen authored fixture than Objective A’s two train/two validation rows. Proposed question: **On project-authored educational systems-language patterns, do three predeclared peak learning rates produce finite optimization and a lower fixed held-out combination CE after 24 updates?** This measures a narrow learning signal, not cybersecurity capability.

| Setting | Proposed bound |
|---|---|
| Data | Prepare a project-authored frozen fixture (for example 64 distinct train windows, one fixed 32-token validation combination window and one locked 32-token test-family window); retain existing byte-bpe-v1, vocab 292; no external corpus. The trainer evaluates only the validation source’s fixed first batch at each event. If authoring does not support these counts, reduce scope and recalculate before dispatch. |
| Split/control | Deterministic seed/order; normalized exact-duplicate separation; document trigram limitations; test family never used for rate selection. Record initial unchanged-weight validation CE. |
| Sweep | Three independent runs at peak rates `1e-4`, `3e-4`, `1e-3`; final rate `1e-5`, 3 one-based warmup updates, linear decay across 24 updates. One microbatch/update. |
| Update/data caps | 24 updates/run; at most 24 microbatches; consumed ≤768 occurrences and supervised ≤768 targets/run; no truncation or overshoot. |
| Work | Formula from admitted ceilings: 24 × (260,000,296 candidate microbatch + 32 reader + 27,434,220 update) + 4 × 100,000,000 validation = 7,298,429,152. Set 8,000,000,000 cumulative work/run (≈9.6% margin); sweep bound 24,000,000,000 across three separate run identities. |
| Memory | 64 MiB run-owned cap with 4 MiB data allowance and matching/lower autograd cap; retain the 128 MiB project hard ceiling. Current measured peak is 42.44 MB, leaving bounded slack. Reject if actual plan does not validate. |
| Runtime | 120 s lineage active cap/run; 90 s per attempt; reserve 30 s shutdown grace in the invoking runner. These are stop-at-safe-point limits, not a hard preemption guarantee. |
| Validation/checkpoint | Every 6 completed updates, checkpoint first then evaluate the fixed validation batches; final checkpoint and report required. Test-family evaluation is one locked post-selection check only. |
| Expected artifacts | One replacing checkpoint of about 11 MB (<16 MiB), metrics/handoff/final report/journal under 100 KB per run; allow <12 MB per trial and <36 MB total for three run roots. These are format-derived estimates, not measured future artifacts. |
| Success | All rates remain finite; at least one run lowers validation mean CE by ≥0.05 from its own initial weights after 24 updates; parameter and both moment inventories change; resume reproduces the selected validation result bitwise; logs/checkpoint/report validate. Select rate only on validation. |
| Failure | Any non-finite loss/gradient/state, budget overrun, checkpoint/lineage inconsistency, missing required logs/report, no validation decrease for all candidates, or test-family use during selection. Stop and preserve artifacts; do not expand steps to rescue a failed criterion. |
| Execution | Bounded CPU GitHub Actions is feasible for a 24-update × 3 sweep at measured ~40 supervised tokens/s (rough estimate ~58 s compute plus setup/validation/checkpoints); use a manually dispatched bounded workflow, no PR-triggered training. Re-measure on runner before approving. |
| External compute gate | Consider only if the frozen objective/data coverage genuinely requires more tokens than measured CPU runtime permits and a bounded CPU trial demonstrates the bottleneck. Require explicit owner budget/runtime authorization, verified official CLI lifecycle, disconnect survival, persistent checkpoints and cleanup. No Colab/GPU provisioning in this audit. |

## S. Owner acceptance

The numerical core and bounded CPU research path are ready for owner review. Accept TASK 006-C only with the stated conditional verdict and limits. Before calling the engine ready for detached use, close F-01 and F-02; decide whether F-03’s trusted artifact-directory assumption is acceptable or must fail closed. No merge, training dispatch, compute-unit use, or trained-model release is authorized by this report.
