# TASK 006-F — First Training Readiness Gate

## Verdict: BLOCKED

The integration candidate has substantial local software evidence, but the first training run is not scientifically admitted or operationally recoverable. Current blockers are substantive: the v3 public test has already been observed; 18 validation examples from one source are inadequate to support a defensible three-rate comparison; no complete v3 workload has demonstrated fit under its per-run work/runtime ceiling; private off-host checkpoint persistence is unverified; and R2/C3 lack acceptance by a reviewer separate from their implementation/audit author.

**Accepted training source SHA: none.** `main` remains the accepted TASK 005 baseline at `32eca2b900c4f7a88294f7a24a5dfae0604ae798`. The F integration branch is a review candidate, not an accepted run source. Do not dispatch from this branch or treat historical PR CI as integrated validation.

## Candidate model/data identity (not frozen for dispatch)

| Field | Current candidate value | Gate status |
|---|---|---|
| Architecture | Native Transformer; vocabulary 292, hidden 96, 8 blocks, 4 heads, FFN 256, context 128; 914,400 parameters / 74 tensors | TASK 005 identity accepted as architecture; exact integrated source is not accepted |
| Reference batch | `B=1, T=32` | Software path only |
| Dataset | Objective A v3, ID `6b8d4b140c8d20636b206cb6259a1789a2734f5a13d7fd234e3d39aad3037c87`; 24/18/18 rows | Reproducible but not admitted; public test exposed |
| Manifest SHA-256 | `db2577c6ed7f0f74c69b643dc9d74683c76084ddf8dc21bddc0ce072c56562b8` | Candidate identity |
| Documents SHA-256 | `76842cc832ec49fe807e5b7f6bbdbe0a4ba08f96190629040512d97be9a8874a` | Candidate identity |
| Tokenizer | `byte-bpe-v1`, 32 train-only merges; SHA-256 `9ac03af34eaa496a54dc7721f83107ecdd14c26c643fb4c51d34de9afe77d6d0` | Candidate identity; train-only fit reproduced |
| Tokenizer fit record SHA-256 | `28d037fb18b57231ede816cd2bb673230b211c92f3c5bf611287b55915b8d203` | Candidate identity |
| Unchanged-weight baseline | Seed 107; state SHA `470fb79f5ec91c7e805115c19b90cb198586ff760b03ed7fafe4742f19f57a44`; train/val/public-test full-sequence CE `5.702619792` / `5.700838645` / `5.699661671` | Baseline plumbing verified; test is public and observed |
| Optimizer/schedule | Native AdamW; v3 two-update smoke uses beta1 .9, beta2 .999, epsilon 1e-8, weight decay .01 | Smoke settings only; no first-run LR/schedule frozen |
| Randomness | Baseline seed 107 | Training seed/config not frozen |

These hashes identify the current candidate artifacts; they are not approval. Any replacement dataset, source, config, seed, optimizer schedule, or evaluation rule requires a new immutable bundle and audit.

## Evaluation protocol and measurable gates

Objective A is a controlled task-learning exercise only. It cannot establish cybersecurity intelligence. Objective B remains outside scope pending its own source, license, and quality review. Retire v3’s public test as a hidden set. A future test must be newly versioned, hidden from training and model selection, and accessed once after all choices freeze.

Before execution, the owner and independent reviewer must freeze a single run configuration and a preregistered protocol with:

1. An independently reviewed train/validation set and adequate held-out examples, grouped against shared template/task-generator/source leakage; deterministic train-only tokenizer fit.
2. One declared optimizer/schedule and seed. The old three-LR sweep is retired. If candidate selection is necessary, use validation only and retain a distinct prospective test.
3. Full-sequence token-weighted validation cross-entropy as primary metric until answer-token alignment is corrected.
4. A frozen unchanged-weight baseline on the same validation set, paired per-example change, confidence interval clustered by family/template, and a minimum practical effect margin selected before training. Require the lower confidence bound to exceed that margin, all metrics finite, and no predeclared catastrophic family regression. Numeric margin/sample size are currently **unresolved**, since the existing validation sample is not adequate to estimate them.
5. Audit logs proving no test access before final evaluation and exact hashes for source, config, dataset, tokenizer, baseline, and final artifacts.

**Current acceptance threshold status: not defined.** No post-update evidence exists from which a numeric threshold can be chosen without post hoc bias. A decreasing training loss alone does not pass.

## Resource limits and current cost evidence

Do not exceed accepted limits: ordinary work 100M; scoped candidate backward 160M; tensor 64 MiB; autograd ledger 128 MiB; training owned memory 128 MiB; checkpoint 16 MiB; 10,000 maximum updates. The proposed bounded Actions pilot is more restrictive: 24 updates, 768 consumed/supervised tokens, 8B work per run, 64 MiB owned memory, 64 MiB autograd, 4 MiB data, 120 seconds active runtime, 90 seconds per attempt, and checkpoint/validation every six updates.

The v3 two-update partial-validation smoke measured `825,200,872` work units; 12-fold projection at that same cadence is `9,902,410,464`, above 8B. Five unchanged-weight baseline evaluators use `3,124,414,056` forward-work units. These are not a full-run cost model. Initialization, full validation, training updates, checkpoint writes, resume, logging, shutdown, and cleanup have not been measured together. The proposed 3-LR experiment is retired; there is no aggregate budget. The exact full workload must be measured at least three times in the candidate provider and the slowest run must satisfy each cap. Do not raise ceilings or omit scientifically required evaluation to make the run fit.

## Checkpoint location, recovery, and retention

No private off-host checkpoint destination is selected or verified. Do not store model weights in public GitHub or public Actions artifacts. The local recovery set consists of `experiment.json`, `run.journal`, checkpoint, immutable config, dataset manifest/payloads, tokenizer/fit record, and logs/reports. The checkpoint is SHA-256 checked, canonical, semantically validated, and atomically published locally. Local resume equivalence is tested; provider disconnection and remote transfer/restore are not.

Before training, the owner must choose an existing private location and approve access scope, encryption, retention, and any cost/egress. A non-training exercise must interrupt a transfer, verify every digest, restore into a new private staging directory, inspect the checkpoint, resume a bounded recovery control, and clean up. Proposed retention is primary plus independently verified private copy through review and at least 30 days afterward; this is not an active storage policy.

## Dispatch, cancellation, and recovery (future authorized run only)

1. Dispatch only after a new acceptance report records an exact merged source SHA, frozen config digest, all data/tokenizer hashes, private recovery destination, provider, and owner approval. The existing Actions workflow requires `authorize_training_run=true`; leave false until those gates pass. This report does not authorize dispatch.
2. The runtime owns limits, logs, loss/step records, instability/nonfinite detection, checkpoints, resume state, failure/final reports, and cleanup. Start with bounded startup checks and then end active supervision; do not maintain an AI polling loop.
3. For graceful stop, invoke `xynoos-train stop` for the artifact root and allow the runtime safe point to checkpoint and exit. A hard signal/provider termination may charge the full reserved attempt runtime; do not assume refund.
4. Before resume, confirm the old process is gone, inspect status, preserve `.run.lock`, verify journal/checkpoint/config/data/tokenizer identity and digests, then resume the same lineage. Missing/corrupt/mismatched state fails closed; preserve it for diagnosis and do not reconstruct it.
5. After completion or failure, verify final artifact inventory and digests, copy privately, validate restored copy, then apply only the owner-approved retention/cleanup policy.

## Required measurable experiment outcomes

All must pass; otherwise reject or classify the run as inconclusive:

- At least one intended trainable parameter changes and the checkpoint reflects the same state.
- Both AdamW moment inventories change for a nontrivial gradient run; counters and bias powers advance consistently.
- Training objective improves against the frozen unchanged-weight baseline under a predeclared rule.
- Held-out validation improves by more than the preregistered practical margin with the paired uncertainty criterion; all family guards pass.
- Deterministic resume exactly reproduces the uninterrupted trajectory and final state for the same bundle.
- No test access before the final one-time test; access log and hashes verify no leakage.
- Every work, memory, data, checkpoint, update, runtime, and attempt cap is respected, including recovery accounting.
- Final artifacts pass integrity checks and are recoverable from the approved private destination.

## Owner actions

1. Obtain separate independent review of the integrated source and R2/C3 runtime evidence; do not rely on CI status as review.
2. Decide whether to replace or extend Objective A v3. Commission independent task/data review, sufficient held-out validation, and a newly hidden prospective test; keep Objective B separate.
3. Freeze the optimizer/schedule, seed, statistical method, practical effect threshold, and minimum sample size before any learning comparison.
4. Select/authorize private storage, retention and cost boundary, then complete the non-training remote recovery exercise.
5. Measure the full immutable workload at least three times under the intended bounded provider; re-admit only if the slowest run meets every cap.
6. Review exact final source/config/data hashes and explicitly authorize a specific future dispatch. No owner action above is silently inferred from this task request.

## Final disposition

**BLOCKED.** Runtime and numerical software controls pass within their tested scope. Scientific evaluation, full workload fit, private remote recovery, independent reviewer acceptance, and a frozen dispatch bundle remain open. Do not train from this candidate. Reassess after all owner actions and evidence are complete; success must be based on a new exact-source readiness gate.
