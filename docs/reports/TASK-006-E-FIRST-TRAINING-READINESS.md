# TASK 006-E — First Training Readiness Acceptance Package

**Verdict: CONDITIONALLY READY — EXPLICIT BLOCKERS REMAIN.**

**Disposition:** preparation and evidence review only. No training workflow was
dispatched, no Colab runtime or accelerator was provisioned, no compute units
were consumed, no external dataset was acquired, no model weights were
uploaded, and no pull request was merged.

**Evidence refresh:** Objective A v3 on open draft PR #28 supersedes v2 as
the current dataset candidate. V2 numbers remain only where they explain why
its prior workload proposal was retired. V3 improves the recorded template
and task-input separation, but remains a public synthetic control without
human review or a post-update learning result.

This package closes the authorized engineering preparation as far as the
present evidence allows. It is not an immutable run configuration and must
not be used to dispatch training. The minimum first experiment remains a
single, engineering-controlled Objective A learning experiment with a
project-authored deterministic task. It can test a narrow learning signal in
the native training path; it cannot establish cybersecurity intelligence,
broad technical capability, or safe deployment behavior. Objective B remains
out of scope pending separate provenance, licensing, and quality review.

## Acceptance package

| # | Required item | Current evidence and disposition |
|---:|---|---|
| 1 | Accepted source SHA | **None yet.** `main` is `32eca2b900c4f7a88294f7a24a5dfae0604ae798`. Open draft PRs are #18, #19, #20, #21, and #23–#28; #28 is the current D1 v3 candidate at `f122f4e283b06d2cd20dd544fc0b18af4c8b6c98`. Exact-head Foundation CI passed at that SHA (run `45855781546`), with 187 debug and release checks. The candidate and other PR heads are not an accepted training SHA. A future run must bind to the exact clean revision after owner review and merge of the dependency chain. |
| 2 | Model configuration and parameter count | Native `Config::RESEARCH_CANDIDATE`: vocabulary 292, width 96, 8 layers, 4 heads, FFN width 256, context 128; **914,400 parameters**. Baseline seed was 107. The first-run seed and complete configuration are not frozen until the dataset and evaluation protocol are accepted. Weights are initialized by the project implementation; no pretrained model is used. |
| 3 | Tokenizer artifact and digest | Current v3 candidate uses `byte-bpe-v1`, 32 train-only merges, vocabulary 292, artifact SHA-256 `9ac03af34eaa496a54dc7721f83107ecdd14c26c643fb4c51d34de9afe77d6d0`. It was fitted on the 24 training records; a separate standard-library audit reproduced the artifact. This candidate identity must be re-attested in a final immutable run bundle. |
| 4 | Dataset manifest and digest | Current v3 candidate: dataset ID `6b8d4b140c8d20636b206cb6259a1789a2734f5a13d7fd234e3d39aad3037c87`; manifest SHA-256 `db2577c6ed7f0f74c69b643dc9d74683c76084ddf8dc21bddc0ce072c56562b8`. It has 60 project-authored synthetic rows: 24 train, 18 validation, and 18 public test, with parity, integer comparison, and modular progression families. It is a reproducible software control, not yet an admitted learning benchmark. No external dataset was acquired. |
| 5 | Split and leakage report | The v3 audit reports zero normalized exact cross-split duplicates, task-input overlap, template overlap, and cross-split pairs at or above 0.82 character-trigram Jaccard; the maximum is 0.461538. Templates and task inputs are partitioned by split. Limits remain: only 60 examples, one authored source/date, public test, no human review, and plausible task-family memorization; source/time generalization is not measured. |
| 6 | Initial unchanged-weight baseline | On clean source `053b2703632921d9869215a2dad664d2f2b41733`, seed 107, unchanged state SHA `470fb79f5ec91c7e805115c19b90cb198586ff760b03ed7fafe4742f19f57a44`: full-sequence train/validation/public-test CE `5.702619792`/`5.700838645`/`5.699661671` (197/199/172 targets); answer-bearing validation/test CE `5.747362546`/`5.692370273` (28/27 targets). All were finite; no optimizer was constructed and no update occurred. Ten answer spans begin within merged BPE tokens, so answer-bearing is not strict answer-only loss. |
| 7 | Multi-batch validation | V3 evaluates deterministic manifest-ordered batches and token-weights aggregation: 12 train batches and 9 each for validation and test. Full-sequence supervised counts are 197/199/172. Separate answer-bearing validation and test use 28/27 positions. The five baseline evaluators consumed 3,124,414,056 forward-work units in total. This confirms evaluation plumbing and unchanged-weight baseline behavior, not learning. |
| 8 | Locked test evaluation | Training admission excludes validation and test data, and evaluation records explicit test access. The v3 baseline read all 9 batches of its public test split (172 full-sequence targets; 27 answer-bearing positions). Because the test is visible in Git and was read for baseline, it is **not a hidden or locked test set**. The two-update smoke evaluated only one validation batch per update and did not access test; it is not an admitted post-update result. A reviewed test-access policy and truly held-out evaluation remain open. |
| 9 | Optimizer and scheduler | Native AdamW and linear warmup/decay are implemented. The v3 software control uses AdamW `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`, weight decay `0.01`, and a two-update schedule; these are test settings, not an accepted experiment choice. The earlier three-learning-rate sweep is retired. A single schedule must be predeclared against reviewed Objective A data before an experiment config is frozen. |
| 10 | Numerical precision policy | Current candidate uses `f32` parameters, gradients, and Adam moment inventories. The optimizer and schedule use `f64` for relevant intermediate accumulation, bias-power, and schedule calculations before narrowing at defined update boundaries; evaluation aggregates token-weighted losses in `f64`. No mixed precision, quantization, CUDA backend, or external ML framework is included. This is the CPU reference policy; any future accelerated backend requires independent numerical parity first. |
| 11 | Work budget | The manual CPU workflow caps runs at 24 updates, 768 consumed/supervised tokens, and 8,000,000,000 cumulative work units. The v3 two-update software control measured 825,200,872 cumulative units with one partial validation batch and a checkpoint per update. A 12-fold projection for 24 updates at that same cadence is 9,902,410,464 units, over the ceiling; this is an extrapolation, not a full-run measurement, and it excludes the initial baseline and full validation coverage. The five-evaluator unchanged-weight baseline separately measured 3,124,414,056 forward-work units. The v2 estimate is retired. Twenty-four updates and 768 tokens are neither selected nor shown scientifically sufficient; the current 24-update cadence does not fit the ceiling even before complete validation. |
| 12 | Memory budget | Workflow ceilings are 64 MiB run-owned memory, 64 MiB autograd allowance, 4 MiB data allowance, and 16 MiB checkpoint size. The local debug v3 two-update software control reported owned-memory peak 42,439,416 bytes, autograd-ledger peak 30,682,296 bytes, and process high-water 37,212,160 bytes. Earlier v1 controls reported different process high-water on local and hosted runs. Application ledgers are not hard RSS limits; none of these tiny controls represents a complete experiment workload. |
| 13 | Runtime budget | Workflow caps are 120 seconds active runtime and 90 seconds per attempt; job timeout is 10 minutes and outer process timeout is 240 seconds with a 30-second termination grace. The local debug v3 control took 17.2875 seconds active for two updates, two one-batch partial validation events, and per-update checkpointing. It is not a provider throughput estimate or complete-workload measurement. Historical v1 rates and the 91.38-second v3 unchanged-weight baseline are not transferable to this workload. No three repeated full-workload measurements exist, so the runtime cap remains unverified. |
| 14 | Checkpoint and recovery | Local checkpoint publication is staged, synced, SHA-256 checked, atomically renamed, and directory-synced; checkpoint semantics and resume compatibility are validated. D2 defines the recovery inventory and interrupted-transfer procedure. The repository is public, Actions artifacts are not private storage, and no remote private destination, upload interruption, restore, resume, or remote retention lifecycle has been exercised. The proposed retention is the primary recovery set plus one verified private copy through review and at least 30 days afterward; it is not an active provider policy. |
| 15 | Logging and monitoring | Runtime-owned JSONL metrics, journal counters, failure/final reports, checkpointing, stop requests, and handoff/integrity metadata are implemented. Resource limits and safe-boundary stop behavior are runtime enforced within the stated application limits. Local detached-process lifecycle tests pass; provider disconnect, remote artifact survival, and remote cleanup remain unverified. AI-agent continuous monitoring is not part of the design. |
| 16 | Compute provider rationale | GitHub Actions remains the default for bounded CI, numerical oracles, and regression. The public `ubuntu-24.04` runner is documented as 4 vCPUs, 16 GB RAM, and 14 GB SSD; bounded CI is 15 minutes, while the manual CPU workflow caps the run at 120 seconds and the job at 10 minutes. No training workflow was dispatched. The installed Colab CLI `0.8.0` was inspected read-only: CPU is the default; GPU/TPU options exist, but there is no native XYNOOS GPU backend. `colab sessions` showed no active sessions, `colab usage` was inspected without recording account balance, and no runtime was created. No provider is selected because the full CPU workload is unmeasured and detached Colab, transfer integrity, persistence, and cleanup remain unverified. |
| 17 | Reproducibility procedure | Before a future dispatch, freeze an immutable repository-relative config and bundle; record repository identity, exact clean source revision, config SHA-256, Objective A manifest and split digests, tokenizer and fit-record digests, model identity, seed, deterministic order, optimizer/schedule, evaluation policy, and budget. Validate the bundle with the native CLI, record exact CI and preflight evidence, and require every identity to match the experiment record. These steps are not yet instantiated into a runnable first-run config. |
| 18 | Stop and failure recovery | Use runtime safe-boundary stop, inspect status and process identity, preserve the persistent lock inode, and do not resume until the previous owner is confirmed stopped. Restore only into a new private staging directory; verify the complete inventory and checkpoint identity before publishing and resuming the same lineage. The complete procedure is in the [D2 recovery report](TASK-006-D2-CHECKPOINT-PERSISTENCE.md). It is locally specified/tested but not verified against a remote provider. |
| 19 | Expected outputs and acceptance criteria | Expected local run artifacts: immutable experiment identity, monotone journal, checkpoint, structured metrics, final or failure report, handoff, and integrity inventory. Store only in an owner-approved private destination. Before dispatch, preregister a meaningful held-out objective and evaluation threshold from the reviewed dataset. After execution, require finite diagnostics, changed parameter and optimizer states, complete budget accounting, bitwise resume equivalence, complete validation coverage, and no locked-test influence on training or selection. Report a held-out test result only after the selection procedure is fixed. No numeric loss threshold is accepted yet; the old proposed threshold tied to 24 updates is retired. |
| 20 | Experiment question and claim boundary | **Question:** can the native training, checkpoint, and resume path learn a predeclared pattern on project-authored Objective A examples and improve on a reviewed held-out combination/family under a bounded, reproducible run? **Claim boundary:** this is an engineering-controlled learning experiment only. It cannot support a cybersecurity intelligence or Objective B claim. |

## Engineering milestones and hosted evidence

The sequence is kept in separate branches and open draft pull requests; no
merge has occurred:

| Milestone | Candidate head | Draft PR | Exact hosted evidence |
|---|---|---:|---|
| C native training implementation | `231d47d2d95f52770fa20f4da93a91be762178d0` | [#20](https://github.com/samuelindra123/xynoos-cyber-model/pull/20) | Foundation CI [45731891128](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45731891128) passed |
| R2 runtime remediation | `21d9990760380195b9d9379b6d43f848ad6c17df` | [#21](https://github.com/samuelindra123/xynoos-cyber-model/pull/21) | Foundation CI [45772399582](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45772399582), all 29 steps passed |
| C3 re-verification | `b7656abbc38703b3a34652cc5c7f9435a2b0c32c` | [#23](https://github.com/samuelindra123/xynoos-cyber-model/pull/23) | Exact-head Foundation CI [45824251360](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45824251360) passed; the five auditor-authored tests and 170 debug/release checks succeeded. |
| D1 evaluation | `92da22973914ab36055c4aa6d32143747d9a372f` | [#24](https://github.com/samuelindra123/xynoos-cyber-model/pull/24) | Foundation CI [45781385665](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45781385665), all 33 steps passed |
| D1 Objective A v2 candidate (historical) | `546c58a710ee7a8edf3a7d96495fd04e9838a221` | [#28](https://github.com/samuelindra123/xynoos-cyber-model/pull/28) | Exact-head Foundation CI [45831989897](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45831989897) passed; superseded as the current corpus candidate. |
| D1 Objective A v3 candidate | `f122f4e283b06d2cd20dd544fc0b18af4c8b6c98` | [#28](https://github.com/samuelindra123/xynoos-cyber-model/pull/28) | Exact-head Foundation CI [45855781546](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45855781546) succeeded: all 33 steps, 187 debug/release checks, numerical and corpus audits, resource checks, and Rust documentation. PR remains an open draft. |
| D2 private-persistence preparation | `3cf67c9e389e7b00ab5b9df9cf045c6f465f255c` | [#25](https://github.com/samuelindra123/xynoos-cyber-model/pull/25) | Foundation CI [45784299844](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45784299844), all 33 steps passed |
| D3 compute readiness | `20bcb3cfb9f257932d2e099845b9f3032d8732ff` | [#26](https://github.com/samuelindra123/xynoos-cyber-model/pull/26) | Exact-head Foundation CI [45824896343](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45824896343) passed all steps. |

These are per-PR exact-head CI results, not evidence that PRs were merged or
that the first experiment ran. Relevant local Rust suites, formatting,
Clippy, documentation, independent numerical/data fixtures, and repository
hygiene are recorded in the linked milestone reports. No training workflow
run exists.

## Evidence classification

### Implemented and executed

- R2 kernel-backed exclusive journal ownership, stale-owner recovery,
  source-attestation checks, and transactional initial sidecar publication.
- C3 auditor-authored process and fault tests, including repeated concurrent
  stale-lock recovery and process-crash recovery; test evidence also covers
  journal monotonicity, checkpoint semantics/publication, trainer logging
  failure, optimizer transaction boundaries, resume equivalence, resource
  accounting, and local detached stop/restart behavior. The sidecar fault
  test disables Git optional index writes so its file-size limit reaches the
  staging write after source attestation; no runtime implementation behavior
  is bypassed.
- D1 deterministic multi-batch, token-weighted evaluation and split
  admission; the v3 unchanged-weight baseline, standard-library split and
  tokenizer audit, and trainer-compatible per-document padded stream are
  recorded on the draft candidate. Its bounded two-update integration control
  measured 825,200,872 work units with partial validation. The audit finds no
  tested cross-split template/task-input overlap, but the corpus remains
  small, public, single-source, and unreviewed.
- D1 answer-bearing baseline positions were cross-checked against the v3
  audit fixture. Ten answer starts intersect merged BPE tokens, so this is
  not strict answer-only loss. No post-update improvement was demonstrated.
- A trainer-facing stream now emits 24 independently padded `[1,32]` batches
  for the v3 train split, preserving document boundaries and 197 supervised
  targets. The two-update integration control confirms finite loss, parameter
  change, validation-only access, checkpoint creation, and no test access. It
  is software-path verification only; it does not demonstrate a meaningful
  held-out improvement or scientific sufficiency.
- D2 owner-only local artifacts, checkpoint integrity and atomic publication,
  and a documented recovery inventory/procedure.
- D3 bounded workflow and CLI capability inspection; local Objective A
  software-control throughput/memory samples.
- Exact hosted Foundation CI for each candidate head listed above.

### Inspected only

- GitHub runner limits, cancellation semantics, and public artifact access.
- Colab CLI command surface, available session state, and usage command.
- Colab remote lifecycle/storage behavior from official documentation.
- Candidate storage design and access requirements; no private provider
  destination or account permission was changed.

### Not verified or not accepted

- An admitted Objective A learning benchmark and independent human review.
  The v3 60-row candidate has no tested cross-split exact, task-input,
  template, or >=0.82 trigram-similarity overlaps, but has one source/date,
  a small task inventory, a public test, and no human quality review.
- A locked/hidden test artifact. V3's public test was read for its initial
  baseline and cannot support a hidden holdout claim.
- A post-update validation improvement or strict answer-only metric. V3 has
  unchanged-weight evidence only, and ten answer spans begin inside merged
  BPE tokens that also contain prompt bytes.
- A full v3 learning workload that fits the current work and runtime budgets.
  The two-update partial-validation control measured 825,200,872 work units;
  its 24-update linear projection is 9,902,410,464, above the 8-billion cap,
  before full validation coverage and initial baseline costs. No complete v3
  training workload was run or admitted.
- A separate reviewer independent of the R2/C3 author accepting the runtime
  remediation. C3 is a separate verification pass with auditor-authored
  controls, but the same autonomous author implemented and audited it.
- Private off-host checkpoint persistence, interrupted transfer, restore,
  remote resume, retention enforcement, or cleanup.
- Three repeated complete-workload measurements on the intended provider,
  exact provider RSS, or detached survival after provider disconnection.
- A frozen optimizer/scheduler, complete first-run configuration, final
  numerical acceptance threshold, and a provider-specific approved budget.
- An accepted exact source SHA after owner review and merge of the dependent
  implementation PRs.

## Explicit blockers before first training

1. Obtain separate human/independent review of the frozen R2 implementation
   and C3 evidence; review and merge the dependent engineering PRs.
2. Have an independent reviewer assess Objective A v3 task validity,
   tokenizer provenance, held-out family coverage, leakage limits, and
   test-access policy. Keep test data out of schedule selection. Decide
   whether v3 is adequate for a narrow engineering learning control or
   prepare a reviewed alternative; it currently establishes evaluation
   software behavior only.
3. Select an owner-controlled private checkpoint destination and retention
   policy, then pass a bounded non-training interrupted-upload, digest,
   restore, checkpoint-inspection, resume, and cleanup exercise. Confirm any
   permissions and cost/egress budget first.
4. Derive a scientifically justified v3 workload using the supported
   per-document `[1,32]` train stream and complete held-out validation. The
   current two-update partial-validation control projects to 9,902,410,464
   work units for 24 updates, already above the 8-billion ceiling. Do not
   increase the ceiling or reduce held-out coverage to fit it. Rework the
   reviewed experiment design and resource admission, then repeat the exact
   complete CPU workload at least three times in the intended bounded
   provider, recording throughput, wall time, work, both memory ledgers,
   process RSS/high-water, checkpoint time, and recovery outcome.
5. Freeze the exact merged clean source SHA, configuration and all data
   digests, predeclared optimizer/schedule and acceptance criteria. Confirm
   private artifact handling and provider lifecycle before selecting a
   provider.
6. Obtain owner authorization for the exact immutable configuration and
   selected provider before dispatch. This report does not authorize it.

## Final disposition

The runtime, scientific-evaluation code, local checkpoint integrity, bounded
workflow, and exact-head CI evidence are prepared. The v3 baseline consumes
3,124,414,056 forward-work units for its five unchanged-weight evaluators.
The two-update trainer control measured 825,200,872 cumulative work units;
its partial-validation 24-update projection is 9,902,410,464, above the
8-billion ceiling and not a full-workload estimate. Current evidence does not
establish an admitted learning benchmark, independent acceptance, a viable
full-coverage workload, private remote persistence, or provider-specific
full-workload fit. The first experiment is **not dispatch-ready**. Complete
the blockers above and issue a fresh acceptance against the exact merged
source and frozen run bundle before any owner-authorized dispatch.
