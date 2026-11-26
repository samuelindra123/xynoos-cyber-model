# TASK 006-E — First Training Readiness Acceptance Package

**Verdict: CONDITIONALLY READY — EXPLICIT BLOCKERS REMAIN.**

**Disposition:** preparation and evidence review only. No training workflow was
dispatched, no Colab runtime or accelerator was provisioned, no compute units
were consumed, no external dataset was acquired, no model weights were
uploaded, and no pull request was merged.

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
| 1 | Accepted source SHA | **None yet.** The implementation sequence remains unmerged draft PRs. The latest D3 candidate is `adab941a5c7372019590b2c4cad5e15d6a6e7371`, with exact-head Foundation CI run [45786197304](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45786197304) passing all 33 steps. The D1 baseline used clean source `da99cba3016b7fe564c42e65d8d2babe44b3b98f`; D3 smoke measurements used clean source `0b798e91b550f30d37683eab2c6bf66382a88c28`. A future configuration must bind to the exact clean source SHA after the owner reviews and merges its dependency chain. |
| 2 | Model configuration and parameter count | Native `Config::RESEARCH_CANDIDATE`: vocabulary 292, width 96, 8 layers, 4 heads, FFN width 256, context 128; **914,400 parameters**. Baseline seed was 107. The first-run seed and complete configuration are not frozen until the dataset and evaluation protocol are accepted. Weights are initialized by the project implementation; no pretrained model is used. |
| 3 | Tokenizer artifact and digest | `byte-bpe-v1`, SHA-256 `86aa40996d4a9bfe15491131bbe6fb1b04f5e325a50b23a3d7f0fa06bb56011c`. Existing fit provenance points to the Task 004 train-only manifest `0041f6e5ff80d175265e5675da78aebe2db12007c0d4744f9b6e06f50a27e502`. The artifact and fit record must be copied into and re-attested by the final immutable run bundle. |
| 4 | Dataset manifest and digest | **Missing for an acceptable first run.** The available Objective A fixture SHA-256 is `63050baf97f301c5ca6a3ad07a1a632bef77938b69f0317622738b5bbc00fa33`; it is a test fixture, not an admitted manifest-backed experiment dataset. No external dataset was acquired. |
| 5 | Split and leakage report | Existing audit covers 2 rows and 64 targets per split. It found no normalized exact duplicates and no cross-split pair above trigram Jaccard 0.82; maximum was 0.717391. Train and validation share `conditional` and `loop` families; both test rows are from one `iteration` family. Character trigrams do not rule out semantic or task leakage. This is not adequate held-out coverage. The public fixture is not a hidden locked test set. |
| 6 | Initial unchanged-weight baseline | At the D1 baseline source SHA, seed 107, initial model digest `470fb79f5ec91c7e805115c19b90cb198586ff760b03ed7fafe4742f19f57a44`: train CE `5.710966587`, validation CE `5.695174217`, test CE `5.690947771`; each split had 2 batches and 64 supervised targets. Losses were finite and all reports shared the same model-state digest. Test access was recorded and did not affect training or optimizer state. These are baseline-generation checks on the tiny fixture, not evidence of generalization. |
| 7 | Multi-batch validation | Deterministic multi-batch evaluation and token-weighted aggregation are implemented and exercised with two batches. The fixture evidence covers 64 validation targets only. The final dataset must set limits for complete, stable held-out coverage; no such accepted limit exists yet. |
| 8 | Locked test evaluation | Training admission excludes validation and test splits; the evaluator tags test access. The initial fixture test baseline was read and the fixture is publicly visible. A genuinely useful held-out test must be newly defined, frozen, access-recorded, and excluded from training and schedule selection. No post-training test result exists. |
| 9 | Optimizer and scheduler | Native AdamW and linear warmup/decay are implemented. The integration control uses AdamW `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`, weight decay `0.01`; its two-update schedule is a software test setting, not an accepted experiment choice. The earlier three-learning-rate sweep is retired. A single schedule must be predeclared against reviewed Objective A data before an experiment config is frozen. |
| 10 | Numerical precision policy | Current candidate uses `f32` parameters, gradients, and Adam moment inventories. The optimizer and schedule use `f64` for relevant intermediate accumulation, bias-power, and schedule calculations before narrowing at defined update boundaries; evaluation aggregates token-weighted losses in `f64`. No mixed precision, quantization, CUDA backend, or external ML framework is included. This is the CPU reference policy; any future accelerated backend requires independent numerical parity first. |
| 11 | Work budget | The manual CPU workflow caps a run at 24 updates, 768 consumed/supervised tokens, and 8,000,000,000 cumulative work units. The two-batch-validation proposal calculates 7,782,315,616 units, leaving 2.72% headroom before separate initial-baseline and locked-test work. One additional validation batch at each cadence exceeds the cap. The 24-update/768-token plan is **not accepted** as scientifically sufficient, and the actual experiment budget is not yet established. |
| 12 | Memory budget | Workflow ceilings are 64 MiB run-owned memory, 64 MiB autograd allowance, 4 MiB data allowance, and 16 MiB checkpoint size. The local two-update control observed application-owned peak 42,439,416 bytes and autograd-ledger peak 30,682,296 bytes; process high-water was measured separately (release test: 130,641,920 bytes). Application ledgers are not hard RSS limits. The exact full workload and selected provider have not been measured. |
| 13 | Runtime budget | Workflow caps are 120 seconds active run time and 90 seconds per attempt; the job timeout is 10 minutes and the outer process timeout is 240 seconds with a 30-second termination grace. Local two-update release control measured 18.913 supervised targets/second; debug measured 2.048. A linear 768-target projection gives about 40.6 seconds release and 375 seconds debug, but omits complete evaluation, checkpoint, initialization, and runner variability. The exact candidate workload has not been measured three times on a candidate provider. The 120-second cap is therefore unproven, not shown to pass or fail. |
| 14 | Checkpoint and recovery | Local checkpoint publication is staged, synced, SHA-256 checked, atomically renamed, and directory-synced; checkpoint semantics and resume compatibility are validated. D2 defines the recovery inventory and interrupted-transfer procedure. The repository is public, Actions artifacts are not private storage, and no remote private destination, upload interruption, restore, resume, or remote retention lifecycle has been exercised. The proposed retention is the primary recovery set plus one verified private copy through review and at least 30 days afterward; it is not an active provider policy. |
| 15 | Logging and monitoring | Runtime-owned JSONL metrics, journal counters, failure/final reports, checkpointing, stop requests, and handoff/integrity metadata are implemented. Resource limits and safe-boundary stop behavior are runtime enforced within the stated application limits. Local detached-process lifecycle tests pass; provider disconnect, remote artifact survival, and remote cleanup remain unverified. AI-agent continuous monitoring is not part of the design. |
| 16 | Compute provider rationale | GitHub Actions remains the default for bounded CI, numerical oracles, and regression. Its public runner limits and workflow behavior were inspected; the training workflow was not run. No provider is selected for the first experiment. Colab CLI `0.8.0` help, sessions, and usage capability were inspected read-only; no runtime was created. There is no admitted GPU backend or measured reason to use an accelerator. |
| 17 | Reproducibility procedure | Before a future dispatch, freeze an immutable repository-relative config and bundle; record repository identity, exact clean source revision, config SHA-256, Objective A manifest and split digests, tokenizer and fit-record digests, model identity, seed, deterministic order, optimizer/schedule, evaluation policy, and budget. Validate the bundle with the native CLI, record exact CI and preflight evidence, and require every identity to match the experiment record. These steps are not yet instantiated into a runnable first-run config. |
| 18 | Stop and failure recovery | Use runtime safe-boundary stop, inspect status and process identity, preserve the persistent lock inode, and do not resume until the previous owner is confirmed stopped. Restore only into a new private staging directory; verify the complete inventory and checkpoint identity before publishing and resuming the same lineage. The complete procedure is in the [D2 recovery report](TASK-006-D2-CHECKPOINT-PERSISTENCE.md). It is locally specified/tested but not verified against a remote provider. |
| 19 | Expected outputs and acceptance criteria | Expected local run artifacts: immutable experiment identity, monotone journal, checkpoint, structured metrics, final or failure report, handoff, and integrity inventory. Store only in an owner-approved private destination. Before dispatch, preregister a meaningful held-out objective and evaluation threshold from the reviewed dataset. After execution, require finite diagnostics, changed parameter and optimizer states, complete budget accounting, bitwise resume equivalence, complete validation coverage, and no locked-test influence on training or selection. Report a held-out test result only after the selection procedure is fixed. No numeric loss threshold is accepted yet; the old proposed threshold tied to 24 updates is retired. |
| 20 | Experiment question and claim boundary | **Question:** can the native training, checkpoint, and resume path learn a predeclared pattern on project-authored Objective A examples and improve on a reviewed held-out combination/family under a bounded, reproducible run? **Claim boundary:** this is an engineering-controlled learning experiment only. It cannot support a cybersecurity intelligence or Objective B claim. |

## Engineering milestones and hosted evidence

The sequence is kept in separate branches and open draft pull requests; no
merge has occurred:

| Milestone | Candidate head | Draft PR | Exact hosted evidence |
|---|---|---:|---|
| R2 runtime remediation | `21d9990760380195b9d9379b6d43f848ad6c17df` | [#21](https://github.com/samuelindra123/xynoos-cyber-model/pull/21) | Foundation CI [45772399582](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45772399582), all 29 steps passed |
| C3 re-verification | `52c13165f164e1212e023e7401ed682ba00e44f1` | [#23](https://github.com/samuelindra123/xynoos-cyber-model/pull/23) | Foundation CI [45773532076](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45773532076), all 29 steps passed |
| D1 evaluation | `92da22973914ab36055c4aa6d32143747d9a372f` | [#24](https://github.com/samuelindra123/xynoos-cyber-model/pull/24) | Foundation CI [45781385665](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45781385665), all 33 steps passed |
| D2 private-persistence preparation | `3cf67c9e389e7b00ab5b9df9cf045c6f465f255c` | [#25](https://github.com/samuelindra123/xynoos-cyber-model/pull/25) | Foundation CI [45784299844](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45784299844), all 33 steps passed |
| D3 compute readiness | `adab941a5c7372019590b2c4cad5e15d6a6e7371` | [#26](https://github.com/samuelindra123/xynoos-cyber-model/pull/26) | Corrected exact-head Foundation CI [45786197304](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45786197304), all 33 steps passed |

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
  accounting, and local detached stop/restart behavior.
- D1 deterministic multi-batch, token-weighted evaluation, split admission,
  test-access reporting, fixture baseline, and split/tokenizer leakage audit.
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

- A larger manifest-backed Objective A corpus, an independently reviewed
  split/leakage analysis, or a meaningful held-out family result.
- A hidden/locked test artifact suitable for first-run generalization
  measurement; the existing fixture is public and too small.
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
2. Author and freeze a larger, manifest-backed Objective A task set with
   reviewed held-out family coverage, tokenizer provenance, split/leakage
   report, and a test-access policy. Keep the test isolated from schedule
   choice. The small public fixture is not a substitute.
3. Select an owner-controlled private checkpoint destination and retention
   policy, then pass a bounded non-training interrupted-upload, digest,
   restore, checkpoint-inspection, resume, and cleanup exercise. Confirm any
   permissions and cost/egress budget first.
4. Derive one experiment's update count, validation cadence, work, memory,
   data, and runtime limits from the frozen dataset. Repeat the complete
   candidate CPU workload at least three times in the intended bounded
   provider; capture throughput, wall time, work, both application memory
   ledgers, process RSS/high-water, checkpoint time, and recovery outcome.
5. Freeze the exact merged clean source SHA, configuration and all data
   digests, predeclared optimizer/schedule and acceptance criteria. Confirm
   private artifact handling and provider lifecycle before selecting a
   provider.
6. Obtain owner authorization for the exact immutable configuration and
   selected provider before dispatch. This report does not authorize it.

## Final disposition

The runtime, scientific-evaluation code, local checkpoint integrity, bounded
workflow, and exact-head CI evidence are prepared. The current evidence does
not establish adequate scientific data, independent acceptance, private
remote persistence, or provider-specific full-workload fit. Therefore the
first experiment is **not dispatch-ready**. Complete the blockers above and
issue a fresh acceptance against the exact merged source and frozen run
bundle before any owner-authorized dispatch.
