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
| 1 | Accepted source SHA | **None yet.** The implementation sequence remains unmerged draft PRs. D3 candidate `adab941a5c7372019590b2c4cad5e15d6a6e7371` has exact-head Foundation CI run [45786197304](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45786197304) passing all 33 steps. The Objective A v2 baseline ran on clean source `68d3f48d345545bb51fc35f833b2a2505fbdc765`; its follow-up candidate is PR #28, which remains open and unmerged. D3 smoke measurements used clean source `0b798e91b550f30d37683eab2c6bf66382a88c28`. A future configuration must bind to the exact clean source SHA after the owner reviews and merges its dependency chain. |
| 2 | Model configuration and parameter count | Native `Config::RESEARCH_CANDIDATE`: vocabulary 292, width 96, 8 layers, 4 heads, FFN width 256, context 128; **914,400 parameters**. Baseline seed was 107. The first-run seed and complete configuration are not frozen until the dataset and evaluation protocol are accepted. Weights are initialized by the project implementation; no pretrained model is used. |
| 3 | Tokenizer artifact and digest | Objective A v2 candidate uses `byte-bpe-v1`, 32 train-only merges, vocabulary 292, artifact SHA-256 `a5a6e350d0863af2a41aeae9a73a6e07d304513f243d293c79e91eb61f39bd09`. Its fit record and artifact are independently reproduced from the 24 training rows. These candidate identities must be copied into and re-attested by a final immutable run bundle. |
| 4 | Dataset manifest and digest | Candidate manifest-backed Objective A v2: dataset ID `2a5eda1b0668ee2ec9671b70b5becacc4e8479a28adec212853fd23d6d4d214c`; manifest SHA-256 `b3574cbcd769359ae4fc7ec4ff3f4df4f8fc5351cb004944826f87f9c38cfac1`. It has 48 project-authored synthetic rows (24 train, 12 validation, 12 public test). This improves identity and coverage over the two-row fixture but is **not admitted** as a sufficient experiment dataset. No external dataset was acquired. |
| 5 | Split and leakage report | The v2 audit finds no normalized exact cross-split duplicate and no held-out operand-combination overlap, but finds **23 cross-split pairs above the fixed 0.82 character-trigram Jaccard threshold**, maximum 0.930233. Each split has 12 rows per task family. The repeated templates make this candidate inadequate for an independent generalization claim; its public test split is not a secret holdout. |
| 6 | Initial unchanged-weight baseline | On clean candidate source `68d3f48d345545bb51fc35f833b2a2505fbdc765`, seed 107, initial model digest `470fb79f5ec91c7e805115c19b90cb198586ff760b03ed7fafe4742f19f57a44`: train CE `5.696532329` (24 batches, 768 targets), validation CE `5.704625408` (12 batches, 384 targets), test CE `5.697565953` (12 batches, 384 targets). All losses were finite and the three reports share the same unchanged model-state digest. The test split was read and tagged, but did not enter tokenizer fitting, training, or optimizer state. These are baseline-generation checks on a narrow synthetic control, not evidence of generalization. |
| 7 | Multi-batch validation | Deterministic token-weighted evaluation now covers all 12 v2 validation batches and 384 supervised targets; the same baseline covers all 24 training batches and all 12 test batches. Ordering and split identities are reproducible. This candidate still has too few templated examples for a meaningful broad learning claim, and the final admitted dataset and experiment work limits remain undecided. |
| 8 | Locked test evaluation | Training admission excludes validation and test splits; evaluator reports tag test access. The v2 baseline read the full test split (12 batches, 384 targets), and the split is public. It is not a hidden locked test artifact. A reviewed, frozen test-access policy and adequate held-out coverage are still required; no post-training result exists. |
| 9 | Optimizer and scheduler | Native AdamW and linear warmup/decay are implemented. The integration control uses AdamW `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`, weight decay `0.01`; its two-update schedule is a software test setting, not an accepted experiment choice. The earlier three-learning-rate sweep is retired. A single schedule must be predeclared against reviewed Objective A data before an experiment config is frozen. |
| 10 | Numerical precision policy | Current candidate uses `f32` parameters, gradients, and Adam moment inventories. The optimizer and schedule use `f64` for relevant intermediate accumulation, bias-power, and schedule calculations before narrowing at defined update boundaries; evaluation aggregates token-weighted losses in `f64`. No mixed precision, quantization, CUDA backend, or external ML framework is included. This is the CPU reference policy; any future accelerated backend requires independent numerical parity first. |
| 11 | Work budget | The manual CPU workflow caps a run at 24 updates, 768 consumed/supervised tokens, and 8,000,000,000 cumulative work units. The earlier 24-update proposal already calculated 7,782,315,616 units before separate baseline and test work. The v2 full-split unchanged-weight baseline alone consumed 3,210,948,288 forward-work units. Adding that baseline to the prior proposal yields 10,993,263,904 units, exceeding the cap by 2,993,263,904 before recovery overhead. Therefore the 24-update/768-token proposal cannot be carried forward with this evaluation protocol; the experiment budget must be redesigned from accepted data and complete workload measurements. |
| 12 | Memory budget | Workflow ceilings are 64 MiB run-owned memory, 64 MiB autograd allowance, 4 MiB data allowance, and 16 MiB checkpoint size. The local two-update control observed application-owned peak 42,439,416 bytes and autograd-ledger peak 30,682,296 bytes; process high-water was measured separately (release test: 130,641,920 bytes). Application ledgers are not hard RSS limits. The exact full workload and selected provider have not been measured. |
| 13 | Runtime budget | Workflow caps are 120 seconds active run time and 90 seconds per attempt; the job timeout is 10 minutes and the outer process timeout is 240 seconds with a 30-second termination grace. Local two-update release control measured 18.913 supervised targets/second; debug measured 2.048. The v2 unchanged-weight release baseline took 11.60 seconds after compilation. These separate measurements do not validate a full training plus evaluation/checkpoint workload. The exact workload has not been measured three times on the candidate provider; the 120-second cap remains unverified. |
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
| D1 Objective A v2 candidate | `913d5e0b8b2994a7ec5dbfdc7cee7a47d8186c64` | [#28](https://github.com/samuelindra123/xynoos-cyber-model/pull/28) | Exact-head Foundation CI [45814498354](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45814498354), all 33 steps passed. The preceding run [45796238654](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45796238654) had only the stale 178-test count assertion; both full profiles passed with 180 checks, and the corrected assertion passed. |
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
  test-access reporting, a full-split v2 unchanged-weight baseline, and an
  independent split/tokenizer leakage audit. The v2 candidate remains too
  small and templated for admission.
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

- An adequate manifest-backed Objective A corpus or independently reviewed
  split/leakage analysis. The 48-row v2 candidate has 23 near-duplicate
  cross-split pairs above the fixed threshold and does not provide meaningful
  held-out generalization evidence.
- A hidden/locked test artifact suitable for first-run generalization
  measurement; v2's 12-row-per-family test split is public.
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
2. Replace or materially redesign the Objective A v2 candidate and obtain
   independent review of its manifest, tokenizer provenance, held-out family
   coverage, near-duplicate/leakage report, and test-access policy. Keep test
   data isolated from schedule choice. The current v2 corpus is a reproducible
   software control, not an admitted learning benchmark.
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
workflow, and exact-head CI evidence are prepared. The v2 baseline also shows
that the prior 24-update proposal exceeds the 8-billion-work ceiling when its
complete initial baseline is included. Current evidence does not establish
adequate scientific data, independent acceptance, private remote persistence,
or provider-specific full-workload fit. Therefore the first experiment is
**not dispatch-ready**. Complete the blockers above and issue a fresh
acceptance against the exact merged source and frozen run bundle before any
owner-authorized dispatch.
