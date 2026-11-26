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
| 1 | Accepted source SHA | **None yet.** The repository's current `main` is `32eca2b900c4f7a88294f7a24a5dfae0604ae798`. The engineering dependency PRs #20, #21, #23, #24, #25, #26, and #27, plus the sibling D1 Objective A v2 PR #28, are all open drafts and unmerged. Current candidate evidence is recorded in the milestone table below; none is an accepted training SHA. The latest v2 baseline ran on clean candidate source `4a83c68a199d8844e2d01038bd5d94d8dc377959`. A future run must bind to the exact clean source SHA after the owner reviews and merges the dependency chain. |
| 2 | Model configuration and parameter count | Native `Config::RESEARCH_CANDIDATE`: vocabulary 292, width 96, 8 layers, 4 heads, FFN width 256, context 128; **914,400 parameters**. Baseline seed was 107. The first-run seed and complete configuration are not frozen until the dataset and evaluation protocol are accepted. Weights are initialized by the project implementation; no pretrained model is used. |
| 3 | Tokenizer artifact and digest | Objective A v2 candidate uses `byte-bpe-v1`, 32 train-only merges, vocabulary 292, artifact SHA-256 `a5a6e350d0863af2a41aeae9a73a6e07d304513f243d293c79e91eb61f39bd09`. Its fit record and artifact are independently reproduced from the 24 training rows. These candidate identities must be copied into and re-attested by a final immutable run bundle. |
| 4 | Dataset manifest and digest | Candidate manifest-backed Objective A v2: dataset ID `2a5eda1b0668ee2ec9671b70b5becacc4e8479a28adec212853fd23d6d4d214c`; manifest SHA-256 `b3574cbcd769359ae4fc7ec4ff3f4df4f8fc5351cb004944826f87f9c38cfac1`. It has 48 project-authored synthetic rows (24 train, 12 validation, 12 public test). This improves identity and coverage over the two-row fixture but is **not admitted** as a sufficient experiment dataset. No external dataset was acquired. |
| 5 | Split and leakage report | The v2 audit finds no normalized exact cross-split duplicate or held-out operand-combination overlap. It finds **23 cross-split pairs above the fixed 0.82 character-trigram Jaccard threshold**, maximum 0.930233; all 23 are within the same task family and same prompt template, with zero high-similarity cross-family or cross-template pairs. Each split has 12 rows per task family. Repeated templates, small size, and a public test split make this candidate inadequate for an independent generalization claim. |
| 6 | Initial unchanged-weight baseline | On clean candidate source `4a83c68a199d8844e2d01038bd5d94d8dc377959`, seed 107, initial model digest `470fb79f5ec91c7e805115c19b90cb198586ff760b03ed7fafe4742f19f57a44`: train CE `5.696532329` (24 batches, 768 targets), validation CE `5.704625408` (12 batches, 384 targets), test CE `5.697565953` (12 batches, 384 targets). All losses were finite and the three reports share the same unchanged model-state digest. The test split was read and tagged, but did not enter tokenizer fitting, training, or optimizer state. These are baseline-generation checks on a narrow synthetic control, not evidence of generalization. |
| 7 | Multi-batch validation | Deterministic token-weighted evaluation now covers all 12 v2 validation batches and 384 supervised targets; the same baseline covers all 24 training batches and all 12 test batches. Ordering and split identities are reproducible. This candidate still has too few templated examples for a meaningful broad learning claim, and the final admitted dataset and experiment work limits remain undecided. |
| 8 | Locked test evaluation | Training admission excludes validation and test splits; evaluator reports tag test access. The v2 baseline read the full test split (12 batches, 384 targets), and the split is public. The independent BPE audit confirms all 48 answer ends align to token boundaries, but 8 answer starts fall inside merged tokens; current evaluation does not report answer-only loss or accuracy. This is not a hidden locked test artifact. A reviewed, frozen test-access policy and adequate held-out coverage are still required; no post-training result exists. |
| 9 | Optimizer and scheduler | Native AdamW and linear warmup/decay are implemented. The integration control uses AdamW `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`, weight decay `0.01`; its two-update schedule is a software test setting, not an accepted experiment choice. The earlier three-learning-rate sweep is retired. A single schedule must be predeclared against reviewed Objective A data before an experiment config is frozen. |
| 10 | Numerical precision policy | Current candidate uses `f32` parameters, gradients, and Adam moment inventories. The optimizer and schedule use `f64` for relevant intermediate accumulation, bias-power, and schedule calculations before narrowing at defined update boundaries; evaluation aggregates token-weighted losses in `f64`. No mixed precision, quantization, CUDA backend, or external ML framework is included. This is the CPU reference policy; any future accelerated backend requires independent numerical parity first. |
| 11 | Work budget | The manual CPU workflow caps a run at 24 updates, 768 consumed/supervised tokens, and 8,000,000,000 cumulative work units. The earlier 24-update proposal already calculated 7,782,315,616 units before separate baseline and test work. The v2 full-split unchanged-weight baseline alone consumed 3,210,948,288 forward-work units. Adding that baseline to the prior proposal yields 10,993,263,904 units, exceeding the cap by 2,993,263,904 before recovery overhead. Therefore the 24-update/768-token proposal cannot be carried forward with this evaluation protocol; the experiment budget must be redesigned from accepted data and complete workload measurements. |
| 12 | Memory budget | Workflow ceilings are 64 MiB run-owned memory, 64 MiB autograd allowance, 4 MiB data allowance, and 16 MiB checkpoint size. The two-update Objective A v1 control's hosted release sample reported application-owned peak 42,439,416 bytes, autograd-ledger peak 30,682,296 bytes, and process high-water 121,409,536 bytes; a local sample reported process high-water 130,641,920 bytes. Application ledgers are not hard RSS limits. These are tiny regression controls, not the full v2 experiment. |
| 13 | Runtime budget | Workflow caps are 120 seconds active run time and 90 seconds per attempt; the job timeout is 10 minutes and the outer process timeout is 240 seconds with a 30-second termination grace. The Objective A v1 two-update release control measured 18.913 supervised targets/second locally and 46.104 on one hosted Actions run; debug measured 2.048 locally. The v2 unchanged-weight release baseline took 10.82 seconds after compilation and used 3,210,948,288 forward-work units. These separate measurements do not validate a full training plus evaluation/checkpoint workload. The exact workload has not been measured three times on the candidate provider; the 120-second cap remains unverified. |
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
| D1 Objective A v2 candidate | `56cecdc3df55a999a3966ee55c4ee352d66b6cc4` | [#28](https://github.com/samuelindra123/xynoos-cyber-model/pull/28) | Exact-head Foundation CI [45824025307](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45824025307) passed all 33 steps and 181 debug/release checks. |
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
