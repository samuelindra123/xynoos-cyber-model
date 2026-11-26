# TASK 006-D3 — Compute and Training Readiness

**Status: provider capabilities inspected; no training provider is approved or
verified for detached execution.** No training workflow was dispatched, no
Colab runtime was created, and no compute units were consumed.

## GitHub Actions

The repository visibility is `PUBLIC`. The normal Foundation workflow uses
`ubuntu-24.04`, `CARGO_BUILD_JOBS=2`, and a 15-minute job timeout. The exact D2
Foundation run completed in 2 minutes 8 seconds. This measures validation-job
wall time, not training throughput. The manually dispatched CPU workflow has
a 10-minute job timeout, a 240-second `timeout` wrapper with a 30-second TERM
grace period, and cleanup under `if: always()`. Its concurrency group does not
cancel an earlier training job. Validation creates an ephemeral mode-`0700`
artifact root; the workflow removes runner-local run state at completion.

GitHub documents a standard public Ubuntu runner as a new VM with 4 vCPUs,
16 GB RAM, and 14 GB SSD; standard public-repository usage is listed as free
and unlimited. GitHub also documents a six-hour maximum job execution time.
Those published limits do not guarantee a particular training throughput,
available disk after tool installation, or wall time under runner variability.
No dispatched training job exists from which to measure those values.
[Runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
· [Actions limits](https://docs.github.com/en/actions/reference/limits)

Actions cancellation sends an interrupt, then a termination signal, and can
kill the process tree; the server can forcibly terminate a cancelling job
after five minutes. `if: always()` allows cleanup steps to continue when
possible, but is not a durable artifact mechanism after forced termination.
[Cancellation semantics](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-cancellation)

The training workflow uploads up to 2 MiB of logs and reports for three days
and explicitly records `model_weights_retained_as_upload: false`. Because the
repository is public, artifacts are accessible to repository readers. No
weights are stored in the runner artifact service, and the ephemeral runner
filesystem is not a recovery store. [Artifact access](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/download-workflow-artifacts)

The exact D1 candidate Foundation CI run `45823059589` also exercised the
bounded two-update Objective A v1 software control on `ubuntu-24.04`: the
release test reported 46.104 supervised targets/second, application-owned
peak 42,439,416 bytes, autograd-ledger peak 30,682,296 bytes, and process
high-water 121,409,536 bytes. This is one short regression sample on the
hosted runner, not a measurement of the v2 candidate training workload or a
throughput guarantee.

## Colab CLI and provider state

Read-only inspection used the installed CLI's `colab version` command and
reported version `0.8.0`. Its actual `new` and `run` help says CPU is the
default when no accelerator is requested; the current CLI lists GPUs T4, L4,
G4, H100, and A100, and TPUs V5E1 and V6E1, subject to availability. Help was
also inspected for `exec`, `ssh`, file `upload`/`download`, `sessions`,
`status`, `stop`, and `usage`. `colab sessions` reported no active sessions.
`colab usage` was read-only and showed no active assignments; account balance
details are intentionally not recorded in the repository. No runtime was
created.

The installed CLI's `run` command accepts a local **Python script**, creates
a fresh VM, and releases that VM after completion unless `--keep` is used.
`ssh -- CMD` runs against an existing session and exits when the remote command
ends; an interactive bare `ssh` may create a session if none exists, so neither
form was invoked. The CLI documents `--rm` for an auto-created interactive
session or proxy-mode disconnect, but says it is ignored for one-shot `-- CMD`.
The official SSH guidance warns that interrupting the local SSH command can
leave the remote command running. Colab also states that VMs can be deleted
after idle time and have a service-enforced maximum lifetime. `upload` and
`download` move files to/from a session filesystem, but the CLI inspection did
not demonstrate checksummed, atomic checkpoint persistence.
The XYNOOS runtime is native Rust; running it on Colab would require a
separately tested remote build/command path, data transfer, persistence, and
cleanup protocol.
[CLI command reference](https://github.com/googlecolab/google-colab-cli)
· [SSH lifecycle notes](https://github.com/googlecolab/google-colab-cli/blob/main/docs/06_ssh_access.md)
· [Colab runtime FAQ](https://research.google.com/colaboratory/faq.html)

The Colab usage/quota inspection is not a spending authorization or a
reserved compute budget. Accelerator availability, usage-unit consumption,
transfer time, and backend termination are uncertain until a separately
authorized run. Do not infer that authentication proves detached execution,
durable checkpoints, or cleanup.

## Measured software workload and limits

The only repeated-training measurement uses the two-row Objective A software
fixture (two training rows, two validation rows, 64 supervised targets and
two optimizer updates). At clean revision
`0b798e91b550f30d37683eab2c6bf66382a88c28`, the same test code reported
2.048 targets/second in the debug profile and 18.913 targets/second in the
release profile. Both runs reported the same 0.140678167 validation-loss
reduction, 42,439,416-byte owned-memory peak, and 30,682,296-byte autograd
ledger peak. The debug test process high-water was 133,312,512 bytes; the
release test process high-water was 130,641,920 bytes. These are local
software-test observations, not a larger-dataset benchmark or an
Actions/Colab provider guarantee.

The profile difference is material. A linear 768-target projection is about
375 seconds using the local debug rate and 40.6 seconds using the local
release rate. The hosted release sample is 46.104 targets/second, but it is
the same two-update v1 fixture, not a full Objective A v2 workload. These
projections omit the exact 24-update cadence, eight validation passes, five
checkpoints, initialization, baseline and test evaluations, and runner
variability. The full candidate configuration has not been timed three times
on the hosted runner, so the current 120-second runtime cap is not yet shown
to fit or fail.

The earlier TASK 006-R optimized measurement of about 40 targets/second came
from a different source revision and evaluation setup. It is retained as
historical evidence, not combined with the post-D1 measurements. The R2/C3
reports also distinguish process high-water from application memory ledgers;
the larger local test-process high-water values above are not hardware
specifications or hard RSS enforcement.

A separate work-budget calculation shows that a 24-update proposal consumes
7,782,315,616 of the 8,000,000,000 cumulative-work ceiling:

| Component | Estimated work |
|---|---:|
| 24 candidate updates | 6,898,429,152 |
| Two-batch validation at configured cadence | 800,000,384 |
| Five checkpoint publications | 83,886,080 |
| **Total** | **7,782,315,616** |
| **Remaining headroom** | **217,684,384 (2.72%)** |

One additional validation batch at each validation point adds 400,000,128
work and exceeds the cap. The proposed 24 updates/768 targets are therefore a
ceiling calculation only. They are not a selected experiment, a claim of
scientific sufficiency, or evidence of cybersecurity learning. Current
throughput has not been measured repeatedly on the candidate provider with
the proposed full workload. It does not justify increasing the runtime cap
or switching to a GPU.

The Objective A v2 unchanged-weight full-split baseline itself requires
3,210,948,288 forward-work units. Combining it with that retired 24-update
proposal yields 10,993,263,904 units, exceeding the 8-billion ceiling by
2,993,263,904 before other recovery costs. The earlier sweep proposal cannot
be used; no update schedule is admitted until the dataset and held-out
evaluation are accepted and the full workload is re-budgeted.

## Deterministic workload-admission policy

Before a future run can be admitted, all of these gates must pass:

1. **Scientific input gate:** use a frozen manifest-backed Objective A set
   with a reviewed split/leakage report, adequate held-out family coverage,
   and an unchanged-weight baseline. Keep Objective B out of scope until its
   provenance, licenses, and quality are reviewed. A test split used for the
   initial baseline stays locked and cannot select the learning rate.
2. **Identity gate:** source repository/revision and clean-tree attestation,
   config digest, dataset-manifest digest, tokenizer digest, model identity,
   and evaluation policy must all match the immutable experiment record.
3. **Work gate:** count updates, training, validation, checkpoints, and
   recovery against the full-run budget. For the current Actions profile the
   hard ceilings are 24 updates, 768 consumed/supervised tokens, 8 billion
   cumulative work units, 64 MiB owned memory, 64 MiB autograd allowance,
   4 MiB dataset allowance, 120 seconds active runtime, 90 seconds per
   attempt, and 16 MiB per checkpoint. These are software caps, not proof of
   scientific adequacy.
4. **Measurement gate:** benchmark the exact model, source, and data path at
   least three times in the candidate provider's bounded CPU environment.
   Record each run's tokens/second, full-run wall time, work, owned and
   autograd peaks, process RSS/high-water mark, checkpoint time, and outcome.
   Admit only if every measurement is finite, completes within the configured
   runtime and work caps, respects both memory ledgers and the provider's
   documented RAM, and leaves the private recovery set intact. Use the
   slowest observed run for the admission decision; do not extrapolate a
   single run upward to claim capacity.
5. **Persistence gate:** prove private checkpoint backup, transfer digest
   verification, restore, checkpoint inspection, and resume using a
   non-training exercise before selecting a remote provider.
6. **Provider gate:** keep GitHub Actions as the default for CI, oracles,
   regression, and a small CPU learning control when all preceding gates fit.
   Consider Colab only if measured CPU limits prevent the scientifically
   required workload. Select an accelerator only if the project has a native
   backend for it, independent precision-appropriate parity, a measured
   benefit, and an authorized resource budget. The current native runtime is
   CPU-only; no GPU backend is admitted.
7. **Authorization gate:** workload admission never dispatches a job. Owner
   authorization for the exact immutable configuration and provider remains
   a separate requirement.

Current D1 data fails the scientific-input gate. The available throughput
measurements are tiny fixture tests from different profiles/revisions, not
three repetitions of the candidate workload on the provider. Private off-host
persistence and remote detached lifecycle also fail their gates. Consequently
no first-training provider is selected yet. Actions remains the default for
bounded software verification; Colab is only a future candidate after those
blockers and the required owner authorization.

## Evidence classification

| Category | Evidence |
|---|---|
| Implemented | Bounded CPU workflow caps, manual boolean authorization input defaulting to false, timeout/cleanup, owner-only artifact roots, local stop/recovery code |
| Executed | Local Objective A software control and resource telemetry; exact-hosted D2 Foundation CI; read-only local Colab CLI help, quota, and session inspection |
| Inspected only | Published GitHub runner, cancellation, artifact-access, Colab CLI/SSH, and Colab FAQ documentation |
| Not verified | Colab runtime creation, process survival after CLI disconnect, Rust transfer/build path, uploaded-byte digest validation, private remote persistence, restore, runtime cleanup, or actual training-provider throughput |
| Not authorized/performed | Training dispatch, Colab runtime creation, GPU/TPU use, paid-unit consumption, external dataset acquisition |
