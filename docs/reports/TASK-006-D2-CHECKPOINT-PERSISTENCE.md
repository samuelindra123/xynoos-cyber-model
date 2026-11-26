# TASK 006-D2 — Private Checkpoint Persistence and Recovery

**Status: prepared; private off-host persistence is not yet verified.** This
milestone hardens the local artifact boundary and specifies a recoverable
package and transfer protocol. No model weights were uploaded, no external
storage was provisioned, and no provider credential or account setting was
changed.

## Implemented and verified locally

- The CLI now rejects an artifact-root directory if its owner-only permission
  bits are not `0700` on the supported Unix runtime.
- New checkpoint, journal, metrics, report, stop-request, process-identity,
  and handoff files are created owner-only (`0600`). Existing task-006
  checkpoint publication remains sibling-staged, SHA-256 protected, file
  synced, atomically renamed, and followed by a directory sync.
- `inspect-checkpoint` validates the checkpoint checksum, experiment/config
  identity, and checkpoint semantics before printing its digest and counters.
- The exact D2 report inventory is a **recovery set**, not just the checkpoint:

| Artifact | Role | Required for resume | Integrity or handling |
|---|---|---:|---|
| `experiment.json` | Immutable experiment identity and artifact digests | Yes | Preserve byte-for-byte |
| `run.journal` | Monotone cumulative work, runtime, and checkpoint sequence | Yes | Preserve with the checkpoint |
| `checkpoint.xyckpt` | Parameters, optimizer, scheduler, RNG, counters, and cursor | Yes | SHA-256 verified by the codec and CLI |
| Immutable training config | Model, optimizer, scheduler, budgets, and paths | Yes | Keep with its exact source revision |
| Dataset manifest and payloads | Ordered training/validation/test identity | Yes | Preserve the admitted dataset snapshot |
| Tokenizer artifact and fit record | Tokenizer identity and fit provenance | Yes | Preserve exact bytes and digests |
| `metrics.jsonl` | Structured progress and evaluation events | No | Preserve for audit |
| `final-report.json` or `failure-report.json` | Completion or failure outcome | No | Preserve for audit |
| `handoff.json` and `artifact-integrity.json` | Run identity, budgets, and post-run digest summary | No | Preserve when present |
| `.run.lock`, `process.pid`, `stop.request` | Local process lifecycle state | No | Never restore an active process identity; see recovery steps |

The checkpoint format caps its file at 16 MiB. Dataset and tokenizer limits
remain those of the immutable experiment configuration and manifest; the
external backup must also bound total bytes before transfer.

## Storage evaluation

The repository is public. GitHub Actions artifacts are available to signed-in
users with repository read access, and the current training workflow uploads
bounded logs/reports for three days while explicitly excluding checkpoint
weights. They are not a private checkpoint destination. [GitHub artifact
access](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/download-workflow-artifacts)
· [workflow retention controls](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-your-repository)

| Option | Observed support and access requirements | Decision |
|---|---|---|
| Owner-controlled local directory | Implemented, mode-checked, atomically checkpointed, and resumable after tested process crashes. It survives only the local storage failure domain. | Use for bounded local validation; insufficient as the sole recovery copy. |
| GitHub Actions artifact or Release | Anyone with repository read access can download artifacts; the repo is public. Release assets are also public. Current workflow omits checkpoint weights and keeps only small reports for three days. | Rejected for private weights. Never publish a checkpoint here. |
| Colab VM filesystem | Installed CLI is `0.8.0`; its `upload`/`download` commands exist, and `run` releases the VM unless `--keep` is used. No VM, transfer, disconnection, durable storage, or cleanup exercise was run. | Ephemeral execution storage only; not an accepted checkpoint store. |
| Google Drive through Colab | Drive mounting requires runtime authorization and access to files in the mounted Drive; Drive operation quotas can cause I/O failures. No mount or permission review was performed. | Candidate only after owner confirms sharing scope and authorizes a transfer exercise. |
| Google Cloud Storage | IAM can restrict bucket/object access; recommended controls include uniform bucket-level access and least-privilege roles. Creating/configuring a bucket needs project permissions; object copy/replace needs object permissions. Project, billing, lifecycle, encryption, IAM, credentials, and costs are not verified. | Preferred remote candidate if an owner-controlled project and budget are approved. No bucket or credential was created. |

References: [Colab FAQ](https://research.google.com/colaboratory/faq.html),
[official Colab CLI command reference](https://github.com/googlecolab/google-colab-cli),
[Cloud Storage IAM](https://docs.cloud.google.com/storage/docs/access-control/using-iam-permissions),
[Cloud Storage access-control overview](https://docs.cloud.google.com/storage/docs/access-control).

The Google Cloud Storage design should use a dedicated bucket or prefix,
uniform bucket-level access, public-access prevention, a narrowly scoped
service identity, immutable object names per experiment/checkpoint sequence,
and a lifecycle rule approved for the intended retention. The minimum role
must support creating and reading the selected objects; replacing objects
also requires delete permission. Storage, request, and transfer costs must be
checked against an owner-approved budget before enabling it.

## Recovery and interrupted-transfer procedure

1. Confirm the prior process is stopped. Run `xynoos-train status` against the
   artifact root and require that it does not report `active`; for a detached
   process, also confirm its OS process is gone. Never restore a live process
   identity.
2. Create a new local destination with `umask 077` and mode `0700`. The CLI
   rejects a broader artifact-root mode. Do not place model files in the Git
   checkout, a public Actions artifact, or a public release.
3. Copy the full recovery set into a **new staging directory**. Do not resume
   from a partially copied directory. Preserve `experiment.json`,
   `run.journal`, checkpoint, config, data manifest/payloads, tokenizer, and
   fit record together. If an upload is interrupted, discard or quarantine
   the incomplete staging prefix and restart into a fresh prefix.
4. Before publishing a transferred set, compare SHA-256 digests for every
   inventoried file to the source inventory. For object storage, upload to a
   unique temporary prefix, verify uploaded object size and digest, then
   publish a small immutable completion manifest last. Never overwrite the
   last verified recovery set in place.
5. Atomically rename the verified local staging directory into its final
   private location on the same filesystem, then sync the directory. For a
   remote provider without atomic directory rename, the completion manifest
   is the publication marker; readers must ignore prefixes without it.
6. Run `xynoos-train status --config <config> --artifact-root <root>` and
   `xynoos-train inspect-checkpoint --config <config> --artifact-root <root>`.
   Verify experiment ID, source revision, run/config digest, dataset digest,
   tokenizer digest, checkpoint SHA-256, checkpoint sequence, and counters
   against the accepted inventory before resume.
7. If `stop.request` exists, confirm the old owner is gone, remove that stale
   request, and sync the artifact directory before resume. Keep the persistent
   `.run.lock` inode; do not delete a lock file to recover ownership. Resume
   with the same immutable config and lineage. If the journal, sidecar,
   checkpoint, or identity disagrees, retain the bytes for diagnosis and do
   not resume.

The protocol relies on provider checksum behavior only as a transfer check;
the local SHA-256 validation and checkpoint decoder remain authoritative.
Provider-side interrupted-upload recovery has **not** been tested.

## Retention

Retain the complete recovery set and one independently verified private copy
until the experiment has passed its scientific and engineering acceptance
review, then retain both for at least 30 days. Retain any accepted final model
only under a separately approved owner retention decision. Keep failed-run
diagnostics until the failure has been reviewed; then remove incomplete
staging data and failed checkpoints from every storage location. These are
the proposed project defaults, not an active remote lifecycle policy.

## Remaining owner action and limits

Off-host recovery remains blocked until the owner selects an existing private
destination or authorizes a provider setup, confirms access controls and
retention, and approves any associated billing/egress. After that, a bounded
non-training transfer exercise must verify upload, interruption, digest
comparison, restore, checkpoint inspection, resume, and cleanup. Until this
exercise passes, do not claim private remote checkpoint persistence or
detached-runtime recovery.
