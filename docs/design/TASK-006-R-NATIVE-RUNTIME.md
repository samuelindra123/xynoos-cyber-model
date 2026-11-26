# TASK 006-R — Native Training Runtime

`xynoos-train` is the bounded native entry point in `xynoos-training`. It constructs only the accepted `xynoos-cyber-0.1` model and admits only the standard manifest-backed, fitted-tokenizer stream. Its execution code has no Python, AI-agent, or external ML-framework dependency. The binary and detached-process contract currently targets Linux/POSIX systems with `/proc`, `/dev/null`, `setsid`, same-directory atomic rename, and file/directory sync support.

## Commands

Every command requires an existing absolute artifact-root directory. Unknown commands, arguments, duplicate flags, configuration fields, source kinds, and invalid identities or caps fail with exit code `2`. Successful validation and safe runtime stops return `0`; runtime errors return nonzero after attempting to preserve a failure report. A completely unavailable artifact filesystem can prevent report persistence, in which case stderr carries the diagnostic.

```text
xynoos-train validate --config CONFIG.json --artifact-root ABSOLUTE_DIR
xynoos-train run --config CONFIG.json --artifact-root ABSOLUTE_DIR [--detach]
xynoos-train resume --config CONFIG.json --artifact-root ABSOLUTE_DIR [--detach]
xynoos-train status --config CONFIG.json --artifact-root ABSOLUTE_DIR
xynoos-train stop --artifact-root ABSOLUTE_DIR
xynoos-train inspect-checkpoint --config CONFIG.json --artifact-root ABSOLUTE_DIR
```

`validate` checks the bounded configuration, dataset manifest and payload identities, train-bound tokenizer fit, split ordering identities, retained-source reservations, paths and current artifact-root identity. It does not construct the Transformer. `run` requires an empty artifact root and creates an immutable experiment sidecar before admission. `resume` requires that sidecar, a compatible checkpoint, and the existing valid lineage journal; it never creates replacement lineage counters. `status` performs bounded read-only checks. `stop` creates `stop.request`; the runtime checks it at safe boundaries. The request remains latched in the artifact directory, so an owner who wants to resume after a graceful requested stop must remove only `stop.request` first. This does not alter the checkpoint or journal counters. `inspect-checkpoint` verifies and decodes bounded checkpoint metadata without constructing a Transformer or changing artifacts.

`run --detach` and `resume --detach` spawn the current executable through `setsid`, direct stdin/stdout away from the controller and append stderr to `process.stderr`. A short-lived `process.pid` handshake lets the launcher return the runtime process PID even when `setsid` forks. The runtime removes that PID file on orderly exit and replaces a stale file after a crash when its PID is no longer live. The child owns subsequent logs, checkpoints, stop checks, failure reports and cleanup. Tests verify that the detached child has its own Linux session and continues after its launching controller exits; this is not a Colab CLI, SSH, remote-filesystem, or provider-runtime persistence claim.

## Immutable JSON configuration schema v1

The UTF-8 JSON file is limited to 16 KiB. Every listed key is required; there are no unlimited defaults. Unknown keys are rejected at every object level. Relative data paths resolve against the config file; absolute paths are accepted by the local CLI. All paths reject parent traversal and symlink components. The manual GitHub workflow requires relative data paths below the configuration directory.

| Field | Required value or constraints |
|---|---|
| `schema_version` | Integer `1` |
| `experiment_id` | Immutable 1–96-character ASCII identifier beginning with a letter or digit; remaining characters limited to letters, digits, `.`, `_`, `-` |
| `source_revision` | Lowercase 40-character Git commit SHA; the manual workflow additionally requires exact equality with the checked-out commit |
| `source_kind` | Exactly `manifest-v1-stream` |
| `dataset_root` | Standard dataset directory containing the canonical manifest and indexed payloads |
| `tokenizer_artifact`, `tokenizer_fit_record` | Existing regular files; fit record must bind this tokenizer to this dataset’s training split |
| `dataset_id`, `dataset_manifest_sha256`, `tokenizer_sha256` | Exact admitted identities; each digest is lowercase SHA-256 where applicable |
| `optimizer` | Exactly `beta1`, `beta2`, `epsilon`, `weight_decay`; validated by the existing AdamW constructor |
| `schedule` | Exactly `total_updates`, `warmup_updates`, `peak_rate`, `final_rate`; validated by the existing one-based schedule |
| `limits` | Exactly `microbatches_per_update`, `max_updates`, consumed/supervised token caps, cumulative work, owned-memory bytes, cumulative and per-attempt runtime seconds, checkpoint cadence and validation cadence; each positive and within existing project maxima |
| `memory` | Exactly `data_allowance_bytes`; checked with the existing `TrainingMemoryPlan` |
| `autograd` | Exactly byte, value, operation and work caps; values may not exceed existing `ResourceLimits` |
| `gradient_clip` | `null` or a finite positive number |
| `seed` | Unsigned 64-bit integer |

The model ID, architecture, 914,400-parameter/74-tensor inventory, vocabulary 292, context and training shape `[1,32]` are fixed by the executable; they cannot be overridden by configuration. Dataset and tokenizer contents, the train/validation split and stream ordering are revalidated before model construction. Standard sources are prepared before the Transformer and checked against the combined memory allowance for simultaneously retained train and validation streams. Custom in-process `BatchSource` implementations remain trusted research extensions, not an adversarial memory boundary.

## Artifact behavior and boundaries

| Artifact | Purpose |
|---|---|
| `experiment.json` | Synced immutable experiment/config identity sidecar |
| `run.journal` | Configuration-bound cumulative resource and attempted-work lineage state |
| `.run.lock` | PID lock rejects simultaneous live attempts; stale PID cleanup is Linux `/proc` based |
| `metrics.jsonl` | Structured identity-bearing progress and lifecycle events |
| `checkpoint.xyckpt` | Canonical bounded checkpoint v1; atomic local publication |
| `handoff.json` | Experiment identity, caps, artifact locations and stop instructions |
| `final-report.json` | Final counters, stop reason, memory and timing observations; failure details when available |
| `failure-report.json` | Separate startup/recovery failure detail; does not overwrite the prior final report or checkpoint |
| `process.stderr` | Detached child stderr, when using `--detach` |
| `process.pid` | Short-lived detached child PID handshake; stale PID is checked before reuse |
| `stop.request` | Latched safe-boundary stop request |

The lineage journal’s unkeyed SHA-256 detects accidental corruption but does not authenticate the artifact directory. Local artifact storage is trusted. Symlink checks have TOCTOU limits; hostile directory races, remote filesystems and power loss beyond requested sync operations are unsupported. Runtime and shutdown checks are safe-point based. Synchronous kernels and filesystem operations can exceed a nominal runtime deadline.

Application owned-memory and autograd ledger values are accounting reservations, not a hard process-RSS limit. Process `VmHWM`, when available on Linux, is reported separately. The manual workflow keeps no public checkpoint artifact and deletes ephemeral runner state; an actual experiment needs separate owner authorization for private checkpoint persistence and retrieval.
