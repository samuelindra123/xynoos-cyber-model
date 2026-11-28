# TASK 006-F — Checkpoint and Recovery Verification

**Outcome: local checkpoint integrity and deterministic resume pass; private off-host persistence and provider-disconnect recovery are unverified.** No model weights were uploaded.

## Local integrity and resume evidence

TASK 006-D2 defines a canonical, bounded checkpoint (maximum 16 MiB) containing parameter and optimizer identities, moments, scheduler/bias-power progress, RNG, data/tokenizer identity, cursor/order, and counters. Encoding is canonical and SHA-256 protected. Decode streams through integrity checks and performs semantic validation, including negative moment rejection. Publication stages a sibling file with private permissions, syncs it, renames atomically, and syncs the directory. Journal and checkpoint are a joint recovery set; the journal is not reconstructed when missing.

Local D2 tests cover truncated/corrupt bytes, digest mismatch, identity and semantic mutations, publication fault boundaries, and exact resume equivalence against uninterrupted execution (parameters, both Adam moments, bias powers, scheduler, cursor/order, next batch, and deterministic metrics). The F journal controls additionally verify corrupted and missing journals fail closed and SIGKILL resume preserves prior resource charges. D2 exact-head hosted CI passed in [run 45784299844](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45784299844); exact F hosted CI is pending.

## Finding F-CKPT-01 — local checkpoint recovery

- **Severity:** High; locally verified for tested process/filesystem behavior.
- **Component:** Checkpoint codec, atomic publication, identity binding, journal and deterministic resume.
- **Reproduction:** `cargo test -p xynoos-training --locked --offline -- --show-output`; then run F runtime controls in debug/release. Inspect D2 test evidence for mutation and trajectory cases.
- **Evidence:** D2 package tests and C3/F journal controls pass; the checkpoint format validates SHA-256 and semantics and resumes to the same deterministic state as uninterrupted execution in covered cases.
- **Root cause:** No local format or replay defect was identified in the covered cases.
- **Remediation:** Preserve all recovery-set files byte-for-byte and use the documented staging/inspection procedure; do not reconstruct a missing journal or restore live process metadata.
- **Verification:** Local control passed. Full integrated exact-head hosted CI is required for release assurance.

## Finding F-CKPT-02 — no verified private remote recovery domain

- **Severity:** Critical for training artifact recoverability.
- **Component:** Private checkpoint backup, interrupted transfer, restore, and provider disconnection.
- **Reproduction:** After explicit owner selection of a private store and budget, run the bounded non-training upload/interruption/digest/restore/inspect/resume/cleanup exercise.
- **Evidence:** No destination/account permissions were changed; GitHub Actions artifacts and repo are public; Actions workflow explicitly excludes weights; Colab filesystem/Drive/GCS lifecycle was inspected only; no remote write, download, or provider-disconnect test occurred.
- **Root cause:** There is no owner-selected private storage target or authorized provider persistence exercise.
- **Remediation:** Owner selects an existing private destination, access scope, retention, encryption, and cost/egress budget; then verify complete recovery set with digest and restore before any training.
- **Verification:** Not performed. Claims of off-host persistence, detached survival, remote resume, and cleanup are prohibited until it passes.

## Recovery procedure and retention proposal

Preserve `experiment.json`, journal, checkpoint, immutable config, dataset manifest/payloads, tokenizer and fit record together. After confirming all prior processes are gone, copy to a new private staging location, verify all SHA-256 values and sizes, inspect the experiment/checkpoint, then publish a completion marker or same-filesystem rename. Resume only with exact source/config/data identity. Keep the persistent `.run.lock` inode; never restore active process identity. If any identity, digest, semantic check, or counter disagrees, quarantine bytes and stop.

D2’s proposed retention is the full recovery set plus one independently verified private copy until acceptance review and at least 30 days afterward, with failed-run diagnostics retained until reviewed. This is not an active retention policy until owner-approved storage lifecycle rules exist.
