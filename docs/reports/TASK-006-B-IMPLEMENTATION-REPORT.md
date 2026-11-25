# TASK 006-B — Native Training Implementation Report

**Status:** implementation complete; draft review requested
**Branch:** `feat/task-006-b-native-training-engine`
**Base:** TASK 006-A design head `26777fa95383600c003ff094216d3d1c310b051c`
**Date:** 2026-11-26

## Delivered

Added the `xynoos-training` crate and connected bounded training to the existing
Transformer, tokenizer, dataset, and autograd APIs. AdamW and the weighted
gradient accumulator live in `xynoos-autograd`; existing SGD remains intact.
The trainer runs the fixed candidate model and inventory through a separate
finite training policy, with the required `B=1,T=32` microbatch and checked
update, token, cumulative-work, owned-memory, and runtime limits. Test runs use
two updates; the 10,000-update value is only a hard policy ceiling.

AdamW validates the complete transaction before publication, stages FP32
moments, uses the frozen FP64 bias-correction arithmetic, applies decoupled
decay once to rank-2 matrices (including the tied embedding), and excludes
rank-1 gains. The transaction publishes parameters, moments, powers, step,
revision, and gradient clearing together. Rejected updates preserve committed
and gradient bits. Ordered weighted sums are accumulated in FP64, normalized
once to FP32, then globally clipped. The independent Python oracle still has
its original 31 records; two independently derived controls were appended for
update-order and tied-embedding double-decay errors. The native checks pass all
33 records. Current fixture SHA-256:

`f1839d7c12a0c9b3b0e75b73ec3b7e53b1eb1c67191c64e6053eb9a199244e61`

The one-based warmup/decay schedule is persisted and deterministically derived
from completed updates. The run lifecycle preflights full batches and resource
charges before forward work, commits scientific counters only with a complete
optimizer update, and discards incomplete windows. The lineage journal retains
admitted work across crashes and resumes. Runtime is capped both per process
attempt and cumulatively across the lineage; offline time is excluded. Attempt
runtime starts before model initialization or checkpoint loading and continues
through finalization.

The run-owned memory plan is capped at 128 MiB and reserves compatible child
budgets before model allocation. The data/tokenizer allowance is at most
64 MiB. Peak charged run-owned bytes, the autograd ledger, and process `VmHWM`
are reported separately. The autograd ledger does not account for data or
tokenizer allocations; those are covered by the training run budget and the
source/tokenizer admission checks.

Checkpoint v1 is specified in
[`TASK-006-CHECKPOINT-V1-FORMAT.md`](../design/TASK-006-CHECKPOINT-V1-FORMAT.md)
and implemented as a bounded streamed codec. It uses a 64 KiB buffer, a 16 MiB
file cap, canonical little-endian typed fields, and SHA-256. Loading validates
framing and integrity in a first pass, then decodes into private typed staging
and publishes only after compatibility and state checks pass. Checkpoints store
only committed cursors and scientific token counts. An interrupted window is
replayed from its last committed cursor; attempted/discarded counters and
cumulative resource charges remain in the lineage journal.

Publication writes and syncs a create-new sibling temporary file before
same-directory rename. Pre-rename failure preserves the old checkpoint. If
parent-directory sync fails after rename, the call reports
`PublicationDurabilityUncertain`; the replacement has already been published.
The documented assumptions are an ordinary local POSIX filesystem with atomic
same-directory rename and file/directory sync support. Path-component TOCTOU
races, remote filesystem behavior, storage-controller loss, and power-loss
behavior beyond those sync requests remain outside the guarantee.

The runtime support writes structured JSONL metrics, a final machine-readable
report, a deterministic handoff file, checkpoint and stop-request artifacts,
and cleans up the trainer at safe boundaries. Stop requests are polled at
bounded safe points; a blocked filesystem operation cannot be interrupted
immediately. No long-running job is included in this milestone.

## Objective A fixture

The frozen fixture is project-authored, educational systems-language material
with no external corpus or private data. It uses the existing byte-BPE-v1
tokenizer artifact and vocabulary 292 without refitting. The generator freezes
seed `0x006B`, sample order, the candidate rate list, the fixed integration
rates, split digests, batch construction, and evaluation procedure. It records
zero exact normalized duplicates, a held-out combination validation split, and
a held-out template-family test split. Character-trigram Jaccard at 0.82 is a
surface-overlap check only; it cannot detect semantic paraphrases or establish
independent reasoning. The test-family rows are not used to tune or select the
two-update integration run.

Fixture SHA-256:

`63050baf97f301c5ca6a3ad07a1a632bef77938b69f0317622738b5bbc00fa33`

The fixed integration made two genuine Transformer/AdamW updates. Its
held-out-combination mean cross-entropy changed from `5.695174217` to
`5.554496050` (delta `-0.140678167`) over 64 supervised training tokens. The
tied embedding parameter and both moment arrays were checked for nonzero
updates. This demonstrates a bounded controlled learning signal only; it is
not evidence of cybersecurity competence or domain intelligence.

## CPU reference measurements

Measurements below came from isolated, serial, optimized-profile Objective A
and resume tests on 2026-11-26. They are observations from this test environment,
not generic hardware claims. Update and validation phases have two samples;
checkpoint writes and direct codec loads have three. Dispersion is sample
standard deviation. For one-sample stages, dispersion is not estimable. Fixture
preparation is one interval covering the four train/validation rows.

| Stage | Samples | Mean ms | Sample SD ms | Min–max ms |
|---|---:|---:|---:|---:|
| Trainer initialization | 1 | 28.549 | — | 28.549 |
| Dataset batch acquisition | 2 | 0.001668 | 0.000426 | 0.001367–0.001969 |
| Forward | 2 | 151.386 | 4.671 | 148.083–154.689 |
| Backward | 2 | 179.341 | 5.477 | 175.468–183.214 |
| Gradient transfer | 2 | 0.142 | 0.003 | 0.139–0.144 |
| Weighted accumulation | 2 | 4.360 | 0.226 | 4.200–4.520 |
| Normalization and clipping | 2 | 13.202 | 1.700 | 12.000–14.404 |
| AdamW staging and commit | 2 | 59.251 | 1.449 | 58.226–60.275 |
| Full optimizer step | 2 | 410.026 | 0.303 | 409.812–410.240 |
| Validation | 2 | 158.775 | 7.357 | 153.572–163.977 |
| Checkpoint write | 3 | 30.887 | 2.047 | 28.527–32.180 |
| Direct checkpoint codec load | 3 | 71.565 | 8.679 | 63.388–80.671 |
| Full trainer checkpoint resume | 1 | 84.652 | — | 84.652 |
| Trainer shutdown | 1 | 2.313 | — | 2.313 |

The single fixture-preparation interval was `0.144147 ms`. End-to-end
supervised throughput was `39.710 tokens/s` for this two-update test, including
its bounded run lifecycle; it is not a sustained-training estimate.

| Memory measure | Objective A test | Resume test |
|---|---:|---:|
| Charged whole-run owned peak | 42,439,416 B | — |
| Charged persistent owned bytes after run | 22,744,448 B | 22,744,448 B |
| Autograd ledger peak | 30,682,296 B | — |
| Autograd ledger after run | 18,314,232 B | 18,314,232 B |
| Process high-water (`VmHWM`) | 35,287,040 B | 57,167,872 B |

The direct codec-load measurement repeats load three times and includes private
staging and full parameter reconstruction. The resume process high-water also
includes the test's uninterrupted, partial, direct-load, and resumed paths;
`VmHWM` is therefore reported separately and must not be read as the charged
owned-allocation peak. Repeated-step accounting returns to the expected
persistent baseline after each run.

## Verification

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets --locked --offline -- -D warnings`
- `cargo test --workspace --locked --offline`
- `cargo test --workspace --release --locked --offline -- --show-output`
- Debug and release profiles each passed 135 active Rust checks; five existing
  hosted-only checks remain ignored in the local workspace run.
- Objective A release integration and bitwise checkpoint-resume tests passed
  in isolation with `--test-threads=1`.
- `python3 tools/adamw_reference.py --check` — 33 frozen records.
- `python3 tools/task006_objective_a.py --check` — frozen fixture digest.
- `python3 tools/check_repository.py` — 150 tracked files checked.
- `git diff --check`.

The resume integration compared parameter bits, both moment arrays, bias
powers, optimizer step, schedule and next rate, next batch identity, cursor,
scientific token counters, and deterministic metrics. It also confirmed that
an interrupted partial window is discarded and recomputed without double
counting committed scientific tokens. Checkpoint tests cover checksum damage,
unknown section, duplicate field, truncation, oversize file, symlink artifact,
failed pre-rename write, and post-rename directory-sync uncertainty.

## Scope and limitations

No corpus was acquired, no sustained pretraining was run, and no Colab, paid
compute, or GPU resources were provisioned. The tiny Objective A fixture is
only a deterministic optimizer and integration check. CPU measurements do not
establish production inference or sustained-training performance. The
checkpoint codec documents filesystem assumptions and remaining crash/path
limitations. The future TASK 006-C/D experiment budget and any larger learning
objective require separate evidence and authorization.
