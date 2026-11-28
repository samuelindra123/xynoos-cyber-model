# TASK 006-F — Compute Feasibility

**Outcome: proposed v3 run is not admitted; complete workload fit is not demonstrated.** No TASK 006-D training workflow was dispatched and no compute provider was provisioned.

## Unchanged hard limits

| Limit | Existing accepted maximum | Pilot workflow maximum |
|---|---:|---:|
| Ordinary work | 100,000,000 | — |
| Scoped candidate backward | 160,000,000 | — |
| Tensor | 64 MiB | — |
| Autograd ledger | 128 MiB | 64 MiB |
| Training-owned memory | 128 MiB | 64 MiB |
| Checkpoint | 16 MiB | 16 MiB |
| Updates | 10,000 | 24 |
| Consumed/supervised tokens | — | 768 |
| Cumulative work/run | — | 8,000,000,000 |
| Active runtime | — | 120 s |
| Single attempt runtime | — | 90 s |
| Data allowance | — | 4 MiB |

No limit was raised or changed by F. The workflow is manual and requires `authorize_training_run=true`; it was not dispatched.

## Measured work and limitations

The integrated v3 release smoke is a two-update software control with one partial validation batch per update and checkpointing every update. It reported `825,200,872` cumulative work, 19 consumed/supervised training tokens, 18 validation tokens, owned peak `42,439,416` bytes, autograd peak `30,682,296` bytes, and process VM high-water `35,950,592` bytes in its recorded hosted/software-control run. A 12-fold cadence projection is `9,902,410,464` work units, above the 8B cap. This is a projection, not a complete 24-update measurement; it omits full validation/baseline and cannot prove the exact proposed cadence is impossible. It does show that the current control does not establish fit.

The five unchanged-weight full-split baseline evaluators separately consume `3,124,414,056` forward-work units. The prior v1 calculation (`7,782,315,616` for one specific 24-update schedule, only 2.72% headroom) is retired and does not transfer to v3. The proposed 3-LR sweep is also retired in TASK 006-E; if treated as 3 independent runs under the 8B per-run ceiling, its worst-case aggregate ceiling would be 24B, and no aggregate compute budget or scientific justification exists. Do not infer an approved 3-run experiment.

No measurement covers initialization, full training updates, full validation, initial baseline, every checkpoint, resume, logging, shutdown, cleanup, and final report as one complete v3 workload. No three repeated provider measurements exist. One short software smoke cannot establish runtime fit. Process VM high-water is distinct from the owned-memory and autograd ledgers.

## Finding F-COMP-01 — full-run resource fit unproven

- **Severity:** Critical for dispatch.
- **Component:** Objective A v3 full training/validation/recovery workload and Actions CPU budget.
- **Reproduction:** Run the existing bounded v3 software control and compare `825200872 × 12` to `8,000,000,000`; inspect the five-evaluator baseline total and workflow caps.
- **Evidence:** The same-cadence projection exceeds the cap; complete schedule/cadence workload is unmeasured. Validation every six updates and checkpoint every six updates have not been measured over the full workload including startup, finalization, and recovery.
- **Root cause:** V3 integration and baseline were added after earlier compute estimates; previous workload was a different v1/v2 task/cadence.
- **Remediation:** After scientific protocol is fixed, instrument the exact immutable full workload and count initialization, train, validation, checkpoint, resume, logs, shutdown, cleanup, and recovery. Run at least three bounded repetitions in the intended provider; admit only if the slowest completes under all work/memory/runtime limits. Do not raise limits or remove required evaluation to fit.
- **Verification:** Not passed. Current evidence supports only a software control, not workload admission.

## Required complete measurement record

For each repeated run, record source/config/data/tokenizer hashes, seed, exact operation counts, consumed tokens, work, owned and autograd peaks, process RSS/high-water, data and checkpoint bytes, initialization time, training/validation/checkpoint/resume/finalization durations, total wall time, process cleanup, and recovered artifact digest. The slowest valid repetition must remain within 8B work, 64 MiB owned, 64 MiB autograd, 4 MiB data, 16 MiB checkpoint, 120 seconds active time, and 90 seconds per attempt. The separate unchanged accepted hard limits remain unchanged as tabled above.

## Provider disposition

GitHub Actions remains the default bounded CPU validation environment, but a training run is not approved. Private remote persistence and detached-provider recovery are unverified. No GPU backend exists; no accelerator, Colab runtime, paid unit, external dataset, or model-weight upload was used. Only after scientific and compute gates pass may the owner choose a provider and authorize the exact run bundle.
