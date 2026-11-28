# TASK 006-F — Numerical and Optimizer Verification

**Outcome: frozen software contracts reproduced; no numerical limit or fixture was changed.** These results verify bounded fixtures and reference calculations. They do not validate learning quality or authorize a training run.

## Findings

### F-NUM-01 — tensor/autograd/optimizer fixture parity

- **Severity:** High; tested software gate passed.
- **Component:** Native tensor, reverse-mode autograd, gradient transfer/accumulation, clipping, AdamW, scheduling, and update transaction.
- **Reproduction:** `python3 tools/numerical_oracle.py --check`; `python3 tools/gradient_oracle.py --check`; `python3 tools/adamw_reference.py --check`; `cargo test -p xynoos-autograd --test task006c_independent --locked --offline -- --show-output`.
- **Evidence:** Numerical oracle: 295 cases / 2,670 outputs. Gradient fixture: 25 gradient rows and 101 learning rows with frozen digests. AdamW oracle: 33 records with frozen digest. Native independent AdamW and weighted-gradient controls passed.
- **Root cause:** No unresolved arithmetic defect observed in the tested supported reference path.
- **Remediation:** None indicated by the available evidence. Keep contracts and fixture hashes unchanged.
- **Verification:** Commands passed on the integrated source in local verification. Historical exact-head CI for #23 and #28 passed their respective source revisions; F exact-head CI must pass before source integration review.

### F-NUM-02 — optimizer state and transaction behavior

- **Severity:** High; tested software gate passed.
- **Component:** Weighted accumulation, global clipping, AdamW bias correction/weight decay, learning-rate schedule, and parameter publication.
- **Reproduction:** `python3 tools/adamw_reference.py --check` and the native independent autograd test above; inspect `crates/xynoos-training/src` optimizer and trainer tests.
- **Evidence:** FP64 accumulation and schedule/bias-power calculations are exercised against frozen references; native update stages candidate parameters and rejects nonfinite/overflow before publishing. Independent controls cover multiple updates, zero gradients, extreme representable cases, weighted accumulation, and a negative control.
- **Root cause:** No discrepancy found within frozen cases/tolerances; untested behavior outside fixture domains remains possible.
- **Remediation:** None to the equations. Preserve FP32 stored parameters/moments and existing FP64 intermediate behavior; any change requires new oracle evidence and owner-reviewed spec revision.
- **Verification:** Passed. No tolerance, optimizer equation, fixture hash, or resource cap was altered during F.

## Preserved numerical and resource contracts

| Contract | Preserved value |
|---|---:|
| Ordinary work | 100,000,000 |
| Scoped candidate backward work | 160,000,000 |
| Maximum logical or retained tensor payload | 64 MiB |
| Autograd ledger | 128 MiB |
| Training owned memory | 128 MiB |
| Checkpoint | 16 MiB |
| Maximum experiment updates | 10,000 |

The 160M scoped candidate-backward allowance is distinct from the ordinary 100M budget. Workflow-specific pilot ceilings are stricter and are summarized in the compute report. `VmHWM` is a process high-water observation, not the application-owned or autograd ledger and not a guaranteed hard RSS limit.

### F-NUM-03 — independent review and full experiment evidence

- **Severity:** Critical for first-training acceptance.
- **Component:** Release assurance and learning evidence.
- **Reproduction:** Review exact source identity and run the final candidate on hosted CI; obtain a reviewer separate from the R2/C3 implementation/audit author.
- **Evidence:** C3 explicitly records same-author implementation and verification; no human/reviewer approval is present in the open PRs. No accepted post-update v3 result exists.
- **Root cause:** The present evidence is a software audit and baseline package, not external sign-off or a scientific training run.
- **Remediation:** Independent review of the frozen diff and evidence; after scientific/workload gates pass, an owner-authorized bounded run with predeclared trajectory criteria.
- **Verification:** Open. No TASK 006-D training was dispatched.
