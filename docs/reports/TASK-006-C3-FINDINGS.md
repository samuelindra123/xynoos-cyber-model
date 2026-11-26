# TASK 006-C3 — Findings and Severity Register

| ID | Severity | Status | Finding | Required action |
|---|---|---|---|---|
| C3-F01 | Medium | Closed in `e6336e3` | Public callers could self-assert `source_clean=true` through `RunIdentity` constructors. | The training crate now embeds build provenance, public attestation uses it internally, `SourceIdentity` fields are private, dataset-backed identity construction requires the opaque attestation, and unchecked construction is crate-private. Debug/release package regressions and a compile-fail doctest pass. |
| C3-P01 | Acceptance blocker | Open | Implementation and C3 audit were performed by the same autonomous agent. The audit has a separate branch, code-free verification pass, and auditor-authored tests, but no independent human or agent reviewed the frozen R2 code. | Obtain review from a party separate from the implementation author before treating C3 as independent acceptance. |
| C3-I01 | Informational | Implemented; hosted verification pending | The R2 hosted workflow's fixed test total was stale; the latest R2 run expected 164 and observed 165 direct checks. | Child test output is suppressed. C3 expects 170 total direct checks (165 R2 plus five C3 checks); exact C3 hosted verification is pending. |

## Severity meaning

- **Medium:** a supported public API can emit materially false experiment provenance, although the official CLI rejects the same unverified checkout.
- **Acceptance blocker:** no implementation defect is implied; the independent-verification governance requirement is not met by a same-agent audit.
- **Informational:** a verification workflow defect or evidence boundary that does not change runtime results.

No stale-lock double-admission, partial-sidecar, or self-asserted public-source-identity defect remains in the audited R2 source. No training dispatch or compute provisioning is authorized by this report.
