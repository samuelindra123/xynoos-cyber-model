# TASK 006-C3 — Findings and Severity Register

| ID | Severity | Status | Finding | Required action |
|---|---|---|---|---|
| C3-F01 | Medium | Open | Public `RunIdentity::new` and `RunIdentity::from_standard_dataset` accept caller-supplied `source_clean=true`. A Rust library caller can bypass the CLI's verified repository/revision/working-tree attestation and create a run identity that claims clean provenance. | Make source identity an opaque attestation value produced only by `attest_source_checkout`; require it when constructing a dataset-backed run identity. Keep any unchecked test constructor crate-private. Add an API-level negative/positive regression and repeat exact-head checks. |
| C3-P01 | Acceptance blocker | Open | Implementation and C3 audit were performed by the same autonomous agent. The audit has a separate branch, code-free verification pass, and auditor-authored tests, but no independent human or agent reviewed the frozen R2 code. | Obtain review from a party separate from the implementation author before treating C3 as independent acceptance. |
| C3-I01 | Informational | Closed | The R2 hosted workflow's fixed 150-check total was stale; the branch had 163 direct checks and child worker output inflated the raw log total. | C3 branch suppresses child test output and updates the expected suite total to 168 after adding five C3 tests. Hosted verification is pending. |

## Severity meaning

- **Medium:** a supported public API can emit materially false experiment provenance, although the official CLI rejects the same unverified checkout.
- **Acceptance blocker:** no implementation defect is implied; the independent-verification governance requirement is not met by a same-agent audit.
- **Informational:** a verification workflow defect or evidence boundary that does not change runtime results.

No stale-lock double-admission or partial-sidecar defect was reproduced after R2. C3-F01 keeps R2 from passing audit until corrected. No training dispatch or compute provisioning is authorized by this report.
