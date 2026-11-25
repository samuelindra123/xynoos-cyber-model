# TASK 005-C2 requirement traceability

| Requirement group | Evidence in this audit branch | Status |
|---|---|---|
| Pinned refs and remediation CI | Git objects resolve; run 45585534830 succeeded on `d6f7075` | Verified baseline |
| Required source review | TASK 005-B/R reports, normative amendment, AGENTS, compute/data policies, engineering direction and roadmap reviewed | Reviewed |
| PR #15 scope | Changed-file inventory inspected; no architecture/tokenizer/dataset/tensor changes identified | Reviewed |
| Candidate policy confinement | Policy and binding source inspected; detailed independent boundary attempts not executed | Partial |
| Auditor-authored math and negative control | Scalar tied-path derivative plus omitted-path negative control; disagreement 0.2405 | Executed locally |
| Full candidate forward/backward | Exact remediation CI report contains run; not independently rerun on C2 | Inherited evidence only |
| Additional candidate gradient coordinates | Not implemented | Missing |
| Candidate finite differences | Remediation evidence reviewed; no additional C2-selected samples | Missing |
| Initialization and rounding | Normative contract and implementation report reviewed; exhaustive independent checks not executed | Partial |
| Transfer/accumulation/failure rollback | Remediation source/report and CI result reviewed; no independent C2 fault test | Partial |
| Resource, work and timing measurements | Existing report values reviewed; no C2 measurement | Inherited evidence only |
| Regressions and exact-head CI | Existing exact remediation CI verified; audit exact-head CI not yet obtained | Missing |
| Draft PR and handoff | Not yet created | Missing |

Do not interpret “inherited evidence” as independent re-verification. The
final audit verdict must remain blocked until all required rows are completed
and the exact audit head has successful hosted CI with inspected logs.
