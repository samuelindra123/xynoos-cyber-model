# TASK 005-C requirement traceability

| Audit area | Evidence inspected | Result / limitation |
|---|---|---|
| Pinned repository and CI | Remote refs, PR #12/#13 metadata, run 45580159102/job 139087889051 | Exact heads match; both PRs open; implementation CI succeeded. |
| Architecture and inventory | `xynoos-transformer` config/model and reference kernels | Decoder-only pre-RMSNorm architecture; tied E; formula yields 7,536 / 62,880 / 914,400. No tensor crate change. |
| Autograd regression | `budget.rs`, `parameters.rs`, `tape.rs`, legacy test suites and CI | New reference path is separate; 105 debug/release checks reported. Legacy semantics retained per source/CI. |
| PAD boundary and CE | Reference batch validation, loss kernel, TASK 004 stream spec | Interpretation A accepted; B accepted for stable algebra and extreme equal logits. |
| Failure lifecycle | Model evaluator and tape reverse path | Interpretation C accepted; evaluation owns tape, reverse failures seal. |
| Initialization and rounding | `init.rs`, reference kernels, frozen Python fixtures | D and E require documented amendments because bit/narrowing boundaries exceed generic spec wording. |
| Attention / RoPE / VJPs | `reference/attention.rs`, `reference/kernels.rs`, operations fixture | Source equations and contiguous/interleaved layouts align; hosted controls cover major listed mutations. No new independent mutation execution in this audit. |
| Oracle independence | `tools/transformer_oracle.py`, fixture digests, CI regeneration | Python is separate from Rust and digests match report; shared choices mean fixtures are not universal proof. |
| Finite differences | Frozen operation/gradient fixtures and CI records | Hosted coordinate counts/errors match report; candidate-wide coordinate coverage remains absent. |
| Resource ledger and work | Budget, tape, report formulas and exact-head resource job | Charges/capacity ownership documented; process RSS properly distinguished. Units are policy counters, not FLOPs. |
| Gate A/B | Exact-head hosted Foundation job | Deterministic reported metrics match required values. |
| Gate C | Exact-head bounded candidate job | Full forward succeeds; reverse preflight rejects at 152,581,015 > 100,000,000 before adjoint allocation; cleanup is reported. |
| Dense reverse floor | Formula independently substituted from candidate dimensions | 116,834,304 units; exceeds existing cap independently of attention memory scheduling. |
| TASK 004 preservation | Workflow and legacy regression report | 90 accepted checks, tokenizer/data fixtures and digests preserved per hosted CI. |
| CI/dependencies | Workflow diff and hosted job steps | Locked dependencies, pinned toolchain, offline validations, read-only permissions, bounded checks, docs and privacy steps succeeded. |

## Interpretation dispositions

* A — **ACCEPT**.
* B — **ACCEPT**.
* C — **ACCEPT**.
* D — **REQUIRES DOCUMENTED AMENDMENT**.
* E — **REQUIRES DOCUMENTED AMENDMENT**.

No confirmed production defect was found in the reviewed paths. Full candidate backward behavior, gradients and reverse peak memory are unverified; this is an acceptance gap, not evidence that the pass succeeds.
