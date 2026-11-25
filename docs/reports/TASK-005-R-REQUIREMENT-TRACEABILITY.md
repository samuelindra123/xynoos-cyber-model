# TASK 005-R requirement traceability

| Requirements | Evidence |
|---|---|
| Baseline and canonical checkout | PR #12–14 heads, baseline Foundation jobs, and the unchanged implementation reproduction in the [remediation report](TASK-005-R-REMEDIATION-REPORT.md). |
| Candidate policy and legacy isolation | `Transformer::evaluate_candidate`, the fixed graph in `crates/xynoos-autograd/src/tape/transformer.rs`, and the exact default-policy and tighter-cap rejection tests. |
| Reverse accounting | The existing checked reachable planner in `crates/xynoos-autograd/src/tape.rs` uses the authorized cap only through the closed candidate graph. Exact boundary controls use the same comparison and a small actual reverse graph. |
| Numerical specification | [Version 1 amendment](../design/TASK-005-NORMATIVE-NUMERICAL-AND-RESOURCE-AMENDMENT.md) for seed, draws, endpoint, arithmetic, masks, loss and lifecycle. |
| Native forward, reverse and inventory | `candidate::full_candidate_b1_t32` in the Transformer reference integration suite. Complete inventory and phase measurements in the [candidate report](TASK-005-R-CANDIDATE-VERIFICATION-REPORT.md). |
| Independent mathematics | `tools/transformer_oracle.py --candidate --check`, frozen candidate fixtures, 225 coordinate checks, 222 aggregate comparisons, 24 finite differences and the independent E paths. |
| Transfer and transactional accumulation | The candidate integration test checks successful transfer, exact inventory and revision, first accumulation and wrong-set rejection. The candidate autograd control checks later staging allocation rollback, failed transfer cleanup, stale revision and destruction to zero ledger. |
| Regression and hosted constraints | The bounded Foundation workflow reruns accepted oracles, 105 baseline checks and four ordinary remediation controls per Rust profile, four separate release experiments, formatting, Clippy, docs and privacy. |
| Handoff and stop boundary | Draft PR #15 retains the implementation branch as base. Session A owns independent re-verification. The roadmap retains its existing acceptance state. |

## Executed negative controls

| Control | Result and state assertion |
|---|---|
| Default policy on Gate C | `WorkLimit` with 152,581,015 requested against 100,000,000. Forward peak is unchanged by rejection and dropping the evaluation restores parameter-only bytes. |
| Tight candidate work cap | `WorkLimit` with 152,581,015 requested against 99,999,999. Rejection preserves the ledger and releases the evaluation. |
| Wrong V, D, N, H, F or context cap | `InvalidCandidateConfiguration` at the model boundary and `InvalidCandidateInventory` at the lower-level boundary. The wrong-context and wrong-head sets have the same tensor shapes as the candidate. |
| Wrong B or T | `InvalidCandidateBatchShape`. The raw factory also rejects `[2,16]` and a correctly sized inventory with scalar entries. |
| Work equal to or one over the limit | The shared backward work-limit comparison accepts 100,000,000 and 160,000,000 at their respective boundaries and rejects one above each. A real small graph runs at demand equal to its limit, then rejects at demand one above its limit before a pending allocation fault could fire. |
| Tensor, aggregate and tape limits | Transformer controls detect the tensor payload limit, inventory and reverse aggregate limits, tape value limit and tape operation limit. The legacy suites also test retained capacity and reverse/transfer peak overlap. |
| Arithmetic overflow | Checked matrix-work estimates and model parameter arithmetic report overflow before payload generation. |
| Allocation and nonfinite inputs | Existing typed tensor/tape controls reject NaN and infinities, detect finite-input overflow and inject forward/reverse allocation errors. The candidate fails partway through reverse before native adjoints can publish. |
| Stale and foreign gradient batches | `StaleGradientBatch` after revision advances in an isolated candidate and `WrongParameterSet` on an auxiliary batch. The case retains prior persistent gradients and releases the rejected batch. |
| Failed transfer | An injected batch-table allocation fault during candidate transfer consumes the tape and restores parameter-only bytes. |
| Failed accumulation | An allocation fault after four staged replacements preserves all existing gradients, values, revision and ledger charges. The legacy nonfinite accumulation rollback also remains covered. |
| Malformed batch | Seven additional mask, PAD and shift combinations exercise typed failures, including interior PAD with loss zero and a wrong unsupervised target. Parameter state and ledger are checked after every rejection. |
| E path omission | The independent fixture detects omission of each E path on both seeded batches. The explicit-weight candidate control checks shared snapshot identity, both native edge contributions and their combined adjoint. |

## Independent review gate

Session A should reproduce default rejection and scoped completion on the
exact draft head, inspect the fixed-graph and inventory binding through
both entry points, re-derive the numeric boundaries and work charges, and
inspect the frozen-fixture provenance and transaction tests. Session B's
implementation and hosted checks supply evidence for that review.
