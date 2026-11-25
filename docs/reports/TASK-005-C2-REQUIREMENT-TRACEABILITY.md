# TASK 005-C2 requirement traceability

| Acceptance requirement | Evidence | Status |
|---|---|---|
| Pinned refs and existing remediation CI | Branch begins at remediation baseline `d6f70752f212b53b0a8a02b6ef50978b5ddc8487`; PR #16 head and prior CI run verified before edits. | Verified |
| Candidate opt-in and exact configuration | `Transformer::evaluate_candidate` checks the exact candidate config and `[1,32]`; wrong V/D/N/H/F/context and B/T controls preserve parameter state. | Verified |
| Inventory identity, not shapes alone | `c2_matching_shapes_without_candidate_inventory_identity_are_rejected` uses an ordinary unmarked inventory with exact ordered candidate shapes and expects `InvalidCandidateInventory`, with state and ledger checks. | Verified |
| Ordinary versus scoped backward ceilings | Existing ordinary candidate experiment rejects 152,581,015 against 100,000,000; scoped reverse succeeds at 152,581,015 against 160,000,000. The stricter 99,999,999 caller limit rejects. | Verified |
| Closed graph, loss-only reverse root | Reviewed `ReferenceTransformerEvaluation::record_candidate`, private tape ownership, `record_graph`, and `backward`; fault control inspects the fixed graph and shared parameter snapshot. | Verified |
| Rejection cleanup | Configuration, batch, inventory and caller-cap negative controls preserve revision/gradients and release evaluation resources. | Verified |
| Additional independent Transformer gradients | Twelve preselected C2 coordinates are frozen by the standard-library Python oracle and compared with the actual 914,400-parameter candidate. Includes E, attention projections, both RMSNorm families, SwiGLU projections, multiple blocks and final gain. | Verified |
| C2 central differences | Twelve additional coordinates at each `h=0.001` and `h=0.003`; actual representable FP32 distances; all pass `2e-3 + 2e-2*abs(fd)` against native and independent analytic gradients. | Verified |
| Transformer negative control and alternate derivation | Tied-E lookup omission on native Transformer differs by `0.063402506057`; small one-block independent operation fixture compares 6,281 forward values and 6,281 adjoints. | Verified |
| Initialization amendment | Normative v1, `init.rs`, and Python oracle reviewed; first SplitMix seed vector and both endpoint cases asserted; all 74 seeded candidate tensor hashes match. | ACCEPT |
| Rounding amendment | Normative operation table, Rust reference kernels and independent Python oracle agree; operation fixture, candidate forward/gradient comparisons, aggregates, finite differences and numerical negative controls pass. | ACCEPT |
| Complete candidate evidence | Candidate remains P=914,400, 74 tensors, eight blocks and 914,400 finite gradient cells; resource and work values remain exact. | Verified |
| Gradient transfer and identity | Complete batch at revision 74 resolves against source parameter IDs; foreign set returns `WrongParameterSet` and preserves receiver state. | Verified |
| Transactional accumulation | Successful first accumulation preserves parameter bits/revision and gradient buffers. Stale and injected allocation failures preserve existing gradient/parameter bits and revision; temporary allocations are released. | Verified |
| Resource cleanup | Isolated transaction set ledger returns to zero after destruction; rejected auxiliary batch cleanup returns its source set to baseline. | Verified |
| Preserved regression evidence | Baseline 109 checks/profile, scalar C2 control, four bounded candidate experiments, numerical/gradient/tokenizer/data/Transformer fixtures and accepted resource contracts remain present. | Verified |
| Exact final audit head and hosted logs | Final SHA, Foundation CI run/job, hosted outcome and log inspection are recorded in the independent re-verification report after dispatch. | Pending |
| Owner acceptance | Awaiting owner review after exact-head hosted CI; no merge or TASK 006 authorization is implied. | Pending owner |

The two pending rows are intentionally different: hosted CI is required to
complete this audit; owner acceptance is a separate decision after the audit
is complete.
