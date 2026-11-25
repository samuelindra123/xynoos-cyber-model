# TASK 002-D — Remediation Handoff from Independent Audit

**Source audit:** TASK 002-C, audit commit `51c4d1cf2faa3b3b639c3e18d6fa79fc8a5515b6`  
**Implementation reviewed:** `cd5089992a706b0ba6a5f09c42571e8f3c3dc7ce`

## Required owner/Sessions decision: C-001

`Tensor::from_vec` accepts a Vec whose logical length satisfies `TensorLimits` and retains the Vec's entire existing capacity. For example, a vector with one initialized value and spare capacity can be accepted under a 4-byte logical limit while retaining the larger allocation. This is memory retention, not unsafe behavior. Existing docs distinguish logical payload from allocator overhead but do not clearly state that caller-provided Vec spare capacity survives adoption.

Session B should choose one explicit contract and report the tradeoff:

1. **Document logical-only policy:** retain zero-copy ownership transfer; state in `from_vec` docs that caller capacity is retained, the cap checks logical length only, and callers must not infer a retained-memory bound. Add a bounded test with a small excess capacity proving the adopted contract.
2. **Normalize retained capacity:** copy values into a fresh vector reserved for logical length, using fallible allocation and accounting for temporary source-plus-copy overlap; update ownership/cost docs and test result correctness and rejection behavior.
3. **Reject excess capacity:** define an acceptable capacity threshold and typed error, with rationale for allocator rounding and portability; test the bounded policy.

Do not claim any choice caps process RSS, aggregate live tensors, or allocator overhead. The owner should decide whether zero-copy ownership is more important than tighter retained-capacity semantics. Re-run debug/release tests, Clippy, docs, and CI after remediation. Audit evidence and exact repro are in `docs/reports/TASK-002-C-FINDINGS.md` (C-001).

No other confirmed implementation defect was found in bounded review. Revisit compute budgeting before exposing matmul to untrusted workloads; this is currently an explicit caller responsibility and not an acceptance blocker by itself.
