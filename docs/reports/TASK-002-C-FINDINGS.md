# TASK 002-C Findings

## C-001 — `from_vec` can retain capacity beyond logical payload limit

- **Severity:** MEDIUM
- **Affected location:** `crates/xynoos-tensor/src/tensor.rs`, `Tensor::from_vec`
- **Evidence:** `validate_data(shape, data.len(), limits)?` constrains logical length; the function then stores the original `Vec<f32>` directly. No check or shrink is applied to `data.capacity()`. The API documentation says the caller may have allocated the buffer before checks, but does not state that arbitrary spare capacity remains retained or excluded from the policy limit.
- **Reproduction:** Construct a Vec with a large capacity and one initialized element, then call `Tensor::from_vec(&[1], vec, TensorLimits::new(4))`. It succeeds and `Tensor` retains that Vec's capacity. Inspecting capacity requires a small audit-only unit harness inside the module or an internal debug view; logically the adopted Vec is unchanged.
- **Expected:** Either the limit contract clearly applies only to logical values and callers knowingly accept retained capacity, or construction enforces an explicit retained-capacity policy.
- **Actual:** A tensor with one logical f32 can retain caller-supplied spare capacity far exceeding the requested 4-byte limit (subject to what the caller could allocate).
- **Impact:** A per-tensor logical payload policy may be mistaken for a bound on memory retained by the Tensor. This is particularly relevant to resource-constrained callers and untrusted integration boundaries. It does not permit safe-code memory unsafety and cannot exceed memory the caller already obtained, but extends its lifetime.
- **Recommendation:** Make the contract explicit and owner-approved, or copy into a fresh fallibly reserved vector sized to `len` (which introduces a copy and temporary overlap), or reject/normalize excessive capacity with a carefully specified rule. Avoid claiming RSS guarantees. Add a focused test that verifies the chosen retained-capacity behavior without large allocations (small Vec with deliberate modest excess capacity).
- **Acceptance:** Conditional; resolve or explicitly accept and document before owner acceptance.
