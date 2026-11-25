# XYNOOS reverse-mode differentiation

## Lifecycle and numerical contract

A parameter set fixes the resource policy for its entire lifetime. Every tape borrows the set and inherits its limits. A tape accepts forward records until its first backward attempt. Success makes adjoints readable and seals the tape. Any backward error discards its adjoints and permanently seals the tape. Forward failures publish no value. An owned gradient batch consumes a successful tape and ends the parameter borrow.

An input is a constant leaf. MSE accepts only a constant input leaf as its target. Its prediction and target have identical nonempty shapes. Differences, squares, sequential row-major accumulation, division, and the backward expression `((g * 2) * (p - target)) / n` use binary64. Each output or edge contribution converts once to binary32 and must be finite.

Disconnected values and constants have missing adjoints. A connected zero contribution produces an explicit gradient. Disconnected parameters retain their previous persistent gradient, if any. Call `zero_grad` to install explicit zeros across the inventory when an update should include disconnected parameters. SGD requires a gradient for each parameter and clears all gradients after commit.

Matmul work includes output initialization and a finite-value scan even for zero inner dimension. An empty output skips the cell kernels. Each derivative product has its own output cell count and inner dimension, so empty forward outputs can still yield nonempty zero derivatives.

## Resource contract

Charge simultaneously retained tensor data and shape capacities, metadata vector capacities, and staging buffers across all tapes and owned batches associated with the same set. External inputs, allocator bookkeeping, fixed-size handles, stack, and process RSS are outside the accounting boundary. Vector growth stages a replacement before moving entries and charges both old and new capacity. A detached allocation with excess reported capacity is released on reconciliation failure. Reservations preflight the requested capacity, and any capacity reported above the request can exist transiently during reconciliation. Fallible vector reservations handle expected allocation errors. The fixed shared accounting handle uses the standard allocator.
