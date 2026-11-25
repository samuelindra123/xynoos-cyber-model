# TASK 005-C candidate resource policy proposal

## Decision

**Recommend a model-scoped reference experiment policy with a 160,000,000-unit backward ceiling, subject to separate owner authorization and the remediation evidence below.** Do not change `ResourceLimits` defaults or the legacy 100,000,000-unit ceiling. The exact candidate demand is 152,581,015, leaving 7,418,985 units (4.86%) of planner headroom. This recommendation authorizes no code or compute in this audit and does not establish candidate training readiness.

## Alternatives

| Option | Assessment |
|---|---|
| A. Keep current cap and defer | Safest and preserves all policy, but blocks full candidate backward and TASK 006 readiness. Appropriate until explicit authorization. |
| B. Scoped reference experiment policy | Recommended after authorization. Small, explicit scope; keeps tensor, aggregate, node and operation ceilings. Requires robust constructor binding and full candidate validation. Limited generalization. |
| C. Equivalent optimized dense scheduling | Worth later study, but cannot evade the 116,834,304 dense contribution floor under current accounting. New schedule/accounting and parity review add complexity; does not currently establish adequate headroom. |
| D. Backend-specific accounting | Future work only. CPU/GPU units and kernels need separate measured definitions and parity; no physical-resource failure is demonstrated here. |

## Session B remediation contract

Implement only after separately authorized. Name the explicit API/policy `CandidateReferenceExperimentPolicy` (or an equally explicit model-scoped name; never a global default). It must bind exactly `V=292,D=96,N=8,H=4,F=256,context_cap=128`, batch `B=1`, sequence `T=32`, and backward ceiling `160_000_000`. Reject any other model or batch/sequence at construction/evaluation with typed errors. Default constructors and all legacy `ResourceLimits` remain unchanged.

Use checked arithmetic to preflight the complete reachable backward graph and all work charges before allocating reverse adjoints. Keep 64 MiB per tensor, 128 MiB aggregate ledger, current node and operation ceilings, capacity accounting, transactional gradient transfer/accumulation, and allocation-failure cleanup. No unbounded fallback, bypass, disabled checks, or silent global work increase. Rejection must leave no gradient publication, parameter mutation, or leaked ledger charge; reverse failure seals the tape consistently.

Before claiming support, bounded hosted CI must execute the exact full candidate backward and report deterministic work and peak ledger bytes. Compare analytic gradients independently across every parameter family and all eight blocks; include tied embedding lookup plus output fan-in. Add justified finite-difference samples spanning embedding, attention gains/Q/K/V/O, FFN gains/gate/value/down, final gain, and multiple blocks; report absolute and relative error separately. Include gradient identity/sum checks and preserve existing tolerances unless a separately justified proposal is reviewed.

Required controls: scoped-policy wrong-config rejection; exact work ceiling and one-unit-over rejection before allocation; tensor/aggregate/node/operation limits; injected allocation failure; nonfinite rejection; no partial gradient or parameter publication; stale gradient-batch rejection; transfer and accumulation transactionality; and default legacy cap regression. Document initialization/rounding amendments and exact resource semantics. Keep execution bounded in hosted CI; no local limit increase, Colab provisioning, or TASK 006 training is part of implementation.

Candidate readiness requires successful full backward, independent gradient evidence, transfer/accumulation checks, measured reverse peak within ledger limits, and exact-head CI. Gate A/B and candidate forward success alone do not qualify.
