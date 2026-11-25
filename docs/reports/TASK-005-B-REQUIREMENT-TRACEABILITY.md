# TASK 005-B requirement traceability

## Scoped interpretations for Session A and owner review

The implementation contract is the six TASK 005-A files at
`f56af8fccbd4d1248ce3919504211c05ae9a3c42`. Production starts from
`334beb088f116240cdd0346a70d9527c26f08e51`.

- PAD in targets has its own boundary. For a valid final input, a PAD target
  is allowed with loss mask 0. An interior PAD target violates the
  next-token shift. An invalid input row entry requires PAD input, PAD target,
  and loss mask 0. TASK 004 emits paired windows and pads both sides together.
  The valid-input/PAD-target case is a consumer edge requested for TASK 005-B.
- Calculate row cross-entropy as `log(sum(exp(z-max))) + (max-z[target])`
  in binary64. Directly calculating `max + log(sum) - z[target]` loses the
  log term for equal logits near the binary32 maximum.
- The reference entry point owns its tape during construction. On a forward
  error it drops that tape and publishes no result. Legacy callers of
  individual tape operations retain their existing forward retry semantics.
  A reverse-pass error still seals the tape.
- Initialization order is E, then each block in execution order with
  attention gain, Q/K/V/O, FFN gain, gate/value/down, then final gain. Gains
  consume no random draws. Box–Muller standard normal outputs are rounded to
  binary32 before multiplication by 0.02 in binary64 and final narrowing.
  The top-53 midpoint formula can round the final midpoint to 1 in binary64.
  Use the adjacent representable value below 1 at that endpoint to maintain
  the specified open interval.
- RoPE trigonometric coefficients round to binary32. Pairwise transforms
  use binary64 products and signed sums before narrowing each output.
  Attention scores and probabilities use binary64 and the value-weighted
  sum narrows at the attention output boundary. SiLU narrows before the
  pointwise multiplication. Local VJPs narrow once per edge element.
  Fan-in is summed in binary64 on the reference path before each node's
  upstream adjoint narrows. Legacy tape arithmetic stays in place.

These are explicit interpretations of under-specified evaluation boundaries
or conflicting wording. Their acceptance is part of the independent review.

## Evidence map

| Requirements | Implementation and evidence |
|---|---|
| 1–3, 27, 29, 31 | Pinned planning basis and accepted main in the implementation report, draft PR #13 and exact-head Foundation CI. |
| 4–7, 9–11 | Native model in `crates/xynoos-transformer`. Fixed-rank operations in `crates/xynoos-autograd/src/reference`. Inventory formula, three exact counts, shape rejections and 7,536 independent initialization values. |
| 8, 21 | `tools/transformer_oracle.py`, three frozen JSON files, operation outputs and VJPs, all mathematical parameter gradients and 76 four-block coordinates, both finite-difference steps. |
| 12 | `Transformer::evaluate_batch` and `ReferenceBatch`. TASK 004 integration test fits BPE, streams accepted validation batches and compares partial versus fixed-size output. |
| 13–15 | Contiguous interleaved head layout, finite-only mask-aware attention, E identity and independent per-path sum, extreme and zero-supervision CE controls. |
| 16–18 | Native gradient batch and legacy SGD retained. FP64 pending adjoints and all new buffers charged through the native ledger. Initialization, forward, backward, transfer and optimizer work are documented by phase in the resource report. |
| 19–20 | Both reference gates, measured ledger peaks, and the unchanged B=1,T=32 research configuration with a typed reverse-work blocker. |
| 22–23 | Executed Python equation mutations, typed Rust failure tests, allocator injection, shape/mask/causality and padding/batch invariance tests. |
| 24 | Existing Foundation checks and locked fixture regeneration, accepted trajectory and tokenizer digests, new checks reported separately. |
| 25–26 | Scalar CPU reference, bounded hosted candidate experiment, no optimizer loop or Colab runtime. |
| 28, 30 | Implementation, resource and traceability reports and separate session handoff. |

## Session A audit

Audit the five interpretation bullets against both pinned mathematical
documents and the TASK 004 stream. Re-derive the local VJPs and confirm
the FP32 boundaries, FP64 fan-in order, and exact E identity. Review
adversarial batch validation, initialized bit patterns, and the disjoint
legacy/reference tape paths. Inspect all ledger ownership and replacement
overlap, reproduce the reverse planner's candidate demand, and check the
exact-head logs and frozen oracle checksums. The candidate requires a
separate decision before any redesign of the effective reverse phase budget.
