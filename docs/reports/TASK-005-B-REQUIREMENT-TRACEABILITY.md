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
