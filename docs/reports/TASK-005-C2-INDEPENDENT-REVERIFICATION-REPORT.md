# TASK 005-C2 independent candidate re-verification

## Scope and verdict

Audit branch starts at remediation head `d6f70752f212b53b0a8a02b6ef50978b5ddc8487`.
The audit did not modify production behavior. The remediation Foundation run
45585534830 completed successfully on that exact SHA (job 139106044725).

**Current verdict: BLOCKED — INSUFFICIENT VERIFICATION EVIDENCE.** The work
below adds an independently derived mathematical control, but does not replace
the full candidate run or complete the requested independent coverage. This
report intentionally does not claim TASK 005 owner acceptance.

## Reviewed sources and pinned state

The pinned planning, implementation, original-audit and remediation commits
resolve locally as `f56af8f`, `97179bc`, `7ada8b8` and `d6f7075`. Reviewed the
TASK 005-B implementation and resource reports, TASK 005-B traceability, TASK
005-R remediation and candidate reports, normative amendment, repository
instructions, compute and dataset policies, efficiency-first direction and
roadmap. PR #15 changes are limited to scoped autograd policy plumbing,
candidate verification, fixtures/oracle, CI integration, and related reports;
no tokenizer, dataset, tensor implementation or model architecture edits were
found in its changed-file inventory. No production file was edited by C2.

The amendment specifies initialization stream v1, FP32/FP64 operation
boundaries, candidate-only policy, and input/failure lifecycle. Its policy
requires the immutable designated inventory, exact shapes, fixed graph, sole
loss root, `[1,32]` batch and tighter caller limits. Source inspection found
these checks in `CandidateReferenceExperimentPolicy`,
`ReferenceTransformerEvaluation::record_candidate`, and
`Transformer::evaluate_candidate`. The policy remains explicitly supplied;
the private tape owns the fixed graph. The acceptance run must still exercise
attempted configuration substitution, lower-level entry points, exact-limit
semantics and all negative policy cases independently.

## Auditor-authored check executed

Added `c2_independent_tied_edge_derivative_and_negative_control` to the
candidate integration test module. For an independently constructed scalar
fixture, `L(x)=0.5*(a*x+b*x)^2`, with `x=.37`, `a=-.8`, and `b=1.3`, the
analytic shared-parameter derivative is `(a+b)^2*x`. A central perturbation
with `h=1e-5` agreed within `1e-10`. The deliberate wrong expectation that
drops the projection path disagreed by **0.240500000000**; the test asserts
that this defect is detected. The test passed locally (1 passed, 0 failed).
It is an independent scalar identity check, not a full Transformer gradient
comparison and does not establish the required additional coordinate coverage
across all families/blocks.

## Evidence retained from exact remediation CI

The hosted run reports success and exact head `d6f7075`; it ran the repository
Foundation checks, candidate experiments, oracle fixture regeneration,
debug/release suites, and the injected transaction fault test. The candidate
report records the expected `[1,32]` to `[1,32,292]` evaluation, 914,400
parameters, 74 tensors, 9,344 logits, complete finite gradient inventory,
225 coordinate comparisons, 222 gradient aggregates, and the requested
finite-difference values. Candidate fixture SHA-256 values independently
recomputed in this checkout are:

| Fixture | SHA-256 |
|---|---|
| `candidate.json` | `46cdcf67b819fcc6658938331728d588c21dbcdf14f047b3938268ea2ee2df59` |
| `candidate-tied-paths.txt` | `1fdaa1553ae3415a88240c20a858d19a94fd8c1d8776cabda8e5e692c4af9f21` |

These are reported remediation evidence, not a substitute for this audit's
required exact-head CI run. No C2 full candidate or independent finite
difference coordinate run is claimed here.

## Outstanding evidence and limitations

Before acceptance, C2 still needs exact-head hosted execution and inspection
of its logs; independent complete candidate and resource-policy boundary
execution; auditor-selected Transformer coordinates and additional finite
differences; direct confirmation of initialization checksums and rounding
amendment behavior; transaction and injected-failure bit-preservation checks;
measured ledger/work/time values; and regression digest/check-total
confirmation. The 109 legacy checks and four candidate experiments must remain
unchanged. The documented original candidate accounting is application ledger
accounting, not process RSS. Timing samples are reference runs, not throughput
guarantees.

TASK 005 evidence cannot establish sustained training readiness or model
intelligence. At most, acceptance may establish the reference architecture,
the full bounded candidate evaluation, transactional gradient handling, and
readiness to begin TASK 006 architecture planning. Any future authorized
training must be detached from continuous agent supervision where supported,
with deterministic logging, checkpoints, failure detection, limits and
termination; verify Colab persistence and cleanup before relying on detached
execution. This audit did not provision Colab or start training.
