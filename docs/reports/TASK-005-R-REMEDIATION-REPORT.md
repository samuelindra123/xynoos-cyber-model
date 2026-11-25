# TASK 005-R scoped remediation report

## Baseline

| Reference | Verified SHA |
|---|---|
| Accepted main | `334beb088f116240cdd0346a70d9527c26f08e51` |
| Planning PR #12 | `f56af8fccbd4d1248ce3919504211c05ae9a3c42` |
| Implementation PR #13 | `97179bcb1af400e56c8b1983590dcad0d4dba946` |
| Independent review PR #14 | `7ada8b8e0f1e8db44521c2250c4866aa0b3afa20` |

Implementation Foundation run 45580159102, job 139087889051, and audit
Foundation run 45581798164, job 139093407976, both succeeded at the listed
heads. The audit adds only its three documents to the implementation head.
Session B reproduced the default-policy rejection on an unchanged checkout
of the implementation revision before changing the code. The fixture's
3,669,304 parameter-only bytes, 20,118,280 initialization units, 66,894,756
forward units, 9,658,268 forward peak bytes and loss 5.736882210 matched
the prior logs. The exact error was
`WorkLimit { pass: Backward, requested: 152581015, limit: 100000000 }`.
The ledger never rose above its forward peak and returned to parameter-only
bytes on disposal. The default test remains in Foundation CI.

## Implementation

`Transformer::evaluate_candidate` selects the authorized
`CandidateReferenceExperimentPolicy` for the model's validated
`Config::RESEARCH_CANDIDATE` and a batch of shape `[1,32]`. Config and
batch mismatches have distinct typed errors. Parameter-set creation binds
the exact config to a private inventory designation. The lower-level
candidate entry point also checks the designation and all registered
shapes, including the transposed orientation of the down matrix relative to
gate and value. A parameter set from a different context cap or head
count fails even when its tensor shapes match the candidate. The scoped
entry point records the fixed model graph and returns an opaque
evaluation with only its scalar loss as reverse root.

The existing graph composition is shared by the two model entry points.
The backward planner still walks the complete reachable graph with
checked arithmetic before reverse-adjoint allocation. Graph traversal,
pending arrays, binary64 fan-in, binary32 publication, all local edge
kernels and finite scans use the TASK 005-B work definitions. The change
applies 160,000,000 only to this evaluation's loss backward at the default
work cap. Caller-supplied lower ceilings retain precedence. Regular
evaluations and raw `Tape` calls remain subject to the 100,000,000 ceiling.
Tensor, aggregate ledger, value, operation, transfer, accumulation,
optimizer and experiment-step limits retain their previous semantics.

The model's decoder architecture and numerical code are unchanged.
[The versioned amendment](../design/TASK-005-NORMATIVE-NUMERICAL-AND-RESOURCE-AMENDMENT.md)
records the SplitMix64/Box–Muller stream, precision boundaries, PAD-target
and stable loss semantics, and transactional failure lifecycle. The pinned
TASK 005-A history and accepted fixture bytes are preserved.

## Verification and disposition

[Candidate verification](TASK-005-R-CANDIDATE-VERIFICATION-REPORT.md) reports
full native backward, independent coordinate and aggregate comparisons,
finite differences, the tied E check, transfer and accumulation, measured
resources, and regression outcomes. [Traceability](TASK-005-R-REQUIREMENT-TRACEABILITY.md)
lists the executed checks and preserved contracts.

The remediation is in draft PR #15 targeting the implementation branch.
Session A must independently review the scope binding, the rounding
amendment, the accounting and transactions, and the mathematical and
exact-head hosted evidence. This experiment measures a fixed CPU reference
graph. It establishes neither model quality nor production training capacity.
TASK 006 remains outside the approved remediation scope.
