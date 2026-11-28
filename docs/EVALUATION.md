# Evaluation Principles

No trained language model or language-model benchmark result is available. The accepted numerical and autodiff engine is evaluated through deterministic arithmetic and gradient oracles, contract tests, and a bounded synthetic linear-regression learning proof. These validate specific engine behavior; they do not establish language-model capability. Future model evaluation should be designed before experiments and report its scope, data provenance, split strategy, contamination controls, metrics, configuration, and resource limits.

Evaluation should examine numerical correctness, task accuracy, reliability across repeated or varied inputs, uncertainty handling, and responsible cybersecurity behavior. Domain tests should be technically reviewed and avoid enabling unauthorized actions. Include baselines and limitations; do not present a single score as proof of broad capability.

The intended specialization is defensive cybersecurity research and engineering, Linux and operating-system internals, systems programming, networking and protocols, secure coding, and computer architecture and embedded systems. Future behavioral evaluation should measure technical accuracy, depth of explanation, honest uncertainty, reproducibility, unnecessary refusal of legitimate research requests, and unsafe assistance. These are objectives only; no language model exists yet and none of these behavioral properties is claimed as achieved. See the [efficiency-first model engineering direction](design/EFFICIENCY-FIRST-MODEL-ENGINEERING.md).

The tensor foundation is evaluated through hand-calculated tests, bounded exact properties, and a deterministic independent Python oracle. Both debug and release builds run the fixture comparisons and verify that the Rust consumer rejects malformed records, unknown operations, invalid binary32 encodings, wrong output shapes or counts, and finite results outside the fixed tolerance. The reports record tolerances, observed rejection diagnostics, and known limitations. [TASK 002-B implementation report](reports/TASK-002-B-IMPLEMENTATION-REPORT.md) · [TASK 002-D remediation report](reports/TASK-002-D-REMEDIATION-REPORT.md)

First-order reverse-mode gradients are checked against independent analytical and finite-difference coordinates, with explicit negative controls. The native learning proof uses five fixed synthetic samples and transactional SGD for exactly 100 updates; the checked-in fixture captures 101 states and CI requires bitwise agreement between debug and release trajectories. This experiment is evidence for that regression case only. See the [TASK 003 implementation report](reports/TASK-003-B-IMPLEMENTATION-REPORT.md) and [independent verification report](reports/TASK-003-C-INDEPENDENT-VERIFICATION-REPORT.md).

Performance studies should define the workload, environment category, measurement method, warm-up and repetition policy, memory use, and variance. Preserve raw results and scripts where licensing and privacy allow. Reproduce results from a recorded source revision and dataset version when practical.

## TASK 006 deterministic evaluation

The native training library now exposes a bounded multi-batch evaluator. Its
limits cover batch count, consumed tokens, forward work, source memory, and
batch positions. It requires a stable source cursor, validates train and
validation order identities, reports token-weighted cross entropy, and records
the dataset, split, model state, tokenizer, evaluation policy, work, memory, and
whether the test split was accessed. If the batch cap stops before source EOF,
the report sets `coverage_complete` to false; such a report is a bounded prefix,
not a full-split baseline or selection metric. Empty-supervision batches are counted but
do not contribute a loss; a split with no supervised targets reports no loss
or perplexity. Test batches are rejected by periodic validation and by the
trainer's training source admission.

Use `evaluate_standard_dataset_split` for manifest-backed evaluations. It
validates the run identity against the admitted dataset and tokenizer, then
constructs the deterministic split stream itself. Generic batch sources remain
available for software fixtures; their caller is responsible for supplying
the contents represented by the recorded split identity.

Initial-weight baseline evaluations use one immutable model with no optimizer
and report train, validation, and test separately. Test access is explicitly
marked. Evaluation policy is part of the run configuration and new checkpoint
records; the checkpoint reader retains the previous default-policy encoding.
The independent standard-library aggregation fixture checks token weighting
against a case where an unweighted mean gives a different answer.

The checked-in Objective A evaluation fixture is a software control, not a
training dataset or evidence of cybersecurity capability. It has only two
examples per split and 64 supervised targets per split. Both training and
validation reuse the same template families, and the test split contains one
family. The split audit found no normalized exact duplicates and no
character-trigram pair at the existing 0.82 near-duplicate threshold, but the
largest cross-split trigram similarity is 0.717391. Character n-grams do not
detect semantic or paraphrase leakage. The tokenizer was fit on a separate
train-only Task 004 manifest; that does not prove the learned merges contain no
fragments that overlap Objective A.

This fixture can check deterministic aggregation and a narrow optimizer
learning signal. It is too small to choose a learning rate or support a
generalization claim. Objective B remains unavailable until a separate,
licensed corpus passes provenance and quality review.

Objective A v3 expands the authored control to 60 rows and prevents the same
task-input combination or prompt template from crossing splits. Its audit finds
no cross-split near-duplicate pair at the fixed 0.82 character-trigram
threshold, but this does not prevent memorization of the small public task
inventory. See the [v3 baseline and leakage report](reports/TASK-006-D1-OBJECTIVE-A-V3-DATASET.md).
The initial weights are evaluated without updates; there is no measured v3
learning improvement. A separate human review remains required before any
experiment admission.
