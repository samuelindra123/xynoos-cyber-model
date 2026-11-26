# Dataset Governance Policy

TASK 004-B includes a tiny project-authored synthetic test fixture. Training use, redistribution and source acquisition require separately reviewed permission. [Dataset implementation and experiment](reports/TASK-004-B-IMPLEMENTATION-REPORT.md)

Before future use, record source provenance, acquisition date, applicable license and terms, permitted use, transformation steps, version or checksum, and responsible maintainer. Review licensing and privacy risks before acquisition or publication.

Dataset preparation should include documented cleaning and deduplication, sensitive-information filtering, quality measurement, dataset versioning, and strict train/validation/test separation. Evaluation data must be protected from training contamination, with overlap checks and documented limitations.

Do not publish credentials, personal data, private repository contents, unauthorized proprietary information, or restricted content. Cybersecurity material must be lawfully obtained, appropriately licensed, and handled for legitimate defensive, educational, or authorized research. Use isolated controlled environments for sensitive evaluation.

Future datasets should be reproducible without redistributing data where licenses prohibit it. Publish manifests and processing code only when lawful, and keep large data outside Git using an appropriate reviewed storage approach.

Dataset licenses, third-party documentation licenses, model-weight licenses, dependency licenses, and generated research artifact terms are reviewed separately from the source code's Apache-2.0 license.

The provenance fields, source-category review questions, deterministic split controls, rejection diagnostics, and bounded streaming contract are implemented at reference scale and independently verified against the [TASK 004 dataset and streaming design](design/TASK-004-DATASET-AND-STREAMING-SPEC.md). A source marked pending or with unknown training permission is not eligible for tokenizer fitting or training. The accepted implementation's experiment uses only project-authored synthetic fixtures; it does not authorize external corpus acquisition. See the [TASK 004 final acceptance report](reports/TASK-004-D-FINAL-ACCEPTANCE-REPORT.md) for evidence and limitations.

The TASK 006 Objective A v2 corpus is project-authored and versioned in Git as a small synthetic software control. Its records and tokenizer are reproducible from the checked-in generator, and its report documents hash-based splits, train-only tokenizer fitting, task overlap, and high template similarity. It is not Objective B, cybersecurity knowledge, or a protected test set. Training admission still requires the separate human quality review recorded in its report.
