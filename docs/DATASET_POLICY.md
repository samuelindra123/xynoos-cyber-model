# Dataset Governance Policy

No dataset is included in this foundation. A source being publicly accessible does not establish permission to download, use for training, or redistribute it.

Before future use, record source provenance, acquisition date, applicable license and terms, permitted use, transformation steps, version or checksum, and responsible maintainer. Review licensing and privacy risks before acquisition or publication.

Dataset preparation should include documented cleaning and deduplication, sensitive-information filtering, quality measurement, dataset versioning, and strict train/validation/test separation. Evaluation data must be protected from training contamination, with overlap checks and documented limitations.

Do not publish credentials, personal data, private repository contents, unauthorized proprietary information, or restricted content. Cybersecurity material must be lawfully obtained, appropriately licensed, and handled for legitimate defensive, educational, or authorized research. Use isolated controlled environments for sensitive evaluation.

Future datasets should be reproducible without redistributing data where licenses prohibit it. Publish manifests and processing code only when lawful, and keep large data outside Git using an appropriate reviewed storage approach.

Dataset licenses, third-party documentation licenses, model-weight licenses, dependency licenses, and generated research artifact terms are reviewed separately from the source code's Apache-2.0 license.
