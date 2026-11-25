# Evaluation Principles

No model or benchmark result is available in this milestone. Future evaluation should be designed before experiments and report its scope, data provenance, split strategy, contamination controls, metrics, configuration, and resource limits.

Evaluation should examine numerical correctness, task accuracy, reliability across repeated or varied inputs, uncertainty handling, and responsible cybersecurity behavior. Domain tests should be technically reviewed and avoid enabling unauthorized actions. Include baselines and limitations; do not present a single score as proof of broad capability.

The tensor foundation is evaluated through hand-calculated tests, bounded exact properties, and a deterministic independent Python oracle. Both debug and release builds run the fixture comparisons and verify that the Rust consumer rejects malformed records, unknown operations, invalid binary32 encodings, wrong output shapes or counts, and finite results outside the fixed tolerance. The implementation report records tolerances, supported numerical claims, and known limitations. [TASK 002-B implementation report](reports/TASK-002-B-IMPLEMENTATION-REPORT.md)

Performance studies should define the workload, environment category, measurement method, warm-up and repetition policy, memory use, and variance. Preserve raw results and scripts where licensing and privacy allow. Reproduce results from a recorded source revision and dataset version when practical.
