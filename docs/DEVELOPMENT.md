# Development Guide

## Languages and scope

Rust and C are the primary implementation languages. Python is appropriate for bounded research scripts, dataset processing, and validation. This foundation milestone contains no model implementation.

## Lightweight local work

Suitable local tasks include source and documentation editing, formatting, static analysis, lightweight unit tests, small numerical and tokenizer experiments, and bounded smoke or inference tests. Avoid large training, large dataset downloads, multi-gigabyte artifacts, heavy dependency installs, prolonged CPU saturation, and unbounded memory allocation.

## Validation and research records

Record commands actually run and their outcomes. Future numerical changes should include correctness checks; experiments should capture source revision, configuration, data version, seeds where applicable, and resource bounds. Keep machine-specific reports and absolute personal paths out of tracked files.

See [compute policy](COMPUTE_POLICY.md), [dataset policy](DATASET_POLICY.md), and [evaluation guidance](EVALUATION.md).
