# Development Guide

## Languages and scope

The tensor workspace uses safe Rust and the standard library. Python 3 runs the independent numerical oracle and repository checks. C remains available for future reviewed work.

## Lightweight local work

Suitable local tasks include source and documentation editing, formatting, static analysis, lightweight unit tests, small numerical and tokenizer experiments, and bounded smoke or inference tests. Avoid large training, large dataset downloads, multi-gigabyte artifacts, heavy dependency installs, prolonged CPU saturation, and unbounded memory allocation.

## Tensor validation

Use the current stable Rust toolchain with Cargo, rustfmt and Clippy. CI selects stable explicitly and records its resolved version alongside the Python version. This toolchain policy tracks stable updates across CI runs while the checked-in `Cargo.lock` fixes dependency resolution. Cargo builds and tests run offline after installing the toolchain.

From the repository root, run local checks with one build job.

```sh
cargo +stable fmt --all -- --check
cargo +stable clippy --workspace --all-targets --locked --offline -j 1 -- -D warnings
python3 tools/numerical_oracle.py --check
cargo +stable test --workspace --locked --offline -j 1
```

For an intentional fixture change, use `python3 tools/numerical_oracle.py --write` and review the generated diff. CI checks the committed bytes and runs the oracle comparison in debug and release, including doctests and a Rust documentation build. It also runs the original repository checks and scans tracked files for obvious privacy and artifact issues. [Implementation report](reports/TASK-002-B-IMPLEMENTATION-REPORT.md)

## Validation and research records

Record commands actually run and their outcomes. Future numerical changes should include correctness checks; experiments should capture source revision, configuration, data version, seeds where applicable, and resource bounds. Keep machine-specific reports and absolute personal paths out of tracked files.

See [compute policy](COMPUTE_POLICY.md), [dataset policy](DATASET_POLICY.md), and [evaluation guidance](EVALUATION.md).
