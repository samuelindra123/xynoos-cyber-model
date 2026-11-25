# Development Guide

## Languages and scope

The tensor workspace uses safe Rust and the standard library. Python 3 runs the independent numerical oracle and repository checks. C remains available for future reviewed work.

## Lightweight local work

Suitable local tasks include source and documentation editing, formatting, static analysis, lightweight unit tests, small numerical and tokenizer experiments, and bounded smoke or inference tests. Avoid large training, large dataset downloads, multi-gigabyte artifacts, heavy dependency installs, prolonged CPU saturation, and unbounded memory allocation.

## Tensor validation

Run Cargo through rustup from the repository root. The checked-in `rust-toolchain.toml` selects Rust 1.100.0, the minimal profile, rustfmt and Clippy. `rustup show` resolves and installs that toolchain and its required components when needed. CI verifies the compiler release against the same file and records the Rust tool versions. `Cargo.lock` fixes dependency resolution, and Cargo builds and tests run offline after toolchain installation. Update the pin as a reviewed change and rerun the validation suite.

From the repository root, run local checks with one build job.

```sh
rustup show
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked --offline -j 1 -- -D warnings
python3 tools/numerical_oracle.py --check
cargo test --workspace --locked --offline -j 1
```

For an intentional fixture change, use `python3 tools/numerical_oracle.py --write` and review the generated diff. CI checks the committed bytes and runs the oracle comparison and bounded negative controls in debug and release. The controls use in-memory variants of one hand-calculated record and check the expected rejection diagnostics. To run just those oracle tests and see the diagnostics locally, use `cargo test -p xynoos-tensor --test numerical_oracle --locked --offline -j 1 -- --show-output`. CI also runs doctests, the Rust documentation build, the original repository checks, and the tracked privacy and artifact scan. [TASK 002-D remediation report](reports/TASK-002-D-REMEDIATION-REPORT.md)

## Validation and research records

Record commands actually run and their outcomes. Future numerical changes should include correctness checks; experiments should capture source revision, configuration, data version, seeds where applicable, and resource bounds. Keep machine-specific reports and absolute personal paths out of tracked files.

See [compute policy](COMPUTE_POLICY.md), [dataset policy](DATASET_POLICY.md), and [evaluation guidance](EVALUATION.md).
