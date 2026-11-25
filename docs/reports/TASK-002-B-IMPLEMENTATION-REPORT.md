# TASK 002-B implementation report

This report records the TASK 002-B baseline. The current capacity contract, pinned toolchain, and oracle rejection evidence are documented in the [TASK 002-D remediation report](TASK-002-D-REMEDIATION-REPORT.md).

## Repository and specification

The implementation branch is `feat/task-002-b-numerical-foundation`. Its verified `main` baseline is `2d8d5180c2a5c1489afe58a9137da864d5df26e3`. The specification revision is `2ae7e83c395ef64f22e68b7429ae829710586fc6`. The audit, numerical specification, validation plan, implementation handoff, and risk register were read from that pinned revision. Planning CI run 45426914897 succeeded on that revision. The draft implementation pull request records the exact final head and its Action run separately from this tracked report. [Planning PR](https://github.com/samuelindra123/xynoos-cyber-model/pull/1) · [Pinned specification tree](https://github.com/samuelindra123/xynoos-cyber-model/tree/2ae7e83c395ef64f22e68b7429ae829710586fc6/docs) · [Planning CI](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45426914897)

The pinned specification describes a copying slice constructor. This task also requires consuming owned data, so the API provides both `from_data` for a borrowed slice and `from_vec` for ownership transfer. Checked immutable and mutable getters implement the handoff's access contract. The implementation uses a fixed maximum rank of 32 and a caller-configurable byte limit bounded by an absolute project ceiling of 64 MiB.

## Architecture and scope

`xynoos-tensor` is a single Rust library with no third-party dependencies. It owns dense contiguous row-major data and dynamic shape metadata. Construction validates rank, each canonical suffix product, the byte count, the logical payload limit, and the supplied length before allocation or copying. Arithmetic results revalidate their own output shape and payload. Rust collection reservation errors map to a typed allocation error. Dot and sum return scalar `f32` values without allocating. Full API, complexity, error precedence, scalar and empty semantics, and finite and non-finite behavior are documented in the crate. [Tensor contracts](../../crates/xynoos-tensor/README.md)

The delivered operations are constructors, shape and data inspection, checked flat and multidimensional access, elementwise add/subtract/multiply, scalar multiply, dot, sum, rank-two matmul, and rank-two transpose. This milestone establishes numerical primitives. Model layers, automatic differentiation, tokenization, datasets, training, model weights, checkpoints, and inference remain future work. The reference kernels are sequential CPU operations. SIMD, GPU, C/FFI, optimized backends, and performance measurements remain deferred.

## Oracle design and calibration

`tools/numerical_oracle.py` uses Python's standard library with seed `0x58594e4f`. It emits ASCII TSV at `crates/xynoos-tensor/tests/fixtures/oracle.tsv`, with LF line endings and one header of the form `# xynoos-oracle-v1 seed=0x58594e4f`. Each subsequent row has seven tab-separated fields in this order.

| Field | Encoding |
|---|---|
| Operation | `add`, `subtract`, `multiply`, `scale`, `sum`, `dot`, `matmul`, or `transpose` |
| Left shape and values | Two fields. Comma-separated decimal dimensions and comma-separated eight-digit binary32 hex words. |
| Right shape and values | Same encoding. An unused operand uses two `-` fields. Scale uses a rank-zero operand. |
| Expected shape and values | Same encoding. Scalar results have an empty shape list and one value. |

The marker `-` means an empty list. Hex words preserve infinity signs and signed zero, and the generator encodes NaN with the canonical fixture word `7fc00000`. The generator checks rounding and signed overflow with `struct` and converts inputs to binary32 before evaluating expressions. It uses `math.fsum` for finite reductions, handles mixed infinities, and uses Python row grouping and column zip for matmul. It never runs the Rust numerical kernels. To reproduce, run `python3 tools/numerical_oracle.py --check` at the repository root. Use `--write` to regenerate intentionally. The current fixture contains 295 cases and 2,670 values, with SHA-256 `cdfde7ba9f44f46ba7e757157019730e1124bcd966e796b73286bddf3894fa0f`. Random dimensions are at most eight and random inputs have magnitude at most ten. Exponent and special-value fixtures are separate.

The fixed finite criterion is `abs(actual - expected) <= 1e-6 + 1e-5 * abs(expected)`, evaluated in binary64. For a random dot or matrix cell, there are at most eight exact binary64 products of magnitude at most 100. The usual sequential rounding bound is below `1e-12` before the final binary32 cast. For sums of up to 64 values of magnitude at most 10 the bound is below `5e-12`. Cancellation near zero is covered by the absolute term. The relative term allows for binary32 rounding at ordinary magnitudes. Exact hand tests constrain cases where the generic tolerance would miss a small absolute error, including a subnormal result and half-ULP rounding to signed zero. NaNs are compared by classification and infinities by sign.

The Python oracle approximates the mathematical reduction more closely than sequential accumulation. Its comparisons cover representative bounded inputs and selected edge values. A severe cancellation example `[1e30, 1, -1e30]` has an accurate sum of one and a sequential binary64 result of zero. A dedicated Rust contract test checks this specified operation order. Exact examples also include rectangular matmul `[2,3] × [3,2]` with row-major output `[58,64,83,90]`, and `[2,0] × [0,3]` producing positive zeros. Fixture and property tests are bounded evidence, with no general guarantees of correctly rounded reductions or bitwise equivalence across targets.

## Validation and resource use

Local validation used stable Rust 1.100.0, Python 3.13.5, offline Cargo, and one build job. The executed checks were as follows.

| Command | Result |
|---|---|
| `cargo +stable fmt --all -- --check` | Passed. |
| `cargo +stable metadata --locked --offline --format-version 1` | Passed. Lockfile contains only the workspace crate. |
| `cargo +stable clippy --workspace --all-targets --locked --offline -j 1 -- -D warnings` | Passed. |
| `cargo +stable test --workspace --locked --offline -j 1` | Passed. Three unit tests, ten contract/property tests, one oracle integration test, and one doctest. |
| `cargo +stable test --workspace --release --locked --offline -j 1` | Passed. Same checks in release. |
| `RUSTDOCFLAGS='-D warnings' cargo +stable doc --workspace --no-deps --locked --offline -j 1` | Passed. |
| `python3 tools/numerical_oracle.py --check` | Passed. Python self-checks and byte-for-byte reproduction. |

The initial local test build caught a temporary borrow in the oracle reader. The reader now copies the binary32 scale factor from the temporary parsed record before the temporary drops. Debug, release, Clippy, and documentation checks ran after the correction.

Tests use at most 1 KiB of logical payload per tensor. Shape, byte and payload boundaries are exercised with synthetic extents, arithmetic-only checks, and small caller budgets. The reservation-error mapping test exercises an impossible capacity and does not try to exhaust memory. The 64 MiB ceiling applies to logical value bytes per tensor. Live inputs and outputs, allocator overhead, spare capacity, and metadata add to total process memory. OS overcommit can still terminate a process after a successful reservation, and the payload ceiling places no practical time limit on a large matrix product.

Routine CI uses one GitHub-hosted Ubuntu 24.04 job, a 15-minute timeout, two Cargo build jobs, read-only permissions, and cancellation of redundant runs. It preserves repository foundation, Markdown and YAML checks, scans tracked files for common privacy and artifact issues, records resolved stable tool versions, and performs the locked offline Rust checks, fixture reproduction, and both test profiles. The workflow logs the actual checkout SHA. Build outputs stay on the runner. Local work used no stress tests, large libraries, benchmark runs, dataset downloads, or training workloads. [Workflow](../../.github/workflows/ci.yml)

## Privacy and repository metadata

The tracked-file scan covers common credential formats, personal path patterns, machine inventory lines, selected artifacts and unexpected binary or large files. The implementation diff and evidence were also reviewed for personal information and unrelated assets. These checks have limited coverage and complement human review. GitHub's repository API reported private vulnerability reporting disabled.

The existing description reads “An open-source, CPU-first language model built from scratch for cybersecurity, Linux, operating systems, networking, and low-level systems engineering.” The proposed replacement is “An open-source research project developing a CPU-first language model from scratch for cybersecurity, Linux, operating systems, networking, and low-level systems engineering.” Repository settings and description require separate owner authorization. [Repository](https://github.com/samuelindra123/xynoos-cyber-model)

## Independent handoff

- Review the crate source in `crates/xynoos-tensor/src/` and public API contracts in its README.
- Review `shape.rs` for zero-dimension suffix products and offset overflow, and `ops.rs` for both matrix traversal orders, zero inner dimensions, and sequential accumulation.
- Review `error.rs` and `tensor.rs` for allocation error mapping and validation before resource use.
- Review `crates/xynoos-tensor/tests/tensor_contract.rs`, `crates/xynoos-tensor/tests/numerical_oracle.rs`, and `tools/numerical_oracle.py` for independence, rejection behavior, and the tolerance and special-value comparisons.
- Match the Action checkout line and each required successful step to the draft PR head. Review computed results in both build profiles.
- Confirm the branch is based on the reported `main` revision and the PR remains a draft targeting `main`. Session A provides the independent review.
