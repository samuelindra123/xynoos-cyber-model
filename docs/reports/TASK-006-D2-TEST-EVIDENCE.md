# TASK 006-D2 — Test Evidence

**Implementation revision tested:** `0b798e91b550f30d37683eab2c6bf66382a88c28`.

All results below distinguish local test evidence from provider inspection.
The test binaries used the clean committed source revision shown above.

## Local validation

| Check | Result |
|---|---|
| `cargo fmt --all -- --check` | Passed |
| `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` | Passed |
| `cargo test --workspace --locked --offline -- --show-output` | All 180 counted tests/doctests passed; no failures |
| `cargo test --workspace --release --locked --offline -- --show-output` | All 180 counted tests/doctests passed; no failures |
| `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked --offline` | Passed |
| `python3 tools/check_repository.py` | Passed; 197 tracked files checked |

The workspace tests include both TASK 006-D2 checks: the CLI rejects an
artifact root with group/world access, and a newly published checkpoint has
mode `0600`. They also rerun checkpoint integrity and semantic validation,
journal monotonicity, sidecar fault injection, Objective A baseline, stop,
detached lifecycle, resume equivalence, multiprocess lock contention, owner
crash/restart, and source-attestation controls. The CLI detached-process test
is an isolated deterministic software test. It is not a TASK 006-D training
experiment.

The first debug attempt was run before committing the change; source
attestation correctly rejected its dirty checkout in two tests. After commit,
both clean-checkout tests passed and the full debug suite passed. This was a
test setup issue, not a product failure.

## Independent numerical and negative controls

- Transformer reference oracle: 295 cases / 2,670 outputs; passed.
- Gradient and learning oracle: 25 gradient rows and 101 learning records;
  passed.
- AdamW oracle: 33 records; passed.
- Frozen Objective A fixture and multi-batch token-weighted loss fixture;
  passed.
- Split/tokenizer leakage audit fixture and tokenizer/data negative controls;
  passed.
- Transformer reference and candidate oracles, including numerical negative
  controls; passed.
- Repository privacy/artifact scan; passed.

These checks use standard-library Python oracles and native Rust tests. They
do not depend on an external ML framework or pretrained weights.

## Provider and storage evidence

Repository visibility was read as `PUBLIC`. The training workflow contains no
checkpoint in its Actions artifact upload list, and its log/report artifact
retention is three days. Read-only CLI inspection found Colab CLI `0.8.0` and
the documented runtime/transfer/session commands; no runtime was created,
no file was uploaded or downloaded, and no compute was consumed. GitHub and
Google Cloud storage documentation was inspected for access requirements.

No private remote bucket, Drive permission set, provider credentials, remote
interrupted-transfer behavior, remote digest validation, restoration, or
cleanup was exercised. Local tests prove only the local filesystem boundary
and restart behavior described above.
