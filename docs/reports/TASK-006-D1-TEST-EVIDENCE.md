# TASK 006-D1 — Test Evidence

**Implementation revision:** `da99cba3016b7fe564c42e65d8d2babe44b3b98f`  
**Working tree:** clean during full debug and release workspace runs and the baseline run.

## Executed Rust checks

| Check | Result |
|---|---|
| `cargo test --workspace --locked --offline -- --show-output` | Passed, 178 tests/doctests, 0 failures |
| `cargo test --workspace --release --locked --offline -- --show-output` | Passed, 178 tests/doctests, 0 failures |
| `cargo test -p xynoos-training --locked --offline --test cli_process detached_process_survives_controller_exit_supports_stop_status_resume_and_crash_recovery -- --exact --show-output` | Passed; stop, status, resume, and crash-recovery lifecycle retained |
| `cargo fmt --all -- --check` | Passed |
| `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` | Passed |
| `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked --offline` | Passed |
| `python3 tools/check_repository.py` | Passed; 194 tracked files checked |

Both Rust test profiles included the 5 auditor-authored C3 process tests, R2
lock and sidecar tests, 6 evaluation tests, the initial baseline test, and the
training test-split rejection. The initial full debug run before the coverage
fix failed the detached lifecycle stop assertion; it exposed the evaluator's
hard error at the batch cap. The final debug and release runs above were
executed after the fix and both passed.

## Independent fixtures and controls

| Command | Result |
|---|---|
| `python3 tools/numerical_oracle.py --check` | 295 cases, 2,670 outputs verified |
| `python3 tools/gradient_oracle.py --check` | 25 gradient rows and 101 learning rows verified |
| `python3 tools/adamw_reference.py --check` | 33 records verified; SHA-256 `f1839d7c12a0c9b3b0e75b73ec3b7e53b1eb1c67191c64e6053eb9a199244e61` |
| `python3 tools/task006_objective_a.py --check` | Objective A fixture SHA-256 `63050baf97f301c5ca6a3ad07a1a632bef77938b69f0317622738b5bbc00fa33` |
| `python3 tools/task006_evaluation_oracle.py --check` | Independent token-weighted aggregation fixture verified |
| `python3 tools/task006_evaluation_audit.py --check` | Split/tokenizer audit SHA-256 `182a4ae677a9ae0545690e48b99dab63ecc94b2ca051ce735eb2eed2abaa62a8` |
| `python3 tools/tokenizer_data_oracle.py --check` | Task 004 manifest, tokenizer, and split negative controls verified |
| `python3 tools/transformer_oracle.py --check` | Transformer reference and negative controls verified |
| `python3 tools/transformer_oracle.py --candidate --check` | Candidate Transformer fixture and negative controls verified |

The clean baseline command and its per-split metrics are recorded in the
[scientific evaluation report](TASK-006-D1-SCIENTIFIC-EVALUATION.md). The
test's source identity was clean and matched the implementation revision
listed above.

## Evidence boundaries

These checks establish software behavior for the checked-in fixtures. They do
not establish broad learning quality, cybersecurity capability, private
checkpoint storage, external detached execution, or GPU/Colab lifecycle. C3's
separate reviewer gate remains open because the implementation and verification
were produced by the same agent. Hosted CI evidence for the final D1 PR revision
is recorded separately after dispatching Foundation CI; no training workflow
was dispatched.
