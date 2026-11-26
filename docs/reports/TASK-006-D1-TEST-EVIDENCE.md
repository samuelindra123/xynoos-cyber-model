# TASK 006-D1 — Test Evidence

| Evidence identity | Value |
|---|---|
| Historical D1 implementation revision | `da99cba3016b7fe564c42e65d8d2babe44b3b98f` |
| Previous full Objective A v2 baseline source | `4a83c68a199d8844e2d01038bd5d94d8dc377959` |
| Latest Objective A v2 candidate head | `8d70584fda7d269762d70681d308936b045887bf` |
| Answer-bearing baseline test source | `d8e4d7df6d5a919acf2002783ed59af39e86d67f` |
| Working tree | Clean for the answer-bearing debug and release tests at the source revision above. |

## Executed Rust checks

| Check | Result |
|---|---|
| `cargo test --workspace --locked --offline -- --show-output` | Passed, 178 tests/doctests, 0 failures |
| `cargo test --workspace --release --locked --offline -- --show-output` | Passed, 178 tests/doctests, 0 failures |
| `cargo test -p xynoos-training objective_a_v2_answer_bearing_baseline_matches_audit_position_fixture --locked --offline -- --show-output` | Passed at `d8e4d7d`, 1 test; 48 answer-span masks matched the audit-position fixture; 78.88 s debug test time |
| `cargo test -p xynoos-training --release objective_a_v2_answer_bearing_baseline_matches_audit_position_fixture --locked --offline -- --show-output` | Passed at `d8e4d7d`, 1 test; 10.23 s test time after compilation |
| `cargo check -p xynoos-training --tests --locked --offline` | Passed at the implementation candidate before its clean-source run |
| `python3 tools/task006_objective_a_v2_audit.py --check` | Passed; answer-position fixture SHA-256 `1a7b328bf1311c59bc21c1e21d0eaed1e0c8340b886742c204fdb9180148aa5c` |
| `cargo test -p xynoos-training --locked --offline --test cli_process detached_process_survives_controller_exit_supports_stop_status_resume_and_crash_recovery -- --exact --show-output` | Passed; stop, status, resume, and crash-recovery lifecycle retained |
| `cargo fmt --all -- --check` | Passed |
| `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` | Passed |
| `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked --offline` | Passed |
| `python3 tools/check_repository.py` | Passed at the historical revision; 194 tracked files checked |

The historical full-workspace local runs above covered 178 tests/doctests at
`da99cba`. The previous v2 candidate's exact-hosted Foundation CI run
`45823059589` checked out `4a83c68a199d8844e2d01038bd5d94d8dc377959` and
passed all 33 steps with 181 debug and release checks. The latest candidate's
exact-head Foundation CI run `45831165420` checked out
`8d70584fda7d269762d70681d308936b045887bf` and passed all 33 steps, including
the updated 182-check debug/release totals, formatting, Clippy, independent
oracles, resource checks, and Rust documentation. The five C3 auditor-authored
tests also passed locally in both profiles on the previous clean candidate.
`python3 tools/check_repository.py` checked 255 tracked files on the latest
candidate.

The answer-bearing follow-up is an additional unchanged-weight baseline on the
same manifest-backed corpus, not a training experiment. Its masks were
cross-checked against a separately implemented Python standard-library audit;
this code-level comparison does not replace external human review. At source
`d8e4d7d`, it reports 89/46/46 answer-bearing targets for train/validation/test
and finite loss on every split. The test split is still public and accessed
only by the explicitly tagged baseline evaluator.

The historical full-workspace profiles included the 5 auditor-authored C3
process tests, R2 lock and sidecar tests, 6 evaluation tests, the initial
baseline test, and the training test-split rejection. The initial full debug
run before the coverage fix failed the detached lifecycle stop assertion; it
exposed the evaluator's hard error at the batch cap. The final historical
debug and release runs were executed after the fix and both passed. The latest
The previous candidate's 181-check hosted run included the additional v2
dataset and native trainer plumbing controls. Exact-head run `45831165420`
confirmed the answer-bearing test and full 182-check workspace suites.

## Independent fixtures and controls

| Command | Result |
|---|---|
| `python3 tools/numerical_oracle.py --check` | 295 cases, 2,670 outputs verified |
| `python3 tools/gradient_oracle.py --check` | 25 gradient rows and 101 learning rows verified |
| `python3 tools/adamw_reference.py --check` | 33 records verified; SHA-256 `f1839d7c12a0c9b3b0e75b73ec3b7e53b1eb1c67191c64e6053eb9a199244e61` |
| `python3 tools/task006_objective_a.py --check` | Objective A fixture SHA-256 `63050baf97f301c5ca6a3ad07a1a632bef77938b69f0317622738b5bbc00fa33` |
| `python3 tools/task006_objective_a_v3.py --check` | 66 generated files; manifest SHA-256 `db2577c6ed7f0f74c69b643dc9d74683c76084ddf8dc21bddc0ce072c56562b8` |
| `python3 tools/task006_objective_a_v3_audit.py --check` | Independent standard-library audit fixture SHA-256 `591627a107921156b51ca9a54b94caf748db73ac07430ff7ba35f354336dcc9b` |
| `cargo test -p xynoos-data --locked --offline -- --show-output` | Passed on the v3 candidate; includes semantic-group same-split and cross-split negative controls and deterministic 24/18/18 v3 streams |
| `cargo test -p xynoos-training objective_a_v3_manifest_unchanged_weight_baseline --locked --offline -- --show-output` | Passed at clean source `8dcfd06125bff3957fad5122af69a12d5a0aecc6`; one test, 91.38 seconds debug; full baseline and answer-bearing validation/test evaluated with unchanged weights |
| Full debug workspace at clean source `053b2703632921d9869215a2dad664d2f2b41733` | Passed; 185 tests/doctests, 0 failures; training crate unit suite 219.53 seconds, including v3 baseline |
| Full release workspace at clean source `053b2703632921d9869215a2dad664d2f2b41733` | Passed; 185 tests/doctests, 0 failures; training crate unit suite 25.08 seconds |
| `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` | Passed on the exact source diff committed at `053b270` |
| `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked --offline` | Passed |
| `python3 tools/task006_evaluation_oracle.py --check` | Independent token-weighted aggregation fixture verified |
| `python3 tools/task006_evaluation_audit.py --check` | Split/tokenizer audit SHA-256 `182a4ae677a9ae0545690e48b99dab63ecc94b2ca051ce735eb2eed2abaa62a8` |
| `python3 tools/tokenizer_data_oracle.py --check` | Task 004 manifest, tokenizer, and split negative controls verified |
| `python3 tools/transformer_oracle.py --check` | Transformer reference and negative controls verified |
| `python3 tools/transformer_oracle.py --candidate --check` | Candidate Transformer fixture and negative controls verified |

The clean baseline command and its per-split metrics are recorded in the
[scientific evaluation report](TASK-006-D1-SCIENTIFIC-EVALUATION.md). The
historical v1 fixture test's source identity was clean and matched
`da99cba3016b7fe564c42e65d8d2babe44b3b98f`; the v2 baseline evidence is
recorded separately at the latest candidate revision above.

## Evidence boundaries

These checks establish software behavior for the checked-in fixtures. They do
not establish broad learning quality, cybersecurity capability, private
checkpoint storage, external detached execution, or GPU/Colab lifecycle. C3's
separate reviewer gate remains open because the implementation and verification
were produced by the same agent. Hosted CI evidence for the final D1 PR revision
is recorded separately after dispatching Foundation CI; no training workflow
was dispatched.

The v3 baseline source revision produced full-sequence train/validation/public
test losses of 5.702619792/5.700838645/5.699661671 and answer-bearing
validation/public-test losses of 5.747362546/5.692370273. The five evaluation
passes totaled 3,124,414,056 measured forward-work units. They reused initial
model-state digest
`470fb79f5ec91c7e805115c19b90cb198586ff760b03ed7fafe4742f19f57a44`; test
reads were explicitly tagged. No optimizer update was performed for these
baseline metrics. The exact metrics, dataset identities, and limitations are
recorded in the [v3 dataset report](TASK-006-D1-OBJECTIVE-A-V3-DATASET.md).

At clean source `74831336ae081ee65b3f18071208ed812bd73f66`, the new v3 data
stream test passed with 24 independently padded `[1, 32]` training batches,
197 supervised targets, zero dropped tails, and deterministic ordering. The
focused `objective_a_v3_manifest_runs_two_bounded_trainer_updates` debug test
passed again at clean source `698541f2273ee511c829c9aa0c559139d37a43c2`
(one test, 17.47 seconds wall time). It completed two updates using only the
train split, recorded two one-batch partial validation events, checked finite
loss and a parameter change, wrote a checkpoint, and verified no test access.
The runtime reported 19 consumed and supervised train tokens, 18 validation
tokens, 825,200,872 cumulative work units, 17.2875 seconds active runtime,
42,439,416 bytes owned-memory peak, 30,682,296 bytes autograd-ledger peak,
and 37,212,160 bytes process high-water. A linear twelve-fold projection for
24 such updates is 9,902,410,464 work units, above the 8-billion ceiling; it
is not a full-workload measurement and does not include the initial baseline
or full validation coverage. This is an isolated integration control, not a
TASK 006-D experiment or evidence that two updates are scientifically
sufficient.

Exact-head Foundation CI for the D1 v3 candidate passed at
`464adc382b41396bd395a8db4248f9fbc57aecdf` in [run 45855234814](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45855234814).
All 33 workflow steps succeeded, including 187 debug and 187 release checks,
independent numerical and corpus fixtures, the 8-billion work policy checks,
repository hygiene, and Rust documentation. Earlier run `45853200528` failed
only because its accepted-check guard still expected 185 after the two new
v3 controls; the guard was updated to 187 and the corrected exact-head run
passed.

The latest local workspace runs at source `053b2703632921d9869215a2dad664d2f2b41733`
passed 185 debug and 185 release checks. The 101-observation deterministic
learning trajectory matched bit-for-bit across profiles with the preserved
SHA-256 `e50d201a133838fe64158e0dcf94df894efcf17241f1f16e52232a4e75d41c6a`.
The local environment lacks both Ruby and PyYAML, so YAML parsing was not run
locally; the exact-head Foundation CI workflow performs that check using its
pinned runner image.
