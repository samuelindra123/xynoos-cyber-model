# TASK 006-F — Scientific Data and Evaluation Audit

**Outcome: Objective A v3 is a reproducible synthetic software-control candidate, not an admitted learning benchmark. Public test exposure, validation size, source coverage, and absent quality review block scientific acceptance. Objective B remains separate.** No external data was downloaded.

## Authoritative corpus and identity

The integrated candidate is Objective A v3 from PR #28, exact head `f122f4e283b06d2cd20dd544fc0b18af4c8b6c98`. It supersedes v2 for the current candidate; v2 is retained as historical evidence. V3 contains 60 project-authored synthetic rows split 24/18/18 among train/validation/public test. Each split contains eight/six/six examples from each of three task families: parity, integer comparison, and modular progression. The corpus uses one authored source/date and eight templates partitioned across the splits. Task inputs and group IDs are unique by split.

| Artifact | Digest/identity |
|---|---|
| Dataset ID | `6b8d4b140c8d20636b206cb6259a1789a2734f5a13d7fd234e3d39aad3037c87` |
| Manifest SHA-256 | `db2577c6ed7f0f74c69b643dc9d74683c76084ddf8dc21bddc0ce072c56562b8` |
| Documents SHA-256 | `76842cc832ec49fe807e5b7f6bbdbe0a4ba08f96190629040512d97be9a8874a` |
| Tokenizer | `byte-bpe-v1`, 32 train-only merges, vocabulary 292 |
| Tokenizer SHA-256 | `9ac03af34eaa496a54dc7721f83107ecdd14c26c643fb4c51d34de9afe77d6d0` |
| Tokenizer fit-record SHA-256 | `28d037fb18b57231ede816cd2bb673230b211c92f3c5bf611287b55915b8d203` |

The project audit scripts verify payload checksums, canonical group hashes, deterministic split assignment, family counts, templates, and train-only tokenizer fitting. An additional auditor-authored standard-library cross-check independently recomputed payload/group/split identities: 60 unique inputs, zero exact cross-split duplicates, zero task-input overlap, and maximum normalized cross-split trigram Jaccard `0.461538462` (pair `oa3-030` / `oa3-045`). Both project audit scripts and Rust v3 manifest tests pass. The threshold check only bounds the tested similarity metric; it does not establish semantic independence.

## Baseline and test exposure

The unchanged-weight baseline used seed 107 and state identity `470fb79f5ec91c7e805115c19b90cb198586ff760b03ed7fafe4742f19f57a44`. Full-sequence train, validation, and public-test cross-entropies are `5.702619792`, `5.700838645`, and `5.699661671` over 197, 199, and 172 target tokens. Answer-bearing validation/test losses are `5.747362546` / `5.692370273` over only 28/27 positions. Ten answer spans begin within merged BPE tokens, so these are not strict answer-only metrics.

The five unchanged-weight evaluators cost `3,124,414,056` forward-work units. The public test is committed in Git and was read to produce its baseline. It is therefore exposed and cannot serve as an untouched/locked holdout or be used to claim prospective final generalization. This does not prove training leakage into the model; it means evaluation results have already been observed and cannot be treated as blind.

## Finding F-SCI-01 — inadequate validation power and provenance

- **Severity:** Critical for scientific admission.
- **Component:** Objective A v3 split, learning-rate selection, and task validity.
- **Reproduction:** Inspect `datasets/task006-objective-a-v3`, run `python3 tools/task006_objective_a_v3.py --check`, `python3 tools/task006_objective_a_v3_audit.py --check`, and compare v3 manifest splits with the v3 baseline report.
- **Evidence:** Only 18 validation examples (six per family), one author/source/date, eight simple templates, no independent human review, and no post-update validation result. Different templates per split reduce verbatim template overlap but preserve the same three simple task rules.
- **Root cause:** A small synthetic control was authored for deterministic software integration; it was not designed or reviewed as a powered model-selection benchmark.
- **Remediation:** Keep it labeled a controlled Objective A smoke. If used for learning claims, independently review task validity and increase examples through independently authored or generator-held-out variants; split by family/template/source group, quantify uncertainty with paired per-example metrics and family/template clustering, and predeclare a practical effect margin before evaluating candidate learning.
- **Verification:** Corpus reproducibility and the specified lexical leakage checks passed; adequacy for learning-rate comparisons and broad generalization did not pass and remains unverified.

## Finding F-SCI-02 — public test is no longer blind

- **Severity:** Critical for final evaluation claims.
- **Component:** Objective A public test protocol.
- **Reproduction:** Read v3 test files and the unchanged-weight baseline record.
- **Evidence:** Eighteen test rows are public in Git; all nine test batches were evaluated for baseline (172 full-sequence tokens; 27 answer-bearing positions).
- **Root cause:** Test evaluation was included in baseline plumbing before test isolation was established.
- **Remediation:** Retire this split as a blind test. Freeze a fresh versioned test only after protocol and model-selection choices are locked; keep it out of public Git and do not access it during selection. Alternatively, report it only as an already-observed diagnostic with explicit disclosure.
- **Verification:** Exposure is confirmed. No replacement hidden test has been created.

## Proposed defensible protocol (not yet admitted)

1. Keep Objective A limited to controlled task learning; do not infer cybersecurity intelligence or use Objective B data.
2. Freeze reviewed train/validation data, tokenizer, exact model seed, and one optimizer/schedule configuration. Any hyperparameter selection uses validation only, with a separately reserved prospective test.
3. Group split by task generator, template, and source where feasible; independently review prompt/answer validity; retain exact content hashes and train-only tokenizer provenance.
4. Use full-sequence token-weighted validation loss as the primary metric unless token alignment is corrected to make answer-only measurement exact.
5. Before training, preregister a minimum practical validation improvement over the unchanged-weight baseline, a paired confidence interval computed over held-out examples with dependence clustered by family/template, and a no-catastrophic-family-regression condition. Require the lower confidence bound to exceed the preregistered effect margin, finite losses, and repeated deterministic agreement. The margin and sample size must be selected after independent task review; neither is frozen here.
6. Report test evaluation once, after all choices are frozen, from the untouched holdout; disclose the full access log.

**Acceptance thresholds:** no evidence-based numerical quality threshold is currently frozen. Resource thresholds are fixed separately in the compute report. Inventing a post hoc CE threshold from this tiny baseline would manufacture a pass.
