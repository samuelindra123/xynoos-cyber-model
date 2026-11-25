# TASK 006 — Data and Compute Research

**Research date:** 2026-11-25
**Status:** literature, repository evidence, and candidate-source review. No external corpus was downloaded and no training was run.

## 1. Existing assets and suitability

The accepted tokenizer/data experiment is a deterministic engineering fixture, not a capability corpus. The fixture has 16 project-authored documents: ten train, three validation, and three test records. It has 1,033 bytes in total. The train split contains 631 bytes and encodes as 455 BPE tokens (0.721 tokens/byte); validation contains 189 bytes and 156 BPE tokens (0.825 tokens/byte); test contains 213 bytes and 164 tokens (0.770 tokens/byte). Exact byte round trips pass. BPE was fitted on ten short train records with 32 merges, yielding the accepted vocabulary size 292.

The domain labels cover short English and Indonesian examples, programming snippets, Linux paths, networking, and Unicode. This is useful for bounded schema, tokenizer, batching, masking, and deterministic-fixture checks. It does not supply representative prose, long documents, current cybersecurity coverage, technical-answer quality, or enough independent examples to estimate generalization.

The two buffer documents are an exact duplicate. The data fixture assigns their shared SHA-256 duplicate group to the same split, so the exact duplicate cannot straddle train and validation/test. It retains both records, however, so training can see duplicated text twice. Near-duplicate, translated, normalized, and semantic leakage detection is not implemented. The current split is deterministic hash grouping at a nominal 80/10/10 ratio; with only 16 records the realized 10/3/3 split is statistically weak.

The 292-token BPE is learned from a tiny synthetic corpus and has only 32 merges above the 260-token byte/special baseline. Token efficiency varies materially by split even in this fixture. It has not been measured on a representative mix of English, Indonesian, source code, Linux/networking docs, and defensive cybersecurity material. The 128-token context counts tokens, not bytes or words. The existing task fixture cannot establish a useful tokenization rate or coverage over cyber terms.

No tokenizer change is proposed. The fixed target remains V=292. If evidence later requires vocabulary V', the tied embedding/output table changes parameter count by 96 * (V' - 292), while token IDs, tokenizer digest, model inventory, parameter count, all checkpoints, and model identity compatibility change. Such a decision must be a separately authorized architecture/model-version proposal; it must not be applied to xynoos-cyber-0.1 silently.

The data provenance schema records source references, revisions, hashes, permissions, transformations, splits, and review fields, and rejects unknown or unapproved training permissions. Those fields are claims to review, not legal verification or source signatures. The repository synthetic fixture carries the project Apache-2.0 label, but no external data source has been approved for a model run.

## 2. Two different research objectives

### Objective A — training-engine correctness

Create a deterministic project-authored miniature technical language task. Generate short defensive systems/security statements and structured examples from a published grammar with parameterized subjects, actions, conditions, and remediations. Keep content educational and non-operational; avoid secrets, personal data, exploit payloads, and real vulnerable targets. Record its source as project-authored, Apache-2.0, with a pinned generator version and corpus digest.

Split by generator rule/template family and combination before rendering examples. Hold out entire combinations and some template families for validation/test, rather than randomly splitting near-identical rendered strings. Exact byte duplicates must be detected and removed from the training contribution or grouped before split. Report both. Use the existing 292 vocabulary and fixed model architecture. Target a measured property: stable finite AdamW updates; changed parameter and moment bits; an independently reproduced loss trajectory; lower training and held-out next-token loss than the random-init baseline on held-out combinations; checkpoint/resume equivalence; and correct token accounting. Thresholds and the sample list must be frozen before execution.

This objective verifies a native Transformer can learn a small, project-owned regularity. Passing it is not evidence of broad language ability, cyber expertise, secure advice, or production usefulness.

### Objective B — early language-model capability

Separately assemble a versioned, substantially broader corpus that provides enough natural technical prose, code/doc context, English and Indonesian material, and defensive cybersecurity coverage to ask a capability question. Define source/time/vendor/language splits before tokenizer fitting or training, reserve a locked test set, and use decontamination at exact, normalized, and near-duplicate levels. Record token/byte and truncation distributions per source and language. Measure source diversity, writing quality, stale/conflicting guidance, unsafe operational content, PII/secrets, licensing, and factual answerability.

An early capability evaluation should include a held-out next-token test set plus task-style questions with answer keys reviewed by domain-qualified humans. Separate memorization probes from novel combinations; include source-held-out and time-held-out cases. Score technical correctness, unsupported claims, safe/defensive framing, and uncertainty. A lower validation perplexity is not sufficient to claim useful cybersecurity knowledge.

Objective B may not be reduced to fit a GitHub runner if that makes its coverage or evaluation scientifically meaningless. First measure the work and resource bottleneck; then choose an authorized environment or redesign the question.

## 3. Public source candidates and terms

The following are candidates for a later source review, not approved downloads or a committed corpus.

| Candidate | Published terms/evidence | Suitability and constraints |
|---|---|---|
| CISA Known Exploited Vulnerabilities catalog and official cisagov/kev-data mirror | The official CISA mirror README says the data repository is CC0 and points to the matching CISA license file. U.S. law excludes copyright protection for qualifying federal employee works domestically, but that premise does not cover contractor or third-party text automatically. | Structured, defensive prioritization fields with useful current coverage but little explanatory prose. Pin a feed revision and checksum. Review each field for upstream text and keep CISA/DHS marks out of model attribution. Do not equate catalog availability with all linked advisory rights. |
| NIST-authored cybersecurity publications and datasets | NIST explains that employee-created non-SRD works generally fall under 17 U.S.C. 105 in the U.S., asks for attribution, and warns that some NIST publications are third-party-authored or third-party-copyrighted. The Standard Reference Data Act can allow U.S. copyright in specific NIST SRD products. | Strong defensive background and standards/control explanations. Select only items whose authoring, SRD status, third-party figures/text, and applicable notice are resolved. Store exact title/version/DOI and any source citation. Do not treat the entire NVD as uniformly public domain. |
| CVE records | MITRE's current CVE Terms of Use grants a broad copyright license to reproduce and prepare derivatives, but requires its copyright designation and license in copies. CVE records are submitted by many CNAs; review the exact record and any incorporated references. | Identifiers and concise descriptions support vulnerability vocabulary and held-out temporal coverage; standalone records are not rich teaching prose. Track CVE version, CNA/source, and required notices. |
| NVD vulnerability/CPE/CWE enrichments | NVD describes itself as a NIST product and says it uses CVE and publicly available material supplied by vendors, researchers, and coordinators. NIST's own-data statement, the current NVD API terms, the CVE Terms of Use, and embedded third-party sources must be considered separately. | Useful for structured metadata, severity, weaknesses, and product names; not a uniformly NIST-authored text corpus. Prefer a pinned feed and exclude free-form third-party text unless the record's rights are resolved. The [NVD general description](https://nvd.nist.gov/general), [NIST data/work terms](https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications), and [NVD API terms](https://nvd.nist.gov/developers/terms-of-use) are primary review points. |
| OWASP ASVS and Cheat Sheet projects | Individual OWASP projects publish CC BY-SA 4.0 terms; examples are ASVS and the API Security Project. CC BY-SA requires attribution and share-alike on shared adaptations and may not address all privacy/publicity rights. | Defensive, procedural explanations and checklists. Pin the repository revision, preserve attribution/change notices, inspect each project and embedded third-party content, and resolve whether any dataset/model artifact distribution would trigger obligations before use. |
| Information Security Stack Exchange data dump | Stack Exchange says public contributions from 2018-05-02 onward use CC BY-SA 4.0; earlier content may carry older CC BY-SA versions. The dump as a compilation also has its own stated license. | Diverse Q&A but heterogeneous quality, personal identifiers, copied excerpts, unsafe operational examples, and versioned attribution. If considered, retain post-level author/license/date attribution, filter and review records, and obtain a specific share-alike and model-artifact analysis. It is not the default first corpus. |

An online source being public, government-hosted, or downloadable is not in itself training permission. Review the current source license and terms, record exact revision/access date, distinguish data copyrights from API contractual terms, identify embedded third-party content, and separately decide redistribution and model-artifact terms. Unknown or conditional training rights remain ineligible under the project dataset policy. The candidate list is not legal advice or owner approval.

## 4. Data quality, provenance, and leakage gates

Before Objective B, define permitted defensive scope and exclude private repositories, credentials, personal data, unauthorized exploit data, and source material whose terms prohibit the intended use. Scan for secrets and PII; assess copyright/attribution obligations; label source revision and transformations; preserve source-specific licenses; and review dangerous operational content.

Deduplicate exact normalized content before split assignment and preserve the original hashes/transform record. Add a conservative near-duplicate pass (for example, shingled similarity) with a versioned threshold and manual audit sample. Group translations, mirrors, versions, same advisories and derivative docs where feasible. Split by source/document family and date, then hash the split manifest. Fit BPE only on training material. Ensure validation/test and benchmark question/answer sources are not used for tokenizer fitting or training.

Research on language-model deduplication reports that repeated training data affects memorization and evaluation behavior; contamination research documents that benchmark text and answers can leak separately. Use these findings to justify both exact and near-duplicate checks, not to claim that any single heuristic proves absence of leakage. See Lee et al., [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499), and [Investigating Data Contamination for Pre-training Language Models](https://arxiv.org/abs/2401.06059).

## 5. Measured compute evidence

Accepted TASK 005-R hosted measurements are for one isolated reference test sample on its recorded implementation SHA and hosted runner. They are not sustained-throughput benchmarks.

| Candidate B=1,T=32 phase | Work units | One hosted sample | Scope |
|---|---:|---:|---|
| Initialization | 20,118,280 | 2.812 ms | Seeded model construction |
| Forward | 66,894,756 | 60.796 ms | Candidate logits and loss |
| Backward | 152,581,015 | 55.503 ms | Complete scoped reverse pass |
| Gradient transfer | 3,658,070 | 0.034 ms | Owned FP32 gradient inventory |
| First persistent accumulation | 914,548 | 0.323 ms | First accumulation into empty slots |
| Combined forward/backward ledger peak | — | 16,037,368 bytes | Existing ledger; candidate optimizer absent |

Run 45584705085 recorded the phase sample on the remediation implementation SHA in TASK 005-R. Accepted TASK 005 main revision and exact-head Foundation evidence are separate: the accepted main is 32eca2b900c4f7a88294f7a24a5dfae0604ae798, Foundation run 45616832319, job 139214284017, successful. The accepted report preserves the candidate measurements and scoped policy.

Each all-supervised B=1,T=32 microbatch carries at most 32 supervised targets; padding or masks can reduce that. Its measured forward plus backward accounting is 219,475,771 project work units. Including the measured transfer is 223,133,841 units before weighted multi-microbatch accumulation, clipping, AdamW, validation, data preparation, and checkpoint I/O. With K microbatches, multiply those per-microbatch terms by K; one optimizer update follows only after the whole weighted accumulation window.

The AdamW update is modeled as roughly 15 scalar arithmetic operations plus one square root per parameter coordinate, or about 13.716 million scalar arithmetic operations and 914,400 square roots for this inventory. Global norm/clipping adds about two scalar operations per gradient coordinate plus one norm square root. These are operation-count estimates, not the project's work units or observed elapsed time.

## 6. Feasibility model, not a training benchmark

The one hosted candidate sample took 116.299 ms for forward plus backward. Dividing 32 target positions by this time gives a component-only sensitivity of about 275 target positions/second. This excludes AdamW, data encoding/streaming, gradient accumulation/clipping, validation, checkpoint writing, process startup, and run-to-run variability. It must not be reported as model-training throughput.

For sensitivity only, 1,000 one-microbatch updates would require at least about 116 seconds if that one pass sample repeated unchanged; 10,000 would require about 1,163 seconds. If unmeasured training overhead multiplied those samples by two or four, the corresponding arithmetic scenarios would be about 4 or 8 minutes for 1,000 updates and about 39 or 77 minutes for 10,000. These multipliers are illustrative, not confidence bounds. K microbatches multiply Transformer pass time approximately by K. A validation pass with N fixed B=1,T=32 windows adds a modeled N * 60.796 ms using the one forward sample; checkpoint I/O is at least the file bytes divided by observed storage bandwidth plus sync/rename time. None of those overheads has been measured.

One optimizer step with K microbatches processes at most 32K supervised targets when every mask is one. S updates therefore cover at most 32KS target positions; token overlap and padding mean unique source tokens are fewer. For example, S=1,000 and K=1 means at most 32,000 targets. This is a scenario only, not an approved budget or selected experiment.

Before selecting a budget, TASK 006-B/C must measure repeated debug/release candidate steps, AdamW update and stage latency, weighted accumulation for each supported K, global clipping, data decode/tokenization, dataset cache and stream buffer peaks, validation at the planned cadence, checkpoint serialization/hash/sync, process high-water memory, and failure/cancellation overhead. Report medians and spread across multiple bounded runs; preserve the exact SHA, runner class, config, sample count, and instrumentation. No current measurement establishes an end-to-end training rate.

## 7. Memory and checkpoint estimates

Parameter values require 914,400 * 4 = 3,657,600 bytes (3.49 MiB). FP32 first and second moments require another 7,315,200 bytes; parameters plus moments are 10,972,800 bytes (10.46 MiB). The token-weighted FP64 accumulator is modeled at 7,315,200 bytes. Accepted complete candidate backward is 16,037,368 ledger bytes; conservative addition of moments, FP64 accumulation, and one overlapping FP32 transfer inventory yields about 34,325,368 bytes before dataset/tokenizer buffers. Optimizer update staging is another 10,972,800 bytes while the retained model, state, and gradients exist, but the training tape is already released at that boundary.

The proposed checkpoint stores model plus both FP32 moments, so its numeric payload is exactly 10,972,800 bytes. Metadata and framing are bounded below 16 MiB total. Writes stream through a fixed buffer. Validation uses the existing reference forward memory model and its own bounded batches; training graph/tape state is released before validation.

These byte sums are logical payload models. The reference ledger charges retained vector capacities, metadata tables, snapshots, adjoints, and transaction overlaps; its 128 MiB ceiling is unchanged. Dataset metadata/token buffers and tokenizer allocations have their own existing limits and are outside the current autograd ledger. Process RSS also includes allocator, runtime, page-cache, and OS overhead. TASK 006-B must measure all three categories and introduce a run-level coordination budget without raising existing caps.

## 8. Execution environments

GitHub Actions remains appropriate for CI, the independent oracle, short correctness proofs, and bounded CPU reference experiments. The current Foundation workflow is explicitly limited to a 15-minute job timeout; GitHub documents a six-hour maximum for a GitHub-hosted job, not a suitable default experiment budget. Heavy training must be manually dispatched, have a strict timeout, concurrency/cancellation controls, small immutable inputs, explicit artifact retention, and an authorized usage budget.

Google Colab is an optional future measured acceleration environment only. The official FAQ states free runtimes are availability- and use-dependent, may time out on inactivity, and run for at most 12 hours; Pro+ continuous execution can run up to 24 hours if sufficient compute units are available. These are changing service limits, not a persistence guarantee. The repository additionally requires explicit owner authorization before provisioning or consuming units. No runtime was started here. Detached CLI survival, checkpoint persistence after disconnect, transfer, and cleanup remain unverified. The native CPU implementation and independent reference remain authoritative even if a future GPU backend is authorized.

Official provider references: [GitHub Actions workflow timeout](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax), [GitHub Actions limits](https://docs.github.com/en/actions/reference/limits), and [Google Colab FAQ](https://research.google.com/colaboratory/faq.html).

## 9. Source links

- [CISA KEV catalog](https://www.cisa.gov/known-exploited-vulnerabilities-catalog) and [official cisagov/kev-data terms and source notes](https://github.com/cisagov/kev-data).
- [17 U.S.C. 105](https://uscode.house.gov/us/usc/t17/s105/) and [NIST data/work copyright and licensing statement](https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications).
- [CVE Terms of Use](https://www.cve.org/legal/termsofuse).
- [OWASP ASVS license](https://owasp.org/projects/asvs) and [Creative Commons BY-SA 4.0 deed](https://creativecommons.org/licenses/by-sa/4.0/deed.en).
- [Stack Exchange content licensing](https://meta.stackexchange.com/help/licensing) and [public network terms](https://meta.stackexchange.com/legal/terms-of-service/public).

Terms and services can change; a future source acquisition must re-check the applicable primary terms before the data is used.
