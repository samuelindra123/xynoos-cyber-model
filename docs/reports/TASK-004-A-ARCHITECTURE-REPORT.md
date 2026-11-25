# TASK 004-A — Architecture Report

**Milestone:** Native tokenization, dataset governance and streaming training infrastructure
**Role:** Independent research architect and engineering supervisor
**Scope:** Research, specification and implementation handoff only
**Verdict at this planning head:** Ready for owner review, subject to exact-head hosted CI

## Executive summary

TASK 004-A recommends a small native tokenizer and bounded dataset foundation. TASK 004-B should implement a fixed byte-token baseline and deterministic byte-level BPE together in a focused tokenizer crate. The byte mapping gives an exact, lossless reference for every possible byte; BPE is a candidate for sequence-length measurements, not an already selected production tokenizer. A second small crate should own provenance-checked records, deterministic splits, bounded streaming and next-token batch construction.

The design preserves byte-for-byte content, gives special tokens explicit IDs and insertion semantics, rejects malformed tokenizer artifacts, and makes held-out-data leakage and uncertain training rights hard validation failures. A tiny project-authored corpus is the sole planned experiment; no external corpus acquisition or model training is authorized. The proposed design supports tiny CPU experiments. It does not claim feasibility for 8B or 14B training, which would require separate compute, precision, kernels, optimizer-state, checkpoint and possibly accelerator/distributed milestones.

The planning artifacts are the [tokenizer specification](../design/TASK-004-TOKENIZER-SPEC.md), [dataset and streaming specification](../design/TASK-004-DATASET-AND-STREAMING-SPEC.md), [validation and resource plan](../design/TASK-004-VALIDATION-AND-RESOURCE-PLAN.md), and [Session B handoff](../design/TASK-004-IMPLEMENTATION-HANDOFF.md). The existing [dataset governance policy](../DATASET_POLICY.md) now points to the proposed detailed controls.

## Verified baseline

The canonical checkout was clean on `main` at `05916e24053b005a67bdd125a695885aab4fd19f`; `origin/main` resolved to the same SHA. The required final TASK 003 GitHub Actions run [45553874659](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45553874659) was checked through its job record: the sole repository foundation/numerical job completed successfully. Its steps include exact checkout verification, hygiene/privacy, pinned Rust toolchain, locked offline Cargo, formatting, Clippy, both independent oracles, debug/release tests and doctests, bitwise trajectory comparison, and rustdoc. The [TASK 003 final acceptance report](TASK-003-D-FINAL-ACCEPTANCE-REPORT.md) records the same exact accepted main SHA and 69 passing Rust checks per profile. Accepted evidence also includes 295 numerical oracle cases with 2,670 reference outputs, 25 independently checked gradient coordinates, and deterministic learning trajectory agreement.

Repository instructions, architecture, roadmap, compute policy, dataset policy, evaluation principles, TASK 003 acceptance and handoff, and CI workflow were read. The accepted workspace contains safe Rust dense CPU tensors, first-order autodiff, a bounded tape, revision-aware parameter/gradient ownership, transactional SGD and MSE. It has no accepted tokenizer, Transformer, language-model dataset pipeline/runtime, model checkpoint format, or inference engine. Numerical oracles and the five-sample linear-regression proof establish only their documented foundation behavior, not language-model capability.

## Recommended tokenizer architecture

Implement the byte baseline and byte BPE in one native `xynoos-tokenizer` crate. Implement governance and streaming in `xynoos-data`. This two-crate split gives tokenizer and data APIs distinct validation/error boundaries without adding a general framework. Keep existing numerical/autograd crates untouched.

The byte baseline is the stable reference and fallback. Native BPE is the research candidate because it learns repeated byte sequences while retaining exact byte coverage. A deliberately simple full-recount trainer is preferable initially because its pair counts, tie selection, and replacement behavior are independently understandable. Work caps make it safe for the small experiment. Any optimized trainer must prove exact equivalence first.

Character tokenization cannot independently promise arbitrary-byte round trips and requires a Unicode/invalid-text policy. WordPiece adds presegmentation and unknown/fallback rules not justified by current data. Unigram tokenization adds a probability model and segmentation search not needed for the first deterministic contract. BPE is not presumed more equitable or efficient across English, Indonesian, source code or technical notation: compare measurements before selecting a production vocabulary. Relevant primary research is linked in the [tokenizer design](../design/TASK-004-TOKENIZER-SPEC.md#1-recommendation-and-comparison).

## Byte-level and BPE decisions

`TokenId` is `u32`. IDs `0..=255` map exactly to raw bytes. The supported raw-byte round-trip is exact, including invalid UTF-8. Valid Rust text encodes as its exact UTF-8 byte sequence. Strict text decode errors on invalid UTF-8; byte decode does not. There is no normalization, replacement, case folding, newline conversion, or whitespace cleanup. Empty bytes yield no byte tokens. Supported tokenizer input and output are bounded at 1 MiB and 1,048,576 tokens respectively, with checked arithmetic and fallible allocation.

BPE training starts from byte IDs, counts ordered adjacent pairs only within document boundaries, chooses highest frequency and breaks ties by smallest `(left_id,right_id)`, then performs left-to-right non-overlap replacement. It fully recounts each round, requires candidate frequency at least two, and stops with fewer merges when no candidate remains. Empty corpus is a zero-merge model. Merge IDs are contiguous from 260; merge order defines rank. BPE encoding repeatedly applies the lowest-ranked available adjacent pair, replacing non-overlapping instances left to right. Decoder expands validated flat bytes. Unknown byte input is impossible, because all 256 symbols exist.

Reference limits are 128 KiB total training corpus, 512 merges, 64 Mi symbol visits for training, 64 Mi visits for encoding, and 4,096 bytes per token expansion. Total vocabulary cap is 65,536. These caps make reference work bounded but intentionally do not imply that training a 65k vocabulary is supported in the first implementation. Candidate sizes 1,024 through 65,536 are future comparison points; the initial fixture uses only a small merge budget.

## Vocabulary and special-token design

IDs 256 to 259 are permanently assigned BOS, EOS, PAD and DOC_SEP. There is no UNK because every raw byte has a representation. Baseline size is 260; BPE size is `260 + merge_count`. Specials are inserted by typed explicit APIs, never inferred from marker-looking text. General decoding requires an explicit special-token policy; byte-only decode rejects specials. PAD is reserved for explicit padded batches. EOS/BOS/document separation are inserted only by declared sample boundaries. Conversation templates and instruction tuning are deferred.

Vocabulary is a systems trade-off: one embedding has `V*D` parameters; untied output projection adds another `V*D` and token logits scale with `B*L*V`. At `D=128`, V=4,096 is about 0.524M embedding parameters; V=32,768 is about 4.194M. At hypothetical D=4,096 and D=5,120, V=32,768 implies 134.2M and 167.8M parameters for one table. At V=65,536: 268.4M and 335.5M. An untied output table doubles those counts. These examples show scale, not optimality. A larger vocabulary can shorten sequences but increases embedding/output storage and projection work. The chosen production V requires measurements on a legally reviewed representative corpus, including English, Indonesian, code, Linux documentation and defensive cybersecurity text.

## Tokenizer artifact format

Use a deterministic, inspectable, UTF-8 LF-only TSV format with `XYNOOS-TOKENIZER`, version 1, algorithm ID, vocabulary size, the fixed special registry, ordered merges, lowercase hex byte expansions, and final SHA-256 over all exact prior lines. Limit file size to 8 MiB. Loader validates canonical fields, contiguous IDs/ranks, earlier-only references, exact expansion, duplicate definitions, maximum expansion and checksum before exposing a model. It rejects unsupported versions, noncanonical forms, truncation and trailing bytes. The checksum detects corruption, not authenticity. A reviewed SHA-256 crate is preferable to custom cryptography. The format is not a neural model checkpoint; no large artifact is committed and loading needs no network.

## Technical-text preservation

Byte APIs preserve C/C++/Rust punctuation, Python indentation, shell quoting, paths, tabs, repeated spaces, line endings, hex/binary notation, addresses, protocol names and Unicode exactly. Text APIs preserve valid UTF-8 exactly; arbitrary bytes stay in byte APIs. Tokenizer markers in prose are literal data. Any data cleaning, redaction or normalization is a separate named/versioned transformation with recorded input/output checksums and authorization. Source code whitespace and syntax stay intact unless an explicitly reviewed transformation says otherwise.

## Dataset governance and provenance

No source is training-approved merely because it is public. The source registry captures source ID/reference/revision, acquisition date, checksum, license and terms, explicit permitted training and redistribution use, transform history, domain, accountable maintainer and dated review status. Pending/unknown rights fail training eligibility. Keep external dataset/source code/model-weight/dependency licenses distinct from the repository's Apache-2.0 code license.

Potential future source categories include openly licensed programming documentation, Linux/OS documentation, networking standards, public-domain technical education, appropriately licensed code, project-authored explanations and defensive cybersecurity material. Every category needs source-specific terms review, privacy and quality screening, version pinning, attribution and redistribution review, deduplication, and held-out evaluation-overlap checks. Standards and public online pages must not be presumed unrestricted. No external bulk data is acquired in TASK 004-A or planned as part of the first implementation experiment.

Credentials, API keys, private repositories, personal information, unauthorized proprietary material and restricted/unlawfully obtained data are prohibited. Cyber content must serve lawful defensive, educational or authorized research. Sensitive-content scanners are heuristics, not privacy guarantees. Rejection records may retain safe IDs, checksums, lengths, stage and code, never original content or secret excerpts.

## Dataset record and manifest design

Use deterministic schema-versioned JSONL source and document records with fixed field order and canonical serialization; payloads are separate immutable UTF-8 files addressed by relative normalized paths. A source record records review and rights. A document record has stable ID, source/revision, relative path, exact payload checksum and length, encoding, sorted domains, document split, cleaning/transformation state, tokenizer compatibility, and exact-duplicate group ID. A sidecar fixes dataset/version, split algorithm/version/seed, count, and hashes of exact canonical registry/manifest bytes. Manifest records resolve to a reviewed source registry. Do not embed absolute machine paths or corpus content in Git.

Reject malformed or partial JSON, duplicate IDs/keys, unresolved provenance, unsupported versions, absolute/traversal paths, invalid UTF-8, unapproved BOM removal, oversized records, missing payloads and checksum/length mismatch. Diagnostics never include original text. The detailed format is in the [dataset spec](../design/TASK-004-DATASET-AND-STREAMING-SPEC.md).

## Streaming and batch-generation design

Read a canonical manifest sequence with a bounded reader buffer, parse/validate one record and document at a time, tokenize incrementally and yield deterministic batches. Synchronous reference queue capacity is zero/one. Clean EOF alone returns no next record; truncated final records and I/O/checksum errors are failures. Dropping/early termination releases file and owned buffers. Report application buffer capacities; do not claim exact process RSS control. Configured accounting includes reader, record, document, token, BPE scratch, queue and batch capacities, but excludes allocator, stack and unrelated process memory.

Default `PerDocument` windows never cross document boundaries. For tokens `t0..t4`, length four yields input `[t0,t1,t2,t3]`, target `[t1,t2,t3,t4]`; IDs remain integers and dimensions are `[L]` or `[B,L]`. Stride, boundary insertion, dropped tails and partial batches are explicit. Default stride is L and final partial batch is returned at actual size. Optional fixed-size padding uses PAD and explicit valid/loss masks. No automatic packing of unrelated documents; any future packing must insert an explicit separator and define masks.

## Dataset splitting and leakage prevention

Group exact duplicates before split. Assign document groups deterministically with SHA-256 over split version, seed and stable group ID; v1 integer cutoffs assign 80/10/10. Record seed/version and split per record. A named source-grouped policy is available where source revisions risk leakage. Reordering input does not affect assignment. Before acceptance, scan exact checksums across splits and validate duplicate groups/source grouping. Fit tokenizer only from authorized train records; save fit provenance as dataset/split manifest checksum. Exact dedup misses near-duplicates, translations and modified content; state this limitation and reserve near-duplicate research for later.

## Resource limits

The reference ceilings are: 1 MiB per document and tokenizer input; 128 KiB BPE corpus; 512 merges; 64 Mi training and encoding visits; 65,536 vocabulary IDs; 4,096-byte token expansion; 8 MiB artifact; 1,048,576 encoded tokens; 64 KiB reader buffer; 1 MiB manifest record; 100k entries for reference manifest validation; 16 MiB retained token/input/target buffers; sequence length 2,048; batch size 64; and 65,536 `B*L` positions. Lower caller limits are allowed. Checked arithmetic and fallible allocations are mandatory. The first reference's BPE recount and in-memory manifest metadata do not scale efficiently beyond these bounds; future deterministic incremental merging and manifest sharding need separate milestones.

## Independent verification strategy

TASK 004-B should add small standard-library Python references independent of Rust for byte mapping, BPE pair counting/encoding/decoding, split assignment and next-token shifting. Rust validates their immutable fixtures. Include all 256 bytes, empty/invalid UTF-8, specials, limits, overlap/tie/document boundary BPE cases, malformed artifacts, deterministic checksums, provenance/schema errors, duplicate and split leakage, streaming EOF/partial/error handling, bounds and batch padding/shift. Negative controls mutate token IDs, merge order, checksum, split assignment, train/test duplicate placement and target shift. No production code is included in this planning phase.

The project-authored synthetic fixture includes English and Indonesian technical prose, Linux commands, safe small source-code samples, indentation and repeated whitespace, Unicode, paths, addresses and technical numeric forms. Measure exact UTF-8 bytes, baseline/BPE tokens, tokens/byte, vocabulary utilization, byte round-trip, deterministic artifact checksum, split integrity and next-token sample count. Repeat and compare exact outputs. These measurements do not generalize to production corpus efficiency. No model is trained.

## Rust crates and dependency boundaries

Add `crates/xynoos-tokenizer` and `crates/xynoos-data`. Tokenizer owns token IDs, specials, encoders/decoders, BPE trainer/model, artifact and `TokenizerError`. Data owns source/document schema, manifest validator, split assignment, bounded reader, stream/batch configuration, `Batch` and `DatasetError`. Standard library handles deterministic loops, buffers and local file IO. SHA-256 requires a reviewed `sha2`-family crate; JSON parsing/canonical serde may use pinned `serde`/`serde_json` only if exact versions, licenses, advisories and transitive dependencies pass review and offline `Cargo.lock` CI. No ML frameworks, pretrained tokenizer, async runtime, database or network runtime.

## Future Transformer compatibility and 8B/14B implications

The vocabulary size defines embedding row count and output class count; each `u32` ID indexes the same row in input embedding and is a categorical next-token target. Batches provide exact `[B,L]` shifted IDs and optional loss validity masks, with document boundaries controlled by the data policy. TASK 005 can add context length and causal attention behavior while preserving these IDs, boundary semantics, artifact identity and data ordering. Tokenizer artifact hash and vocabulary size must be part of any later model configuration/checkpoint compatibility key.

Tiny experiments are supported by byte vocab and small BPE candidate. A larger V changes sequence length and parameter/logit cost; representative evaluation is required. An 8B or 14B architecture remains conditional ambition with no compute commitment. The current 128 MiB autograd accounting, binary32 CPU kernels, and initial reference data caps do not support billion-parameter training. Mixed precision, optimized/GPU kernels, a larger explicit memory policy, optimizer state, checkpointing, inference and distributed execution are separate future design and validation milestones, not features delivered by TASK 004.

## Risks, unresolved decisions, and Session B handoff

Owner review must decide whether these strict caps and the v1 split fractions are suitable for implementation; they are conservative engineering defaults, not empirical corpus findings. Source-specific legal terms and privacy review remain human decisions. Vocabulary/merge targets, representative corpus access, BPE efficiency by language/domain, near-duplicate policy and scalable manifest sharding remain unresolved. Dependency versions require implementation-time security/license/offline review.

Session B should implement exactly the linked specifications, first freeze independent fixtures and Python algorithms, then add crates, then test a tiny synthetic dataset. It must preserve every accepted TASK 002/003 check and obtain exact-head bounded CI. The complete handoff and acceptance list are in [TASK-004-IMPLEMENTATION-HANDOFF.md](../design/TASK-004-IMPLEMENTATION-HANDOFF.md).

**Explicit non-goals:** Transformer, attention, embeddings, language-model training, Adam, cross-entropy, checkpoints, inference, GPU/distributed support, external corpus acquisition, large datasets, production tokenizer/vocabulary claims, and treating 8B/14B as delivery commitments.
