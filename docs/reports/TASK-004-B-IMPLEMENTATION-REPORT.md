# TASK 004-B Implementation Report

TASK 004-B adds two safe Rust crates for fixed-byte and native byte-level BPE tokenization, approved-source manifests, and local next-token batch generation. The synthetic fixture uses original project-authored technical material. It measures tokenizer behavior and contains no language-model training.

The implementation starts from accepted main `05916e24053b005a67bdd125a695885aab4fd19f`. The design revision is `2eb254688ba2b441c7cd0b6ced99b319e2f6c587`. The four design documents and architecture report are preserved in the planning branch. [Planning PR #8](https://github.com/samuelindra123/xynoos-cyber-model/pull/8), [exact-head planning CI](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45558628062)

## Tokenizer API and byte contract

`Tokenizer::byte` constructs the fixed tokenizer. `BpeTrainer::new`, `push_document` and `finish` provide the algorithm-level byte trainer. `Tokenizer` exposes `encode_bytes`, `encode_text`, `encode_with_specials`, `decode_bytes`, `decode` and `decode_text`. Byte and BPE models share the same `u32` IDs. The API preserves byte values, source-code indentation, mixed newlines, literal marker spellings and Unicode encoding. The byte API round-trips arbitrary bytes and accepts empty input. The text decoder requires valid UTF-8.

| IDs | Meaning |
|---|---|
| 0–255 | Exact byte value |
| 256, 257 | BOS, EOS |
| 258, 259 | PAD, DOC_SEP |
| 260 and above | Learned merges in rank order |

There is no UNK. Document controls are optional BOS at the first position and optional EOS at the last position. `Boundaries` selects their insertion. `SpecialDecodePolicy` explicitly rejects, omits, or renders valid document controls. Rendering is for display and is ambiguous with literal marker text. PAD is reserved for batch suffixes. DOC_SEP is reserved for a future explicit packing grammar. The document decoder rejects PAD, DOC_SEP, misplaced or repeated boundary controls, and out-of-vocabulary IDs.

Training counts every ordered adjacent pair within each document, including overlapping occurrences. It selects the highest frequency of at least two and breaks ties by ascending operand IDs. It assigns the next contiguous ID, replaces left to right without overlap, and recounts. A push failure invalidates the trainer. Encoding scans adjacent pairs using a binary search of the sorted rank lookup, selects the lowest available rank and applies that merge left to right. Replacements compact the same token vector. Decoding preflights the exact byte length and appends flat expansions. For a maximum of `M` rounds and `N` input symbols, the reference trainer performs `O(NM)` scan work with a bounded pair table initialized once. The encoder performs `O(NM log(1+M))` comparisons and `O(NM)` replacement visits in the worst case. The budget also covers pair selection and initialization, document loop visits, expansion copies and rank-table insertion. The measured work counters use a binary ceiling of 67,108,864 units each for training and encoding.

## Tokenizer artifacts and fit binding

`TokenizerArtifact` serializes and loads version-1 `XYNOOS-TOKENIZER` TSV with exact algorithm, vocabulary, registry, ordered merges, lowercase hex expansions, and final LF. Its trailer is SHA-256 over all preceding bytes, including the LF immediately before the trailer. Loading checks the complete checksum and validates the algorithm, registry, ranks, contiguous IDs, earlier-only operands, exact concatenations and byte limits before returning a model. Distinct IDs can share byte expansions. Duplicate pairs, duplicate IDs, forward references, self references and other inconsistent definitions fail. The loader and trainer each allow at most 512 merge records, or 772 total vocabulary entries. Version-1 loaders in this milestone reject larger externally supplied tables. The 65,536-entry format namespace is an independent research ceiling. Candidate 32K and 64K vocabularies require separately reviewed work.

`Dataset::fit_bpe` accepts the validated Train selection, checks each selected payload before fitting and returns a `FittedTokenizer`. The separate canonical fit record binds the exact artifact SHA-256 to the dataset ID, train record count, merge request and SHA-256 of the exact canonical Train rows in manifest order, with LF terminators. `Dataset::verify_fit` validates these claims against a selected dataset and model before producing a bound handle. `Dataset::fitted_batches` requires that handle. `Dataset::batches` uses the fixed byte tokenizer, whose mapping is independent of data. Checksums establish byte integrity and consistent references. Review of authorization, authenticity and externally claimed training history remains a human responsibility.

## Dataset schema, splitting and provenance

`ManifestValidator::open` reads `manifest.json`, `sources.jsonl` and `documents.jsonl`. Version-1 schema types in `schema.rs` define field order. Canonical records are UTF-8 compact JSON objects followed by one LF. Struct deserialization rejects repeated, missing and unknown fields. Re-serialization checks exact field order, escaping, spelling and whitespace. Source records require stable IDs, revision, acquisition reference and UTC date, source checksum, license ID and version, terms reference, explicit approved training, reviewed redistribution status and conditions, responsible maintainer, approved dated review, domain, and transformation history. References are bounded stable identifiers or restricted HTTPS references. Version 1 recognizes an audited BOM removal transformation with version, parameters, process, input hash and output hash. Document transformations must resolve to their approved source history and exact output digest.

Documents declare an ID, source and revision, relative path, exact byte length and digest, UTF-8 encoding, sorted unique domain labels, split, cleaning status, tokenizer compatibility, and duplicate group. Empty payloads, invalid UTF-8, a present UTF-8 BOM and NUL bytes fail content validation. An audited removal records the before/after checksum and stores the post-transformation payload. Typed diagnostics contain only error variants and I/O kinds. Fixture provenance covers original synthetic examples under the project license. A caller-supplied registry remains a claim about rights, and privacy screening remains heuristic.

The sidecar records dataset ID, schema and dataset versions, tokenizer compatibility, split parameters, record count and both exact JSONL checksums. Identity is SHA-256 of `XYNOOS-DATASET-V1` plus NUL followed by the source JSONL length as unsigned big-endian u64 and its exact bytes, then the document JSONL length and exact bytes. The manifest SHA-256 reported below covers the whole sidecar including its final LF.

The duplicate group ID is the lowercase content SHA-256. The splitter hashes ASCII `1`, NUL, seed, NUL, and the group ID. It interprets the first eight digest bytes as big-endian u64 and compares the integer value multiplied by 100 with `80 × 2^64` and `90 × 2^64`. Duplicate checks use the content digest independently of supplied labels. Exact matches across different source IDs remain in the same split. Same-split duplicates remain in manifest order. Source grouping and near-duplicate detection are deferred. Seed `task004-synthetic-v1` is versioned and fixed, and split labels are generated from the algorithm.

## Filesystem and streaming contract

The caller supplies a trusted local dataset root. Paths use normalized ASCII slash-separated components. Absolute paths, empty components, dot and traversal components, drive syntax, reserved device names and symlinks are rejected. The validator checks every path component and verifies that the final component is a regular file, including the metadata files. The reader repeats these checks at use, then verifies exact length, SHA-256 and strict UTF-8 before returning a document or using it for a batch. Integrity for unselected payloads is established when a reader selects them. The root and its ancestor directories must be controlled by the caller, and the tree must remain stable during reads. Separate metadata checks and opens have a filesystem race under hostile concurrent mutation. Hard links and mount configuration are part of the trusted-local-root assumption.

Manifest lines are read with a bounded reusable buffer. Whole-dataset validation retains a separately bounded metadata index and temporarily reuses a sorted index to check IDs, duplicate groups and path consistency. There is no payload prefetcher. A stream processes one current tokenized document and one in-progress batch. Dataset errors are terminal for the reader, and clean EOF returns `None`. Dropping the reader or stream releases its owned buffers. Any consumed partial final metadata record fails.

`BatchConfig` supplies sequence length, stride, batch size, tail policy and optional fixed final batch size. Defaults use stride equal to sequence length, drop the short tail and return the actual final row count. Windows stay within each document. A window of length four uses `[t0,t1,t2,t3]` as input and `[t1,t2,t3,t4]` as target. BOS can start an input window and EOS can supervise the last pair when explicitly configured. A padded tail fills both arrays with PAD for missing pairs and sets validity and loss to zero. An explicitly padded final batch adds whole masked rows. Masks are equal byte arrays and are present when padding is enabled. Tail counts track the remaining shifted positions at the next configured stride. A padded tail is emitted once per document. The crate supplies no attention mask.

## Resource accounting

| Resource | Hard ceiling or accounting rule |
|---|---|
| Payload and tokenizer input | 1 MiB each |
| BPE training inputs | 128 KiB total, at most 4,096 documents |
| BPE merges and work | 512 merges, separate 64 Mi work units for training and encoding |
| Flat expansion and artifact | 4,096 bytes per token, 8 MiB serialized artifact |
| Encoded token output and decoded bytes | 1,048,576 IDs including explicit specials, and 1 MiB decoded bytes |
| Tokenizer owned buffers | 16 MiB covering model tables, expansion capacities, training token vectors and pair tables, or encoding/read/serialization staging as applicable |
| Metadata I/O | 64 KiB reader buffer, 1 MiB serialized record including LF |
| Metadata retention | 100,000 rows per registry/manifest and a shared 64 MiB index budget, including nested field capacities and temporary sort indices. Each metadata file is also bounded by this budget |
| Batch assembly | 16 MiB for owned token, input, target, validity and loss vector capacities |
| Batch dimensions | Sequence length 2,048, batch size 64, product 65,536 positions |

Limits are validated, can be lowered, and are checked before allocation or scans where sizes are known. Project-owned collections use fallible reservation where available and charge actual retained capacities. Pair counts have checked u64 increments. A dense count table and bounded touched-index array make pair selection deterministic and avoid hash-table probe cost. Model ranks use binary search and encoding charges each comparison. Replacement is in place. The stream shares its token budget between the current document and the in-progress batch, and separately limits payload bytes, manifest metadata, and the borrowed tokenizer model. `retained_metadata_bytes`, `retained_bytes` and `peak_token_buffer_bytes` expose measured capacities within their respective boundaries.

The input record cap also bounds JSON decode staging. The selected schema uses typed fields, bounded arrays and the parser's default depth limit. Serde's internal strings, escape scratch, parser state and standard-library path operations are implementation-bounded but are not all measured or fallibly exposed. Allocation bookkeeping, allocator rounding, transient reallocation internals, caller-owned results and buffers, OS page cache and thread stacks are outside the capacity counters. These limits describe application buffers, with no exact process RSS target.

## Locked dependencies and independent verification

Direct dependencies are pinned to `sha2 0.10.9` with defaults off, `serde 1.0.228` with the `alloc` and `derive` features, and `serde_json 1.0.149` with standard-library support. The SHA crate provides reviewed SHA-256, and serde provides schema-based JSON decoding with duplicate-field rejection. Optional assembly packages, arbitrary-precision numbers, map-order features and unbounded JSON depth are disabled. Upstream maintenance resides with RustCrypto and Serde. SHA uses the 0.10 release line and its lockfile-pinned support crates, including the older generic-array 0.14 line. The lockfile also selects current patch releases of the parser's support and derive crates at review time. [sha2 manifest](https://docs.rs/crate/sha2/0.10.9/source/Cargo.toml), [Serde release](https://github.com/serde-rs/serde/releases/tag/v1.0.228), [serde_json manifest](https://docs.rs/crate/serde_json/1.0.149/source/Cargo.toml.orig)

Cargo metadata reports MIT or Apache-2.0 for the three direct crates and for `block-buffer 0.10.4`, `cfg-if 1.0.5`, `cpufeatures 0.2.17`, `crypto-common 0.1.7`, `digest 0.10.7`, `itoa 1.0.19`, `libc 0.2.191`, `proc-macro2 1.0.107`, `quote 1.0.47`, `serde_core 1.0.228`, `serde_derive 1.0.228`, `syn 2.0.119`, `typenum 1.20.1` and `version_check 0.9.5`. `generic-array 0.14.7` and `zmij 1.0.23` report MIT. `memchr 2.8.3` reports Unlicense or MIT. `unicode-ident 1.0.27` reports MIT or Apache-2.0 with Unicode-3.0. `libc` is target conditional. The proc-macro dependencies build serde derives. The 21 exact registry packages in `Cargo.lock` were queried through OSV, with zero returned advisories at review time. This is a point-in-time check. [RustSec advisory interchange](https://rustsec.org/)

Hosted CI explicitly runs `cargo fetch --locked` once before locked offline Cargo metadata, Clippy, debug/release tests and documentation. The compiler pin is Rust 1.100.0. Foundation CI continues to verify exact checkout, hygiene, privacy, numerics, gradients and bitwise learning trajectories.

The standard-library Python oracle independently computes byte IDs, expansions, artifact checksums, dataset identity, split groups, shifted windows, padded batches and masks. The oracle generates the frozen JSON fixture twice and compares the exact bytes. Rust reads the fixture, compares learned artifacts and fit provenance byte for byte and checks every encoded ID, shift, batch and mask. The experiment runs twice on separate synthetic fixture materializations. CI compares the resulting digest across profiles. Negative controls cover wrong byte and special IDs, altered merge order with a valid checksum, invalid artifact checksum, incorrect split metadata, a duplicate across held-out splits, both held-out fit selections, and an unshifted target. The native tests also exercise alias expansions, buffer and work failures, malformed manifests, target alignment, symlinks, and pre-yield payload verification.

## Synthetic corpus results

The fixture contains 16 records with English and Indonesian technical prose, safe Linux examples, C/C++/Rust/Python, LF and CRLF, indentation, repeated whitespace, Unicode, paths, documentation-range addresses, numeric bases and protocol vocabulary. One pair is an exact duplicate and both records land in Test. Measured split counts are 10 Train, 3 Validation and 3 Test. BPE learned 32 merges from the 10 approved Train rows. The byte vocabulary has 260 entries, and the resulting BPE vocabulary has 292. Each document round-trips exactly through both tokenizers. Counts and utilization exclude inserted controls. The byte tokenizer returns one token per byte for every measured record.

| Split | Document | UTF-8 bytes | Byte tokens | BPE tokens | BPE tokens/byte | Distinct byte/BPE IDs |
|---|---|---:|---:|---:|---:|---:|
| train | shell | 81 | 81 | 52 | 0.641975 | 27/33 |
| train | c | 47 | 47 | 39 | 0.829787 | 26/30 |
| train | rust | 60 | 60 | 39 | 0.650000 | 25/26 |
| train | python | 65 | 65 | 38 | 0.584615 | 21/27 |
| train | unicode | 68 | 68 | 54 | 0.794118 | 40/42 |
| train | protocol | 69 | 69 | 49 | 0.710145 | 24/29 |
| train | mode | 60 | 60 | 43 | 0.716667 | 22/26 |
| train | paths | 72 | 72 | 54 | 0.750000 | 26/31 |
| train | hex | 50 | 50 | 39 | 0.780000 | 21/25 |
| train | newline | 59 | 59 | 48 | 0.813559 | 27/31 |
| validation | queue | 77 | 77 | 64 | 0.831169 | 23/28 |
| validation | cpp | 51 | 51 | 43 | 0.843137 | 26/29 |
| validation | address | 61 | 61 | 49 | 0.803279 | 28/28 |
| test | buffer | 69 | 69 | 52 | 0.753623 | 21/26 |
| test | length | 75 | 75 | 60 | 0.800000 | 25/28 |
| test | buffer-copy | 69 | 69 | 52 | 0.753623 | 21/26 |
| **train** | **All 10** | 631 | 631 | 455 | 0.721078 | 79/105 |
| **validation** | **All 3** | 189 | 189 | 156 | 0.825397 | 41/55 |
| **test** | **All 3** | 213 | 213 | 164 | 0.769953 | 27/38 |

Both batch configurations insert BOS and EOS with length 8 and batch size 3. Drop uses stride 8. Pad uses stride 3, pads one remaining tail per document and explicitly pads the final batch. Tail positions exclude fixed-row padding in the Drop column. Pad positions include both partial tails and fixed rows.

| Split | Tokenizer | Drop samples | Drop batches | Dropped tails/positions | Pad samples | Pad batches | Padded tails/positions/rows |
|---|---|---:|---:|---:|---:|---:|---:|
| train | baseline | 76 | 26 | 9/33 | 202 | 68 | 10/31/2 |
| train | bpe | 54 | 18 | 7/33 | 144 | 48 | 10/17/0 |
| validation | baseline | 22 | 8 | 3/16 | 61 | 21 | 3/22/2 |
| validation | bpe | 19 | 7 | 3/7 | 51 | 17 | 3/9/0 |
| test | baseline | 25 | 9 | 3/16 | 68 | 23 | 3/11/1 |
| test | bpe | 19 | 7 | 3/15 | 53 | 18 | 3/15/1 |

| Evidence | SHA-256 |
|---|---|
| Dataset ID | `ee1aa4f39a03a0304bc644cc1f6634fd1f75d987c0167aa7aa32c835dd95cee0` |
| Dataset manifest | `45d2a0d377c1230cbde998f59c813858ae7f40130f1e8a79ee67938017e3ddab` |
| Ordered Train manifest | `0041f6e5ff80d175265e5675da78aebe2db12007c0d4744f9b6e06f50a27e502` |
| Full BPE TSV | `86aa40996d4a9bfe15491131bbe6fb1b04f5e325a50b23a3d7f0fa06bb56011c` |
| Full fixed byte TSV | `20fbe72ba0dfdd39827cb23bcaea82cb943fc3c380cc1cda3fdcbbea0b1c3bad` |
| Frozen JSON fixture | `dc4dd5b6587862bd616b4566bd018fb8dce473b50e61b4216109adb1b34c119d` |
| Complete experiment output | `3cd12b220e38e423e5dab918b30d86708d64c8fa5fb9c8c1c8168d51bfb8b5d1` |

The corpus measures exact behavior for the listed synthetic records. Vocabulary choice, representative sequence efficiency, near-duplicate analysis, source grouping and scaling beyond the reference limits require further research.

## Independent verification handoff

Verify the draft implementation PR's exact head against Foundation CI and the accepted main baseline. Regenerate the Python fixture with the standard-library `--check` path. Inspect the checksummed negative controls and the positive alias fixture. Independently recompute the split thresholds, the Train-only fit hash, the rank conflict behavior and a padded shifted batch. Check the filesystem tests and the trust assumptions around the stable root. Review the workspace diff for unexpected numerical, optimizer, learning-trajectory, data or model changes. Owner acceptance follows Session A's independent review.
