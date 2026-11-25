# TASK 004 — Session B Implementation Handoff

**Recipient:** Session B, after owner review of the planning PR and explicit implementation authorization.
**State:** Planning only. This handoff does not authorize production implementation, data acquisition, training, or model work.

Read this document with the [tokenizer contract](TASK-004-TOKENIZER-SPEC.md), [dataset and streaming contract](TASK-004-DATASET-AND-STREAMING-SPEC.md), and [validation/resource plan](TASK-004-VALIDATION-AND-RESOURCE-PLAN.md). Those specifications control exact behavior; do not fork the contracts while coding.

## Selected architecture

Create two focused native crates: `xynoos-tokenizer` for fixed bytes, explicit specials, BPE training/encoding, and the versioned tokenizer artifact; `xynoos-data` for source/document manifests, deterministic split validation, bounded readers, and shifted next-token batches. Keep `xynoos-tensor` and `xynoos-autograd` unchanged. The split isolates reusable token semantics from governed dataset IO and gives each crate a focused error and test surface. Avoid adding crates beyond these unless concrete needs and licensing are reviewed.

Implement both the fixed 256-byte reference and a native deterministic BPE candidate in TASK 004-B. Fixed mapping is the correctness baseline and BPE is primary candidate for measurement, not a production recommendation. BPE uses whole-round deterministic pair recounting and bounded left-to-right replacement. Do not optimize until the independent Python reference agrees. The BPE trainer can accept a deterministic stream of authorized training documents without retaining a corpus concatenation.

Dataset reading is synchronous and local-file based. Use canonical deterministic manifest order, one bounded record/document at a time and queue capacity zero/one. Exact duplicates are grouped before deterministic split assignment. The default sample builder windows each document separately, performs one-position input/target shifting, and does not pack across document boundaries.

## Rejected alternatives

- Character tokenization alone is not a byte-safe representation for arbitrary binary or malformed UTF-8.
- WordPiece adds text presegmentation, unknown/fallback behavior and trie policy without evidence for this project.
- Unigram tokenization introduces probabilistic segmentations and a more involved trainer/decoder than needed for a deterministic first pipeline.
- A third-party pretrained/tokenizer runtime, copied implementation, external framework, or external tokenizer as native production behavior is prohibited.
- A clever heap/incremental BPE trainer, parallel data loader, heavyweight database, distributed filesystem, data lake, cloud service, or general workflow engine is not required for bounded reference work.
- A custom hash or cryptographic checksum implementation is rejected; use a reviewed SHA-256 crate. A custom JSON parser is unnecessary; use a small reviewed parser under locked offline CI.
- No standalone checkpoint crate, instruction format, conversation template, preprocessing/normalization framework, or generalized ETL system.

## Exact contracts to implement

### Tokenizer

- `TokenId = u32`; byte token ID equals byte value `0..=255`.
- Stable specials: BOS 256, EOS 257, PAD 258, DOC_SEP 259; no UNK.
- Baseline vocab 260; BPE token IDs start at 260 and follow merge order contiguously.
- Raw bytes round-trip exactly for all supported byte inputs. Valid UTF-8 string convenience APIs pass exact UTF-8 bytes. Strict text decoding rejects invalid UTF-8; byte decoding preserves it. Nothing normalizes or replaces.
- Empty input produces no byte tokens. BOS/EOS insertion is explicit. Marker-looking literal text has no special meaning. General decode requires an explicit special policy; byte-only decode rejects specials by default.
- BPE details, tie-break, recount, merge order, overlapping replacement, encoding rank procedure, no-cross-document training, flat decoded expansion, and artifact invariants are exactly as stated in tokenizer spec.
- `xynoos-tokenizer` exports opaque checked `TokenId` semantics, `SpecialToken`, `Tokenizer`, `ByteTokenizer`, `BpeModel`, `BpeTrainer`, `TokenizerArtifact`, `TokenizerLimits`, and typed `TokenizerError`. Do not expose mutable invalid vocab state.

### Tokenizer artifact

Use versioned canonical LF-only UTF-8 TSV `XYNOOS-TOKENIZER\t1`, algorithm ID, vocab size, fixed registry, ordered merge rows with lower hex expansions, and trailing SHA-256 over exact preceding bytes. Maximum 8 MiB. Strictly reject malformed/noncanonical fields, duplicate or forward references, altered IDs/ranks/expansions, cycles, checksum errors, trailing bytes and unsupported versions. No executable payload, network lookup, model checkpoints, or committed large artifact. Hash is integrity detection, not authenticity.

### Dataset governance and files

Canonical version-1 JSONL source registry and document manifest plus relative-path immutable UTF-8 payload files and a deterministic manifest checksum sidecar. Field order, schema, exact checksum bytes, enums, path rules, checksum rules and provenance requirements are in the dataset spec. Training inputs resolve only to approved source records with explicit training permissions. Keep source, dataset, model, dependency, and code licensing separate. Project synthetic fixtures only for this milestone; no external large corpus acquisition.

Reject malformed and partial records, missing/unapproved rights, duplicate IDs, duplicate exact content across splits, invalid encoding, oversized entries and content/hash mismatch. Safe diagnostics contain codes/IDs/lengths/checksums only and never payload text. Privacy scanners are heuristics, not guarantees. Cleaning is explicit, versioned and checksum-audited; preserve source-code whitespace.

### Split and streaming

- Document/group level split, SHA-256 stable assignment, fixed v1 integer thresholds (80/10/10), exact duplicate grouping first, optional named source-grouped algorithm, manifest version/seed/count/checksums.
- Fit byte/BPE tokenizer only on approved train split. Record tokenizer fit's dataset/split checksum.
- Stream bounded manifest records/documents in deterministic order. Synchronous local reader, explicit bounded owned buffers, clean EOF only as `None`, partial final line/error as failure, cancellation via iterator drop. No corpus-sized payload/token allocation.
- Default `PerDocument` batching; no implicit cross-document packing. `input[j]=tokens[k+j]`, `target[j]=tokens[k+j+1]`, both integer IDs and equal shape. Sequence length, batch, stride, optional pad/loss mask and final partial-batch semantics as in spec. Reject bad IDs/configurations and arithmetic overflow.
- Public `xynoos-data` concepts: `SourceRecord`, `DocumentRecord`, `DatasetManifest`, `ManifestValidator`, `DatasetReader`, `StreamConfig`, `BatchConfig`, `Batch`, `DatasetLimits`, and typed `DatasetError`. `Batch` carries input/target `Vec<TokenId>` with shape/config metadata and only explicit masks. Iterator follows `Result<Option<Batch>, DatasetError>` semantics.

## Resource limits

Treat values in the [validation plan](TASK-004-VALIDATION-AND-RESOURCE-PLAN.md) as hard ceilings: 1 MiB per document/tokenizer input; 128 KiB BPE corpus; 512 merges; 64 Mi BPE train and encode work visits each; total vocab 65,536; token expansion 4,096 bytes; tokenizer artifact 8 MiB; output token count 1,048,576; reader buffer 64 KiB; record 1 MiB; max 100k entries in reference manifest validation; 16 MiB retained token/input/target buffers; sequence 2,048; batch 64; `B*L` 65,536. Caller may lower ceilings. Preflight `checked_add`/`checked_mul`, then use fallible allocation and charge actual owned capacities at boundaries. No RSS guarantee.

Use typed errors for invalid ID/special policy, malformed artifact, checksum mismatch, unsupported version, invalid UTF-8 in text APIs, invalid JSON/schema/provenance/path, duplicate IDs/content, missing payload, invalid split/tokenizer fit, oversized record/input/output, work/vocabulary/expansion/batch limits, I/O, arithmetic overflow, and allocation failure. Error variants should be actionable but must not carry sensitive payload excerpts.

## Dependency and crate policy

Prefer standard library for ordered deterministic algorithms, filesystem streaming, and buffers. A SHA-256 dependency is required rather than a locally written cryptographic primitive. If JSONL uses `serde`/`serde_json`, pin exact compatible versions in `Cargo.lock`, review their licenses/advisories and transitive graph, and confirm cached artifacts support `--locked --offline` CI. The canonical writer must have stable struct field order; parser must reject duplicate fields (do not silently keep first/last). Keep normal CI free of data downloads. Do not add an ML library, tokenizer package, async runtime, database, or framework.

## Independent validation and experiment

Build a Python standard-library oracle independently for byte IDs, BPE counts/ties/merges/encoding/decoding, split assignment, and sample shifting. The oracle does not call Rust to create expected outputs. Native tests cover every byte, invalid UTF-8, specials, artifact corruption, merge overlaps and ties, cross-document behavior, checked limits, provenance/manifest cases, stable splits, contamination, bounded streaming, batches and errors. Include negative controls for wrong token ID, altered merge order, bad checksum, altered split, duplicate leakage, and off-by-one target shift. All fixtures are tiny.

Run the project-authored synthetic corpus comparison exactly as scoped in the validation plan: byte counts, byte/BPE token counts, tokens/byte, utilization, round trips, stable artifact hash, split integrity, and next-token sample count. Run twice and compare exact outputs. It provides no production corpus efficiency claim and trains no language model.

## CI preservation and acceptance criteria

Preserve accepted TASK 002/003 checks without changing their fixture semantics: pinned Rust 1.100.0, locked offline metadata, formatting, Clippy warnings denied, debug/release tests/doctests, docs, 295-case numerical oracle, gradient oracle and negative controls, deterministic learning experiment and bitwise debug/release trajectory, repository privacy/artifact check, exact-head verification, read-only permissions, two build jobs, bounded 15-minute CI, and PR triggers for `main` and `feat/**`. Add the new small tokenizer/data oracles and tests after the workspace dependencies and fixtures are reviewed. No GPU, paid runner, large downloads, large dataset, weights, or model training.

Acceptance requires all of the following:

1. Byte IDs round-trip every byte exactly; strict text errors remain distinct from byte decoding.
2. Specials are explicit and marker-like literals remain bytes.
3. BPE trainer, encoder, decoder and artifact match independent Python and hand-calculated fixtures including ties, overlaps, boundaries and malformed models.
4. All resource/work/overflow limits are enforced before unbounded allocation/work, with typed errors and no partial accepted state.
5. Provenance/license review, source registry, manifest, checksums, transformation log and privacy-safe rejection behavior are enforced.
6. Stable document/split order and leakage checks pass; tokenizer fit excludes held-out splits.
7. Streaming buffers remain within configured application accounting; deterministic EOF/errors/cancellation and shifted batches are validated.
8. Synthetic corpus comparison is exact, reproducible and accurately described.
9. All existing TASK 002 and TASK 003 regression checks remain unchanged and pass on the exact final PR head and after merge.
10. No external training data is acquired, no model is trained, and no personal/hardware/credential material is committed.
11. Owner reviews accepted planning PR and separately authorizes TASK 004-B implementation before Session B edits.

## Risks and unresolved decisions

- The reference BPE and JSONL canonicalization rules are intentionally strict. Independent Python and Rust fixtures must freeze edge semantics before larger artifacts exist.
- Corpus/split limits constrain initial scope. Sharded manifests, scalable duplicate indices, near-duplicate detection, and incremental BPE are future work, each requiring separate design and validation.
- Dataset rights interpretation and privacy screening need human accountable review. Neither manifest completeness nor automated screening establishes legal permission or privacy safety.
- Production vocabulary size, merge count, tokenizer sequence efficiency by language/domain, model-context design, and tied versus untied output embeddings remain unselected until representative lawful data and TASK 005 design exist.
- The accepted autograd 128 MiB application contract does not support an 8B or 14B training claim. Mixed precision, optimized kernels, GPU backends, larger-memory compute policies, optimizer states, distributed execution, checkpointing and inference are separate future milestones.
- The repository presently has a small dependency set; additions can impact offline builds and supply-chain risk. Any chosen exact versions and their licenses must be reviewed in the implementation PR.

## Explicit non-goals

No Transformer layers, attention, embeddings, language-model training, cross-entropy, Adam, model checkpoints, inference, GPU support, distributed training, external corpus acquisition, corpus scraping, production dataset, conversation/instruction templates, normalization, advanced near-deduplication, database, cloud data platform, custom cryptography, unsafe Rust, or claims that future 8B/14B ambitions are committed deliverables.
