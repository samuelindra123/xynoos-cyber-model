# TASK 004 — Validation and Resource Plan

**State:** TASK 004-A plan for independent bounded verification; no test or production implementation is added here.

## 1. Resource ceilings for the reference implementation

All values are project ceilings. API callers may choose lower limits; no caller can raise these ceilings. Every multiplication, addition, conversion, count and capacity request uses checked arithmetic before allocation or work. An error leaves no partial accepted artifact/batch. Limits cover application-owned buffers only and do not promise process RSS or prevent allocator/OS overhead.

| Resource | Maximum | Enforcement / accounting boundary |
|---|---:|---|
| One UTF-8 document payload | 1 MiB | Check manifest length before open/read and actual bytes while streaming. |
| One tokenizer input | 1 MiB | Check before encoding; arbitrary raw bytes are allowed by byte APIs. |
| Total BPE training corpus | 128 KiB | Sum exact train document bytes with checked `u64`; never materialize corpus concatenation. |
| BPE merge count | 512 | Stop with a valid smaller model only when no pair occurs at least twice; reject configured/serialized excess. |
| BPE training work | 64 Mi symbol visits | Count each pair-count and replacement scan visit; checked and preflighted. |
| BPE encoding work | 64 Mi visits | Count comparisons/scans; return `WorkLimitExceeded` if the next step exceeds budget. |
| Vocabulary size | 65,536 total IDs | Supports candidate research cap, including specials; reject beyond. |
| Expansion of one BPE token | 4,096 bytes | Validate on training and artifact load; decode by prevalidated flat expansion. |
| Tokenizer artifact | 8 MiB | Bounded read before parse; no partial model escapes failure. |
| Output tokens per encoding | 1,048,576 | Check before reserve and every append; specials count too. |
| Reader buffer | 64 KiB | One reusable fixed buffer in reference reader. |
| One serialized manifest record | 1 MiB | Incremental line cap; reject rather than allocate unbounded. |
| Dataset manifest entries | 100,000 | Bounded reference manifest/split validation; future sharding requires explicit format/version. |
| Retained token/input/target buffers per iterator | 16 MiB | Sum actual `Vec::capacity()*size_of::<T>()` for owned token arrays and scratch that can coexist. |
| Sequence length `L` | 2,048 tokens | `L>0`; reject above ceiling. |
| Batch size `B` | 64 samples | `B>0`; actual partial final batch may be smaller. |
| Batch token positions `B*L` | 65,536 | Checked multiplication before allocating. Input and target each use u32; masks, if any, are charged separately. |

Reference streaming memory accounting includes the reader buffer, one record line, one document byte buffer, one encoded document token buffer, BPE scratch, one assembled batch, and queued owned records (queue capacity zero/one in synchronous reference). For batch construction, at least `2 * B * L * sizeof(u32)` bytes are counted for inputs and targets. Any pad/loss mask is included. The manifest validator's 100,000 entries are metadata owned for whole-dataset split checks; it is a separate bounded validation phase, not silently counted as zero. Large future manifests need deterministic shards or streaming external merge as a separately reviewed extension.

The simple BPE trainer's time scales with corpus symbols times number of merge rounds because pairs are recounted. A low corpus cap and explicit measured-work budget make tiny fixtures tractable but cap efficiency. It is a correctness/reference design, not an efficient large-corpus trainer. Future scaling may replace recounting with deterministic incremental pair maintenance after independent equivalence tests. Streaming payload files avoids corpus RAM growth; full split validation still scales with manifest metadata and requires sharding beyond 100k documents. CPU inference and model output projection costs are not solved by the tokenizer.

## 2. Byte-tokenizer tests

The native suite must cover:

- Each of all 256 single byte values and the complete `0..=255` sequence.
- Empty input, repeated bytes, all byte values in arbitrary order, and deterministic generated sequences within small bounds.
- Valid UTF-8 including Indonesian diacritics and multi-byte Unicode; invalid UTF-8, embedded NUL, newline, tabs and CRLF through byte and text APIs.
- Exact byte round trips, distinct strict UTF-8 decode behavior, and zero input mutation/normalization.
- Literal control-marker spellings encoded as ordinary bytes; typed special insertion only; all decode policies; invalid placements and PAD use.
- Unknown/out-of-range IDs, baseline/BPE vocabulary mismatches, malformed special sequences, and output-length limits.
- Synthetic tiny limits to prove input/output/retained-buffer exhaustion before large allocation; checked-overflow helpers at boundary values.

Every failure path must assert a typed error and no partial output is reported as complete. Negative controls alter one expected byte-token ID and one special ID and require the oracle to reject them.

## 3. BPE independent validation

Implement a Python standard-library reference in TASK 004-B that independently expresses pair counts, tie rule, non-overlap replacement, ranks, rank-first encoding, expansion, and decode. It must not invoke the Rust library to construct expected results. Fixtures are hand-calculated and small enough for full enumeration.

Required cases: known pair counts; frequency tie resolved by lexicographically smallest ID pair; repeated symbols and overlap (`a a a`); multiple documents with no cross-boundary merge; zero-byte and empty corpus; insufficient candidates; several merge rounds and encoding rank conflicts; all-byte round trips; token expansion; train determinism after identical ordered input; explicit corpus order semantics; repeated serialize/load checksum; vocabulary utilization and counters; merge/vocabulary/work/expansion limits; counter overflow tested via isolated checked-counter helpers.

Malformed artifacts: altered checksum, duplicate merge pair, wrong ID or rank, special as operand, forward reference, cyclic/self reference, expansion mismatch, duplicate token bytes, oversized expansion, missing merge, extra trailing field, unknown version/field, noncanonical integer/hex, CRLF, truncated final line, oversized artifact and unsupported algorithm. Negative controls reorder valid merge definitions, alter a token ID and alter a checksum; all must be rejected or produce a reference-detected disagreement. SHA-256 fixture values should be independently checked by Python `hashlib`.

## 4. Dataset and streaming validation

Use only tiny project-authored fixture records. Test canonical serialization and reproducibility under repeat generation. Validate approved provenance and reject missing source, pending/unknown rights, invalid license enums, missing terms reference, absent maintainer/review, unauthorized redistribution claim, personal absolute path, path traversal, noncanonical order, duplicate keys/IDs, unsupported version, invalid UTF-8/BOM, byte-length mismatch, content checksum mismatch, empty/oversized/partial records, malformed JSON, and unknown transformation/domain values. Diagnostics must not contain source text; test this using a sentinel secret in rejected fixture content.

Split tests: independently calculate the fixed SHA-256 assignment; demonstrate stable assignment under record order changes; validate exact duplicates remain in one split; enforce optional source grouping; verify manifest checksums; deliberately inject a duplicate checksum across train/test and require rejection. Tokenizer fit tests pass only train metadata and reject validation/test records. Negative controls inject held-out content into training fit, tamper with split version/seed, and alter manifest checksum.

Streaming tests use a tiny reader buffer and tiny configured caps. Test deterministic record order independent of directory enumeration, bounded live-record count, bounded token capacity, clean EOF, empty dataset, truncated final line, mid-record I/O failure where practical, cancellation/drop cleanup, batch order, repeat checksum, and error propagation without leaking content. Keep all fixtures local and small.

Window/batch tests cover exact shift (`[a,b,c,d] -> ([a,b,c],[b,c,d])`); integer ID validation; no cross-document window in default mode; configured EOS/BOS/DOC_SEP; stride 1 and stride L; dropped tail counts; partial final batch; optional PAD behavior, boolean/byte valid and loss masks, PAD excluded from loss, and invalid zero/oversized sequence, batch, stride, product overflow, and token count. A faulty offset negative control must be detected by the independent reference. Do not test or implement attention masks here.

## 5. Tiny reproducible tokenizer experiment

TASK 004-B should author and review a tiny fixed corpus with at least one document per split (and enough train documents to train BPE): English technical explanation, Indonesian technical explanation, Linux commands, safe minimal C/C++/Rust/Python examples, indentation/tabs/repeated spaces, newline variants, filesystem paths, IPv4/IPv6 or CIDR, hexadecimal and binary notation, protocol names, and Unicode. All prose/examples are authored for this fixture; do not copy external documentation. Clearly mark that it is a tokenizer fixture, not broad training data.

Use a recorded seed and fixed document IDs. Split assignment must be generated by the documented algorithm; if the tiny set produces an inconvenient split, version the fixture documents or seed explicitly before freezing; do not hand-edit split labels after inspecting content. Fit BPE on training split only, with a small fixed merge count. For every document report exact UTF-8 byte count, baseline byte token count, BPE token count, tokens/byte, token IDs or utilization summary, and exact byte round-trip. Report tokenizer artifact SHA-256, train/validation/test counts and integrity results, and generated next-token sample count under documented `L`, `B`, stride and boundary choices. Run twice and compare exact artifacts, manifests, counts and batches. Report measured values only; do not claim production compression or language-model quality. No language model is trained.

## 6. Independent oracle and CI plan

Add bounded standard-library Python reference scripts/fixtures for selected byte mapping, BPE and split/batch algorithms. Python generates independent expected files; Rust consumer strictly validates schema, counts, checksums, token ranges, outputs and tolerances as relevant. Keep fixture count/size bounded and reviewed. Preserve unchanged TASK 002 numerical fixture and TASK 003 gradient/learning fixtures.

Preserve existing `.github/workflows/ci.yml` guarantees: Rust 1.100.0, `cargo metadata --locked --offline`, formatting, Clippy with `-D warnings`, debug/release tests and doctests, docs, original numerical and gradient oracles, bitwise trajectory agreement, privacy/artifact checks, exact-head checkout, read-only permissions, bounded 15-minute job, two Cargo build jobs, pinned actions, and PR triggers for `main` and `feat/**`. Add tokenizer/data oracle generation and consumer tests to the same small workflow only after fixture scope is measured. No external dataset download, paid runner, GPU, model weights, or large token corpus in ordinary CI. Any additional dependency must be locked and cached or available under the repository's offline build constraints; offline CI must fail cleanly if dependencies are not cached and must not silently fetch.

TASK 004-A planning CI itself should run existing workflow on the PR exact head. Since only docs change, no production suites are altered. Check Markdown, repository privacy/artifact rules, exact-head checkout, and all existing regression checks in hosted CI. Do not use local full Cargo builds for this documentation task.
