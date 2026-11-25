# TASK 004 — Tokenizer Specification

**State:** TASK 004-A planning contract; no tokenizer implementation is included.

## 1. Recommendation and comparison

Use two native representations in TASK 004-B: (1) a fixed byte tokenizer as the correctness oracle and permanent fallback, and (2) deterministic byte-level BPE as an experimental tokenizer. They share the same byte alphabet and special-token registry. Implement both in one small crate. The baseline is nearly trivial and makes correctness, binary behavior, and BPE decoding independently testable; putting BPE behind a later milestone would defer the central measurement this milestone is intended to produce. BPE training remains bounded and is not required to produce a useful production vocabulary.

| Method | Coverage and decoding | Sequence length / CPU and memory | Assessment for this project |
|---|---|---|---|
| Character | Needs a character definition; Unicode scalar values preserve valid text but cannot represent arbitrary bytes without fallback. Invalid UTF-8 needs a separate policy. | Often much shorter than bytes; mapping and normalization choices complicate code and multilingual coverage. | Useful teaching baseline, but not the byte-safe reference. |
| Fixed bytes | Every byte is represented exactly; arbitrary binary and malformed UTF-8 round-trip. | Up to one token per byte; fixed 256-entry mapping, tiny memory, linear encoding/decoding. | Required reference baseline and guaranteed fallback. |
| Byte BPE | Starts from every byte and merges byte strings; remains lossless for arbitrary bytes. | Usually fewer tokens on repeated patterns, with learned vocabulary and training cost. | Preferred candidate primary tokenizer, subject to representative measurements and strict work bounds. |
| WordPiece | Usually begins with pre-tokenized text and uses a learned vocabulary; unknown behavior depends on policy. | Greedy longest-match can be efficient with a trie; word boundaries and normalization affect code/text. | Rejected for first milestone: extra segmentation and unknown-token policy without a demonstrated advantage. |
| Unigram | Probabilistic vocabulary/segmentation; exact input reconstruction depends on a specified byte/character fallback and detokenizer. | More complex training, model probabilities, and segmentation than deterministic BPE. | Rejected for first milestone; a later measured comparison could revisit it. |

The project will not call Hugging Face Tokenizers, SentencePiece, tiktoken, or a copied implementation for production. Such tools may be used only as documented optional independent comparisons. BPE and subword tokenization are established research choices, not proof that a particular vocabulary is best for XYNOOS; see Sennrich et al. on open vocabulary subwords, Kudo and Richardson on language-independent SentencePiece, Kudo on unigram segmentation, and Radford et al. for byte-level BPE use ([ACL paper](https://aclanthology.org/P16-1162/), [SentencePiece paper](https://aclanthology.org/D18-2012/), [Subword Regularization](https://aclanthology.org/P18-1007/), [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)).

## 2. Exact ID and byte contract

Use `TokenId = u32`. IDs are model inputs and categorical targets; they are never represented as floating point. The stable namespaces are:

| IDs | Meaning |
|---|---|
| `0..=255` | One raw byte, with ID exactly equal to its unsigned byte value. |
| `256` | `<|bos|>` |
| `257` | `<|eos|>` |
| `258` | `<|pad|>` |
| `259` | `<|doc_sep|>` |
| `260..` | BPE merge tokens, in learned merge order; absent from the fixed-byte model. |

There is no UNK token. All byte values are representable, so UNK would discard information and has no required role. A baseline model vocabulary size is 260; a BPE model has `260 + merge_count` entries. IDs are contiguous, unique, and less than `vocab_size`; all arithmetic when assigning them is checked. New special tokens require a tokenizer format/version and compatibility decision. Existing IDs are never repurposed. A larger future registry is an explicit incompatible artifact revision.

Raw byte encoding maps each input byte `b` to token `b` in the baseline. `decode_bytes` maps byte tokens to themselves and BPE tokens to their recursively validated byte expansion, concatenated in order. For every byte slice of at most the configured input limit, the byte baseline guarantees `decode_bytes(encode_bytes(x)) == x`, including empty input and invalid UTF-8. BPE has the same guarantee for every supported input.

Text convenience encoding first takes the exact UTF-8 bytes of a Rust `str`; it performs no normalization, case folding, newline conversion, or whitespace processing. `decode_text` is separate: it first obtains bytes under a caller-selected special-token policy and then uses strict UTF-8 validation. Invalid byte output returns `InvalidUtf8`; no replacement character is inserted. `decode_bytes` remains available for arbitrary data.

Literal text such as `<|eos|>` is ordinary UTF-8 bytes. Only typed calls such as `encode_with_specials` / `push_special(SpecialToken::Eos)` can add control IDs. The plain text encoder never recognizes marker spellings. `decode_bytes` rejects control IDs rather than guessing their representation. An explicit `SpecialDecodePolicy` must be supplied to the general stream decoder: `Reject` (default and required for training text), `Omit`, or `RenderRegisteredSpelling`; the last is display-only and does not promise round-trip of original literal spellings. PAD is allowed only in padded batches, BOS/EOS/DOC_SEP only at explicitly chosen boundaries, and none may occur inside a raw-byte segment.

Empty bytes encode to zero tokens. Adding BOS/EOS is explicit; an explicitly encoded empty document may therefore contain just those boundary tokens. Invalid ID, a special where bytes are required, an ID outside the model vocabulary, or a malformed special sequence returns a typed error. Decoder APIs do not silently skip an unknown ID.

The 004-B defaults are: at most 1 MiB per encoder input, at most 1 MiB output tokens (with checked exact preflight; bytes are a tighter upper bound absent explicit specials), and no allocation above that output bound. Callers can lower but not raise these project ceilings. All capacity growth uses `try_reserve`/`try_reserve_exact`, checked length arithmetic, and typed allocation/resource errors. A configurable `ResourceLimits` object is passed rather than hidden global state. The limit bounds application-owned input/output buffers, not allocator metadata, stack, other allocations, or process RSS.

## 3. Native byte BPE training and encoding

### Training algorithm

The reference trainer starts each document as its own sequence of byte IDs, never concatenating adjacent documents. It counts adjacent ordered token-ID pairs within each document using checked `u64` counts. Pair frequency is the total number of occurrences across the training documents in the current sequence state. At each iteration it chooses the pair with greatest frequency; ties select the lexicographically smallest `(left_id, right_id)`. A candidate must occur at least twice. If none remains, training stops successfully with fewer merges. An empty corpus (including a set of empty documents) produces the byte-only BPE model with zero merges.

The new token receives the next contiguous ID. Its byte expansion is the concatenation of its operands' expansions. After selecting a pair, scan each document left to right and replace non-overlapping occurrences of exactly that pair. Thus in `a a a`, a merge `(a,a)` replaces the first two symbols and leaves the third; the resulting sequence is `[new,a]`. Recount all pairs from the resulting documents before selecting the next merge. Merges are ordered by this sequence and their zero-based positions are their ranks. The algorithm stores an ordered merge list and expansion table; it does not rely on hash iteration order.

This deliberately simple recounting algorithm is the reference implementation. No incremental heap, parallel counting, or approximate candidate selection is in scope until the reference has independent agreement. Work is measured as pair-count visits plus replacement scan visits across all rounds, checked before doing work; exceeding the configured budget returns `WorkLimitExceeded`, not a partial model advertised as complete. Pair counts and token IDs use checked arithmetic. Training on only authorized training split documents is mandatory. BPE must not be learned from validation/test documents.

Limits: 128 KiB total input corpus bytes, 512 merges, 64 Mi counted pair/replace symbol visits, and 4,096 bytes maximum expansion for any token. A learned expansion exceeding that maximum is rejected before insertion. These are project ceilings for the reference trainer, not useful-vocabulary guarantees. Training reports a complete model plus counts or a typed error; it never silently truncates a corpus or merge list.

### Encoding and decoding

Initialize the current sequence with one byte ID per input byte. Repeatedly find the adjacent pair with the lowest learned merge rank that exists in the current sequence, then replace all non-overlapping occurrences of that selected pair in a left-to-right scan. Repeat until no learned pair remains. This rank-first procedure defines overlap behavior and is deterministic. It does not cross calls or document boundaries. Its worst-case work is bounded by `input_token_count * merge_count` pair inspections plus replacement scans; the encoder enforces a 64 Mi comparison/visit budget and returns `WorkLimitExceeded` before exceeding it. Thus an input below the byte-size cap can still be rejected under a small work budget. A future optimized implementation must match these exact outputs.

Decode uses the artifact's already-validated expansion bytes. Every normal vocabulary token has a nonempty byte expansion no longer than 4,096 bytes. Before appending, the decoder checks cumulative output length against the caller's maximum (default 1 MiB) and uses fallible reserve. No model-supplied expansion triggers recursive unbounded expansion at decode time.

Artifact loading validates every token and merge: operands refer only to byte IDs or earlier merged IDs; result IDs are the exact next ID; result bytes equal concatenated operand expansions; no duplicate pair or result ID; each expansion is nonempty and within limit; ranks are contiguous; no special ID is an operand; vocabulary size equals the ID inventory. Two distinct token IDs may have the same byte expansion, so alias expansions are permitted and remain deterministic by token ID/rank. These rules reject forward references, cycles, duplicate definitions, malformed merges, and expansions that cannot be decoded safely.

## 4. Vocabulary candidates and model cost

Candidates below are total vocabulary sizes including four special IDs. The byte-only baseline has 260. BPE additions above that point count learned merges.

| Total `V` | Status in 004-B | Expected trade-off to measure |
|---:|---|---|
| 260 | Required reference | Maximal coverage and simplest behavior; one token per byte. |
| 1,024 | Tiny fixture candidate | Small output layer; limited recurring chunks. |
| 4,096 | Tiny research candidate | Moderate compression potential; inexpensive output head. |
| 8,192 | Research candidate | More lexical/code fragments; larger output layer. |
| 16,384 | Research candidate | More merges, tokenizer artifact and softmax cost. |
| 32,768 | Research candidate | Plausible evaluation point, not a recommendation. |
| 65,536 | Upper research candidate | Larger embedding/output matrices and sparse tail risk on small data. |

No production size is selected before measurements on lawfully reviewed representative English, Indonesian, source-code, Linux-documentation, and defensive cybersecurity data. BPE can reduce common sequence lengths but skew representation across languages and may fragment uncommon code, numeric forms, or scripts; this must be measured rather than assumed.

For hidden dimension `D`, one input embedding table contains `V*D` parameters. A separate untied output projection adds another `V*D`; tying can share the table but its feasibility is a later architecture decision. At `D=128`, V from 260/1,024/4,096/8,192/16,384/32,768/65,536 implies approximately 0.033/0.131/0.524/1.049/2.097/4.194/8.389 million input embedding parameters. For illustration only, at hypothetical `D=4096` and `D=5120`, V=32,768 means 134.2M and 167.8M embedding parameters respectively; an untied output table adds the same count. At V=65,536 it is 268.4M and 335.5M. These are arithmetic illustrations, not proposed architectures. Compared with hypothetical 8B or 14B total parameters, those tables are material but not dominant; the output logits still cost roughly `batch_tokens * V * D` multiply-add scale per projection, and more byte-level sequence positions also raise attention and feed-forward work. Tiny experiments must therefore report both tokens/byte and downstream token count; a short sequence alone does not determine total training cost.

## 5. Versioned tokenizer artifact

Use an inspectable UTF-8, LF-only canonical TSV artifact, no executable code and no network dependency. Format `xynoos-tokenizer`, version `1`. Fields are ASCII and tabs/newlines are forbidden inside fields. Byte strings are lowercase hex. Canonical line order is:

```text
XYNOOS-TOKENIZER\t1
algorithm\tbyte-v1|byte-bpe-v1
vocab_size\t<decimal>
special\t256\tBOS\t<|bos|>
special\t257\tEOS\t<|eos|>
special\t258\tPAD\t<|pad|>
special\t259\tDOC_SEP\t<|doc_sep|>
merge\t<rank>\t<left-id>\t<right-id>\t<new-id>\t<lowercase-hex-expansion>
...
sha256\t<64 lowercase hex digits>
```

The byte model omits merge lines and has `vocab_size=260`; BPE has one line per contiguous merge rank. The SHA-256 covers all bytes preceding the checksum line, including their LF terminators. Serialization must be byte-for-byte deterministic for the same model. The checksum detects accidental corruption; it is not a signature or proof of trusted origin. TASK 004-B may use the `sha2` crate rather than implement cryptography. Pin and review dependencies under the existing offline lockfile policy.

Maximum artifact file size is 8 MiB. The loader reads at most that amount, checks lengths before allocation/conversion, rejects CRLF, missing final LF, noncanonical decimal/hex spelling, unknown fields, duplicate fields, extra trailing data, checksum mismatch, unsupported version/algorithm, invalid IDs/ranks/expansions, or inconsistent `vocab_size`. Failed load returns no partially usable model. No large model/tokenizer artifact is committed to Git. This is deliberately not a general model checkpoint format.

## 6. Technical-text handling

Tokenization preserves exact bytes: C/C++ operators and comments, Rust punctuation and identifiers, Python indentation, shell quoting, filesystem paths, LF/CRLF bytes, tabs, repeated spaces, hexadecimal/binary values, IPv4/IPv6 and CIDR syntax, protocol names, English, Indonesian, and Unicode all enter as supplied. A text API represents valid Unicode using its exact UTF-8 encoding. Invalid UTF-8 is accepted by byte APIs and rejected only by strict text decoding. No lowercasing, whitespace collapse, Unicode normalization, syntax rewriting, or special-marker recognition occurs. Dataset cleaning is a distinct, logged transformation stage; it cannot be smuggled into the tokenizer.

## 7. Implementation errors and API boundary

Proposed public concepts: `TokenId`, `SpecialToken`, `Tokenizer`, `ByteTokenizer`, `BpeModel`, `BpeTrainer`, `TokenizerArtifact`, and `TokenizerLimits`. Methods include `encode_bytes`, `encode_text`, explicit special insertion/encoding, `decode_bytes`, and strict `decode_text`. Trainer accepts a deterministic iterator of document byte slices with declared train-only provenance context. All fallible operations return a typed `TokenizerError`, including `InputTooLarge`, `OutputTooLarge`, `WorkLimitExceeded`, `VocabularyLimitExceeded`, `ExpansionLimitExceeded`, `InvalidTokenId`, `InvalidSpecialSequence`, `InvalidUtf8`, `MalformedArtifact`, `ChecksumMismatch`, `UnsupportedVersion`, `ArithmeticOverflow`, and `AllocationFailed`.
