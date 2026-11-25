# TASK 006 — Canonical Checkpoint Format v1

This document freezes the version-one field map used by the native codec.
All offsets and lengths are checked before conversion to host `usize`. There
are no optional or ignorable version-one fields.

## File framing

The 40-byte header is, in order:

| Offset | Width | Type | Value |
|---:|---:|---|---|
| 0 | 18 | bytes | ASCII `XYNOOS-CHECKPOINT` followed by NUL |
| 18 | 2 | u16 | major version 1 |
| 20 | 2 | u16 | minor version 0 |
| 22 | 2 | u16 | header length 40 |
| 24 | 8 | u64 | exact complete file length |
| 32 | 4 | u32 | endianness marker `0x01020304` |
| 36 | 1 | u8 | precision code 1: IEEE binary32 parameters/moments |
| 37 | 1 | u8 | section count 10 |
| 38 | 2 | u16 | reserved zero |

Each section is `tag[4], flags:u32, payload_length:u64, payload`. Flags must
be zero. Sections occur exactly once and in this order: `IDEN`, `ARCH`,
`MODL`, `ADAM`, `SCHD`, `RNGS`, `DATA`, `CNTR`, `ACUM`, `HASH`.

Each non-integrity payload is a sequence of `field_id:u16, length:u64,
payload` records. IDs strictly increase and every listed field is required.
`HASH` contains field 1 of exactly 32 bytes: SHA-256 over all file bytes before
the `HASH` section header. The hash section is terminal. No trailing bytes are
allowed.

## Required fields

Lengths below are payload lengths; all integers and IEEE bit patterns are
little-endian. UTF-8 strings are exact bytes with no normalization and have
the stated inclusive byte limits.

| Section | ID | Field | Exact width / bound |
|---|---:|---|---|
| IDEN | 1 | model ID, exactly `xynoos-cyber-0.1` | 16 bytes |
| IDEN | 2 | source revision, lowercase hexadecimal | 40 bytes |
| IDEN | 3 | clean-source flag | 1 byte, 0 or 1 |
| IDEN | 4 | canonical configuration digest | 32 bytes |
| IDEN | 5 | dataset manifest ID | UTF-8, 1–128 bytes |
| IDEN | 6 | dataset manifest SHA-256 | 32 bytes |
| IDEN | 7 | train split SHA-256 | 32 bytes |
| IDEN | 8 | validation split SHA-256 | 32 bytes |
| IDEN | 9 | test split SHA-256 | 32 bytes |
| IDEN | 10 | provenance revision | UTF-8, 1–128 bytes |
| IDEN | 11 | tokenizer algorithm ID | UTF-8, 1–32 bytes |
| IDEN | 12 | tokenizer artifact SHA-256 | 32 bytes |
| IDEN | 13 | vocabulary size | u32, exactly 292 |
| IDEN | 14 | special-token registry SHA-256 | 32 bytes |
| IDEN | 15 | canonical run configuration | exactly 125 bytes, `RUN1` structure below |
| IDEN | 16 | resource-policy identity SHA-256 | 32 bytes |
| ARCH | 1 | vocabulary, width, layers, heads, FFN, context | 6 × u32 |
| ARCH | 2 | parameter count | u64, exactly 914400 |
| ARCH | 3 | tensor count | u32, exactly 74 |
| ARCH | 4 | ordered inventory SHA-256 | 32 bytes |
| ARCH | 5 | tied-identity code | u8, 1 means one shared embedding/output tensor |
| ARCH | 6 | ordered inventory records | bounded container, described below |
| MODL | 1 | tensor count | u32, exactly 74 |
| MODL | 2 | parameter records | bounded container, described below |
| ADAM | 1 | optimizer version | u32, exactly 1 |
| ADAM | 2 | beta1 | f32 |
| ADAM | 3 | beta2 | f32 |
| ADAM | 4 | epsilon | f32 |
| ADAM | 5 | weight decay | f32 |
| ADAM | 6 | completed optimizer step | u64 |
| ADAM | 7 | beta1 power | f64 |
| ADAM | 8 | beta2 power | f64 |
| ADAM | 9 | group policy | u8, 1 means rank-2 decay/rank-1 no decay |
| ADAM | 10 | paired moment records | bounded container, described below |
| SCHD | 1 | schedule kind | u8, 1 means linear warmup/linear decay |
| SCHD | 2 | schedule version | u32, exactly 1 |
| SCHD | 3 | total updates | u64 |
| SCHD | 4 | warmup updates | u64 |
| SCHD | 5 | peak rate | f32 |
| SCHD | 6 | final rate | f32 |
| SCHD | 7 | completed updates | u64, equal to ADAM field 6 |
| RNGS | 1 | initialization stream ID | u32, exactly 1 (SplitMix64/Box–Muller v1) |
| RNGS | 2 | initialization seed | u64 |
| RNGS | 3 | order-generator ID | u32, exactly 1 (SplitMix64 v1) |
| RNGS | 4 | order-generator state | 16 bytes: state u64 then draw count u64 |
| DATA | 1 | deterministic data-order identity | 32-byte SHA-256 |
| DATA | 2 | epoch | u64 |
| DATA | 3 | next document ordinal | u64 |
| DATA | 4 | next window ordinal | u64 |
| DATA | 5 | next batch ordinal | u64 |
| DATA | 6 | tail/padding policy | u8, 0 drop, 1 pad |
| DATA | 7 | next-batch identity | 32-byte SHA-256 |
| DATA | 8 | validation-order identity | 32-byte SHA-256 |
| CNTR | 1 | committed training input-token occurrences | u64 |
| CNTR | 2 | committed supervised targets | u64 |
| CNTR | 3 | validation supervised targets | u64 |
| CNTR | 4 | attempted input-token occurrences | u64 |
| CNTR | 5 | discarded input-token occurrences | u64 |
| CNTR | 6 | cumulative charged work | u64 |
| CNTR | 7 | cumulative active runtime nanoseconds | u64 |
| CNTR | 8 | completed optimizer updates | u64, equal to ADAM field 6 |
| CNTR | 9 | last committed metric sequence | u64 |
| CNTR | 10 | checkpoint sequence | u64 |
| CNTR | 11 | attempted optimizer updates | u64 |
| CNTR | 12 | attempted supervised targets | u64 |
| ACUM | 1 | accumulation format version | u8, exactly 1 |
| ACUM | 2 | partial-window microbatch count | u64, exactly 0 |
| HASH | 1 | file integrity digest | 32 bytes |

An inventory-record container begins with `count:u32`. Each record is
`index:u32, name_length:u16, name[UTF-8], rank:u8, dimensions:rank×u32`.
Indices are exactly 0 through 73, names are the canonical Transformer
inventory names, ranks and dimensions match the fixed architecture, and no
bytes remain after the final record.

The model-value container begins with `count:u32`. Each record is
`index:u32, element_count:u32, values:element_count×f32_bits`; tensor order and
element order are inventory and row-major order. The moment container uses
the same record prefix followed by first-moment values then second-moment
values. Its exact element count is derived from the validated inventory.
Values and moments must all be finite; second moments must be nonnegative.

The 125-byte `RUN1` field is `version:u32=1`,
`microbatches_per_update:u32`, then fourteen u64 values in order: maximum
updates, consumed-token cap, supervised-token cap, cumulative-work cap,
owned-memory cap, lineage active-runtime cap, per-attempt active-runtime cap,
checkpoint interval, validation interval, data/tokenizer allowance, autograd
byte cap, autograd value cap, autograd operation cap, and autograd work cap.
It ends with `clip_enabled:u8` and `clip_threshold:f32`. All caps are positive,
update count is at most 10000, data/tokenizer allowance is at most 64 MiB,
owned memory is at most 128 MiB, clip flag is 0 or 1, and an enabled clip
threshold is finite and positive. This field contains limits needed to validate
a resume; its SHA-256 is the resource-policy identity. The canonical
configuration digest in `IDEN` field 4 also binds the optimizer, schedule,
model, source, data, tokenizer, and order identities.

The writer streams sections through a reusable 64 KiB buffer while hashing.
No complete checkpoint byte vector is created. The decoder first streams and
checks framing, order, exact section/field lengths, file size, and SHA-256,
then streams the file again into private typed staging arrays and validates
all identities, shapes, counters, numerical values, schedule, and digest
before returning a trainer.

## Cursor, counters, and lineage contract

The committed cursor is `DATA` field 5 and always identifies the next batch
after the last fully committed optimizer update. A private in-flight cursor may advance while
building an accumulation window, but it is never checkpointed. Stop, failure,
or process interruption before optimizer commit discards that window and
replays it from the committed cursor after resume.

The bounded candidate data path uses manifest order without shuffling. Its
SplitMix64 order stream is therefore persisted at the initial seed with draw
count zero; `DATA` field 5 carries the advancing batch ordinal. Validation
uses its fixed first batch at every validation event, so it has no advancing
validation cursor.

Committed scientific token counters advance only with a successful optimizer
update. Attempted input-token counters advance when a valid batch is admitted;
the discarded count is derived from attempted minus committed and includes
windows that are abandoned or replayed. Cumulative work is charged before each
admitted operation and also belongs to the run lineage, so restarting from an
older checkpoint cannot refund work. The runtime cap applies both per process attempt and to cumulative
active execution time across resumes; offline time is excluded. The journal is
atomically replaced and synced before admitting work. If its write is uncertain,
the runtime stops without beginning the operation.

For input-token occurrences, `discarded = attempted - committed` is maintained
with checked subtraction. This includes a window discarded at an orderly stop
and admissions after the last checkpoint that must be replayed after a crash.
Replayed occurrences remain charged as attempted work, while committed
scientific counters advance only for the optimizer transactions present in the
checkpointed model state. The trainer attempt clock starts before model
initialization or checkpoint loading and remains active through run finalization;
dataset/tokenizer preparation performed by the caller before constructing the
trainer is outside that attempt clock but is charged to the configured data
memory allowance when the source and tokenizer are admitted.

## Publication contract and filesystem assumptions

The artifact root and every existing path component must be ordinary
directories/files, not symlinks. Checkpoints use a fixed safe filename under
the configured root and a create-new temporary sibling. The supported
publication contract assumes a local POSIX filesystem where same-directory
rename is atomic, file `sync_all` flushes the file, and opening/syncing the
parent directory is supported.

Any failure before successful rename leaves the previous checkpoint in place;
the temporary sibling is removed when cleanup itself succeeds. After rename,
the new checkpoint is already published. A later parent-directory sync error
returns `PublicationDurabilityUncertain`; it does not claim the old file still
exists. The contract does not eliminate path-component TOCTOU races, storage
controller loss, remote filesystem behavior, or power-loss behavior beyond
the stated file and directory sync requests. Tests exercise atomic rename and
faults before and after publication on the CI filesystem.
