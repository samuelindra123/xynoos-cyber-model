# TASK 006 — Checkpoint and Resume Specification

**Status:** proposed version-one reference format for TASK 006-B/C. No checkpoint code or model weights are produced in TASK 006-A.

## 1. Boundary and identity

Version one checkpoints only at a completed optimizer transaction after persistent gradients have been cleared. It therefore records an accumulation count of zero. A mid-window checkpoint is not supported: an interrupted partial window is discarded and recomputed after resume. The format reserves an accumulation section and rejects a nonzero partial count in version one.

The checkpoint identifies experiment and model state separately. Required identities are:

- Canonical model ID xynoos-cyber-0.1, architecture field values, parameter count and ordered inventory digest.
- Source Git revision and clean-source status used to build/run.
- Canonical experiment configuration bytes and SHA-256 digest.
- Dataset manifest ID and digest, train/validation/test split digests, and provenance revision.
- Tokenizer algorithm, artifact digest, vocabulary size, and special-token registry.
- Initialization seed and stream version, even after the initialized weights have changed.
- Optimizer type/version/configuration, schedule kind/version/configuration, and completed update count.

Runtime-only owner identities, memory addresses, temporary paths, and host names are not serialized. They are recreated after validating the saved inventory and attached to a newly constructed parameter set.

## 2. Canonical binary format

The proposed version-one file is little-endian and has:

1. Fixed magic XYNOOS-CHECKPOINT, major/minor format version, header size, total byte length, endianness marker, precision code (f32 parameters/moments), and section count.
2. A fixed-order sequence of typed sections. Each has a four-byte tag, flags, unsigned 64-bit little-endian payload length, then payload. Required section tags and order are: identity/configuration, architecture/inventory, model values, AdamW config and state, schedule state, RNG state, dataset/order cursor, counters, accumulation marker, and integrity trailer.
3. An integrity trailer containing SHA-256 over all preceding exact bytes.

Every section payload is a sequence of field records: unsigned 16-bit field ID, unsigned 64-bit payload length, then exact payload bytes. Field IDs must be strictly increasing in each section. This makes duplicates, unknown fields, missing fields, and non-canonical ordering detectable. Repeated parameter records have a separately specified repeated-record container and must have strictly increasing inventory indices. Strings are exact UTF-8 bytes; no Unicode normalization is performed. Numeric model and optimizer arrays are IEEE-754 binary32 bit patterns in inventory and row-major order. FP64 bias powers are IEEE-754 binary64 bit patterns. Token IDs and counters use fixed-width unsigned little-endian integers. Booleans and enums have fixed one-byte values. Numeric hyperparameters use their declared IEEE bit representation, never locale-dependent decimal text.

The parser rejects unknown major versions, unsupported minor versions, unknown section tags, duplicate or out-of-order fields, unsupported flags, missing required sections, unexpected trailing bytes, invalid endianness/precision, invalid UTF-8, invalid lengths, count/shape mismatches, non-finite values, and checksum mismatch. A future optional field requires a minor-format rule that explicitly permits it; version one has no silently ignored fields.

All additions/multiplications for expected section lengths, tensor cell counts, and total size use checked u64 and checked host-size conversions. Reject any file above the configured maximum before allocating. For the fixed target, parameters and two FP32 moment arrays require exactly 10,972,800 bytes. Metadata, scalars, section headers, and digests are bounded; the initial hard file-size cap is 16 MiB. This cap leaves space above the fixed numeric payload without permitting an arbitrary archive. A changed architecture must recompute and explicitly approve its own cap.

## 3. Persisted state

The model section contains all 74 parameter tensors, each with canonical name, rank, dimensions, element count, and values. Tied input/output embeddings appear once and the inventory digest records that tying. The optimizer section contains first and second FP32 moments for every tensor, beta-power scalars, optimizer step, and its canonical hyperparameters/groups. The schedule section contains total updates, warmup updates, peak/final rates, and completed updates.

The RNG section records initialization stream version/seed and the training-order PRNG algorithm/state. Current Transformer dropout is absent; if stochastic model operations are introduced later, their RNG streams must be versioned and persisted before use. The data section records immutable dataset and tokenizer identities, epoch, deterministic order/permutation seed and digest, next document/window/batch cursor, and exact tail/padding rules. The counters section records input-token occurrences consumed, supervised targets consumed, validation targets, completed optimizer steps, last committed train/validation metric IDs, and stop/checkpoint sequence. The experiment section records all limits and hyperparameters, including the resource-policy identity. Any secret-bearing values are forbidden.

## 4. Atomic write and interrupted-write handling

Write to a uniquely named temporary file in the target directory using create-new semantics. Stream sections through a fixed small buffer while hashing; never build a second complete checkpoint in RAM. Validate the full staged state before creating the temporary file. Write all bytes, flush userspace buffers, sync file contents, rename atomically over the destination on the supported local filesystem, and sync the parent directory where the platform supports it. Do not claim these semantics for a remote/object filesystem without a separate tested contract.

If creation, writing, checksum, sync, or rename fails, leave the previous complete checkpoint intact and report checkpoint_failure. A crash may leave an orphan temporary file; on next start, ignore unrecognized temporary names and remove only files matching this run's own safe prefix after validating they are ordinary files beneath the configured artifact directory. Never follow symlinks. Do not delete the previous checkpoint until replacement succeeds.

## 5. Bounded, corruption-safe loading

Open only a configured regular file under the artifact root; reject symlinks, traversal, and unsupported file types. Read file metadata and reject zero/oversized lengths. First pass streams the full file, validates section framing/order and checksum without constructing model tensors. The parser checks exact expected payload sizes from the fixed model inventory before any tensor allocation.

After the first pass, allocate a private staging model and optimizer under the resource budget and read a second pass into them. Validate every finite value and identity while staging. Only return the newly loaded trainer after all sections and final digest match again. If any step fails, drop the private model, optimizer, reader and staging; the caller's current trainer remains untouched. Use fallible bounded allocation and explicit per-tensor and aggregate ledger checks. Never restore a partially parsed model.

## 6. Resume equivalence

For one source revision, target, configuration, data/tokenizer digest, seed, deterministic ordering, compiler/toolchain, and CPU reference profile, compare:

- An uninterrupted run of S updates.
- A run checkpointed after K completed updates, destroyed, loaded, and continued to S.

At S require exact equality of model FP32 bits, both optimizer moments, beta powers, update and schedule counters, data cursor/order identity, token counters, and corresponding deterministic scalar metrics. Also require that the next batch identity and next learning rate match immediately after resume. Store the checkpoint hash and source/config/data identities in both run reports.

Run wall time, log timestamps, process RSS and filesystem metadata are not equality inputs. A resumed runtime owner ID may differ but must be freshly bound. A format/architecture/tokenizer/optimizer mismatch is a typed compatibility rejection; no best-effort partial load or silent reset of moments is allowed.

## 7. Checkpoint test matrix

TASK 006-B/C must include unit and independent tests for canonical bytes, exact length, endian marker, unknown major/minor version, duplicate/unknown/reordered fields, oversized sections, checked-arithmetic overflow, malformed UTF-8/metadata, non-finite model/moment/power values, invalid tensor shapes, bad parameter inventory/tied identity, wrong dataset/tokenizer/source/config identities, checksum corruption, truncated header/section/trailer, trailing data, temporary-write failure, rename failure, symlink/path traversal, orphan temporary cleanup, and previous-checkpoint preservation.

The integration test compares uninterrupted and save/destroy/load/resumed runs at an interior update boundary. A fault-injected load must preserve the caller's live model. A failed write must preserve the prior checkpoint. Tests use small fixtures where possible and a bounded fixed-candidate checkpoint only when the training resource policy is explicitly enabled.

No checkpoints or model weights are committed to Git or included in PR artifacts.
