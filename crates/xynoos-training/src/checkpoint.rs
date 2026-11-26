use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use sha2::{Digest, Sha256};
use xynoos_autograd::{AdamW, AdamWConfig, ResourceLimits};
use xynoos_tokenizer::Tokenizer;
use xynoos_transformer::{Config, Transformer};

use crate::{
    LinearWarmupDecay, Result, RunIdentity, RunLimits, TrainingConfig, TrainingError,
    config::MODEL_ID,
};

const MAGIC: &[u8; 18] = b"XYNOOS-CHECKPOINT\0";
const HEADER_BYTES: usize = 40;
const FILE_LIMIT: usize = 16 * 1024 * 1024;
const IO_BUFFER: usize = 64 * 1024;
const TAGS: [&[u8; 4]; 10] = [
    b"IDEN", b"ARCH", b"MODL", b"ADAM", b"SCHD", b"RNGS", b"DATA", b"CNTR", b"ACUM", b"HASH",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CheckpointCounters {
    pub committed_cursor: u64,
    pub committed_consumed_tokens: u64,
    pub committed_supervised_tokens: u64,
    pub validation_supervised_tokens: u64,
    pub attempted_consumed_tokens: u64,
    pub attempted_supervised_tokens: u64,
    pub discarded_consumed_tokens: u64,
    pub cumulative_work: u64,
    pub cumulative_runtime_ns: u64,
    pub completed_updates: u64,
    pub last_metric_sequence: u64,
    pub checkpoint_sequence: u64,
    pub attempted_updates: u64,
}

#[derive(Debug)]
struct Entry<'a> {
    name: String,
    shape: Vec<u32>,
    values: &'a [f32],
    first: &'a [f32],
    second: &'a [f32],
}

#[derive(Debug)]
pub struct CheckpointSnapshot<'a> {
    identity: &'a RunIdentity,
    config: TrainingConfig,
    entries: Vec<Entry<'a>>,
    adam: AdamWConfig,
    step: u64,
    beta1_power: f64,
    beta2_power: f64,
    counters: CheckpointCounters,
    inventory_digest: [u8; 32],
}

impl<'a> CheckpointSnapshot<'a> {
    pub fn new(
        identity: &'a RunIdentity,
        config: TrainingConfig,
        model: &'a Transformer,
        optimizer: &'a AdamW,
        counters: CheckpointCounters,
    ) -> Result<Self> {
        identity.validate_static()?;
        config.validate_static()?;
        if model.config() != Config::RESEARCH_CANDIDATE
            || optimizer.expected_revision() != model.parameters().revision()
            || optimizer.step_count() != counters.completed_updates
            || optimizer.config() != config.optimizer()
            || counters.completed_updates > config.schedule().total_updates()
            || counters.committed_supervised_tokens > counters.committed_consumed_tokens
            || counters.attempted_consumed_tokens < counters.committed_consumed_tokens
            || counters.attempted_supervised_tokens < counters.committed_supervised_tokens
            || counters.attempted_updates < counters.completed_updates
        {
            return Err(TrainingError::InvalidConfiguration);
        }
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(model.parameter_count().min(74))
            .map_err(|_| TrainingError::AllocationFailed)?;
        for (spec, id) in model.inventory() {
            let tensor = model.parameters().value(id)?;
            let first = optimizer
                .first_moment(spec.index())
                .ok_or(TrainingError::InvalidConfiguration)?;
            let second = optimizer
                .second_moment(spec.index())
                .ok_or(TrainingError::InvalidConfiguration)?;
            if tensor.data().len() != first.len() || first.len() != second.len() {
                return Err(TrainingError::InvalidConfiguration);
            }
            if !tensor.data().iter().all(|value| value.is_finite())
                || !first.iter().all(|value| value.is_finite())
                || !second
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0)
            {
                return Err(TrainingError::InvalidConfiguration);
            }
            let mut shape = Vec::new();
            shape
                .try_reserve_exact(spec.shape().len())
                .map_err(|_| TrainingError::AllocationFailed)?;
            for &dimension in spec.shape() {
                shape
                    .push(u32::try_from(dimension).map_err(|_| TrainingError::ArithmeticOverflow)?);
            }
            entries.push(Entry {
                name: spec.to_string(),
                shape,
                values: tensor.data(),
                first,
                second,
            });
        }
        if entries.len() != 74 || model.parameter_count() != 914_400 {
            return Err(TrainingError::InvalidConfiguration);
        }
        let inventory_digest = inventory_digest(&entries)?;
        Ok(Self {
            identity,
            config,
            entries,
            adam: optimizer.config(),
            step: optimizer.step_count(),
            beta1_power: optimizer.beta1_power(),
            beta2_power: optimizer.beta2_power(),
            counters,
            inventory_digest,
        })
    }

    pub fn counters(&self) -> CheckpointCounters {
        self.counters
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointData {
    pub identity: RunIdentity,
    pub config: TrainingConfig,
    pub parameters: Vec<Vec<f32>>,
    pub first_moments: Vec<Vec<f32>>,
    pub second_moments: Vec<Vec<f32>>,
    pub optimizer_step: u64,
    pub beta1_power: f64,
    pub beta2_power: f64,
    pub cursor: u64,
    pub next_batch_identity: [u8; 32],
    pub order_state: u64,
    pub order_draw_count: u64,
    pub counters: CheckpointCounters,
    pub digest: [u8; 32],
    pub configuration_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointPublication {
    pub path: String,
    pub bytes: u64,
    pub sha256: [u8; 32],
    pub durable: bool,
}

pub struct CheckpointCodec;

impl CheckpointCodec {
    /// Stream a canonical checkpoint to a create-new sibling, sync it, then
    /// atomically replace `checkpoint.xyckpt`. A post-rename directory-sync
    /// failure is reported as publication uncertainty.
    pub fn write_atomic(
        root: &Path,
        snapshot: &CheckpointSnapshot<'_>,
    ) -> Result<CheckpointPublication> {
        Self::write_atomic_impl(root, snapshot, false, false)
    }

    fn write_atomic_impl(
        root: &Path,
        snapshot: &CheckpointSnapshot<'_>,
        fail_before_rename: bool,
        fail_after_rename: bool,
    ) -> Result<CheckpointPublication> {
        validate_root(root)?;
        let target = root.join("checkpoint.xyckpt");
        reject_unsafe_existing(&target)?;
        let lengths = SectionLengths::new(snapshot)?;
        if lengths.total > FILE_LIMIT as u64 {
            return Err(TrainingError::ResourceLimit {
                requested: usize::try_from(lengths.total).unwrap_or(usize::MAX),
                limit: FILE_LIMIT,
            });
        }
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let serial = NEXT.fetch_add(1, Ordering::Relaxed);
        let temp = root.join(format!(".xyckpt-{}-{serial}.tmp", std::process::id()));
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        let result = (|| {
            let mut output = StreamWriter::new(BufWriter::with_capacity(IO_BUFFER, file));
            write_snapshot(&mut output, snapshot, lengths)?;
            let digest = output.digest();
            output.write_unhashed(TAGS[9])?;
            output.write_unhashed(&0_u32.to_le_bytes())?;
            output.write_unhashed(&42_u64.to_le_bytes())?;
            output.write_unhashed(&1_u16.to_le_bytes())?;
            output.write_unhashed(&32_u64.to_le_bytes())?;
            output.write_unhashed(&digest)?;
            output.flush_sync()?;
            if fail_before_rename {
                return Err(TrainingError::Io(std::io::ErrorKind::Other));
            }
            fs::rename(&temp, &target)?;
            if fail_after_rename {
                return Err(TrainingError::PublicationDurabilityUncertain);
            }
            match File::open(root).and_then(|directory| directory.sync_all()) {
                Ok(()) => Ok(CheckpointPublication {
                    path: "checkpoint.xyckpt".into(),
                    bytes: lengths.total,
                    sha256: digest,
                    durable: true,
                }),
                Err(_) => Err(TrainingError::PublicationDurabilityUncertain),
            }
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
        result
    }

    /// Validate identity and config after two bounded streaming passes. The
    /// returned vectors are private typed staging values, never a byte archive.
    pub fn load(
        root: &Path,
        expected_identity: &RunIdentity,
        expected_config: TrainingConfig,
        tokenizer: &Tokenizer,
    ) -> Result<CheckpointData> {
        validate_root(root)?;
        let path = root.join("checkpoint.xyckpt");
        reject_unsafe_existing(&path)?;
        let metadata = fs::symlink_metadata(&path)?;
        if !metadata.is_file()
            || metadata.file_type().is_symlink()
            || metadata.len() == 0
            || metadata.len() > FILE_LIMIT as u64
        {
            return Err(TrainingError::Checkpoint("invalid_file_size_or_type"));
        }
        let digest = scan_file(&path, metadata.len())?;
        let mut data = decode_file(&path, digest)?;
        data.identity.validate(tokenizer)?;
        if data.identity != *expected_identity
            || data.config.digest(&data.identity) != expected_config.digest(expected_identity)
            || data.config.digest(&data.identity) != data.configuration_digest
        {
            return Err(TrainingError::Checkpoint(
                "identity_or_configuration_mismatch",
            ));
        }
        if data.config != expected_config {
            return Err(TrainingError::Checkpoint("configuration_mismatch"));
        }
        data.digest = digest;
        Ok(data)
    }
}

#[derive(Debug, Clone, Copy)]
struct SectionLengths {
    payloads: [u64; 9],
    total: u64,
}

impl SectionLengths {
    fn new(snapshot: &CheckpointSnapshot<'_>) -> Result<Self> {
        let inventory = inventory_payload_len(&snapshot.entries)?;
        let values = tensor_payload_len(&snapshot.entries, 4)?;
        let moments = tensor_payload_len(&snapshot.entries, 8)?;
        let run_config = run_config_bytes(snapshot.config);
        let payloads = [
            section_payload(&[
                16,
                40,
                1,
                32,
                len(snapshot.identity.dataset_id().len())?,
                32,
                32,
                32,
                32,
                len(snapshot.identity.provenance_revision().len())?,
                len(snapshot.identity.tokenizer_algorithm().len())?,
                32,
                4,
                32,
                run_config.len() as u64,
                32,
            ])?,
            section_payload(&[24, 8, 4, 32, 1, inventory])?,
            section_payload(&[4, values])?,
            section_payload(&[4, 4, 4, 4, 4, 8, 8, 8, 1, moments])?,
            section_payload(&[1, 4, 8, 8, 4, 4, 8])?,
            section_payload(&[4, 8, 4, 16])?,
            section_payload(&[32, 8, 8, 8, 8, 1, 32, 32])?,
            section_payload(&[8; 12])?,
            section_payload(&[1, 8])?,
        ];
        let mut total = HEADER_BYTES as u64;
        for payload in payloads {
            total = add_u64(total, add_u64(16, payload)?)?;
        }
        // HASH section and its single 32-byte digest field.
        total = add_u64(total, 58)?;
        Ok(Self { payloads, total })
    }
}

fn write_snapshot<W: Write>(
    out: &mut StreamWriter<W>,
    s: &CheckpointSnapshot<'_>,
    lengths: SectionLengths,
) -> Result<()> {
    let mut header = [0_u8; HEADER_BYTES];
    header[..18].copy_from_slice(MAGIC);
    header[18..20].copy_from_slice(&1_u16.to_le_bytes());
    header[20..22].copy_from_slice(&0_u16.to_le_bytes());
    header[22..24].copy_from_slice(&(HEADER_BYTES as u16).to_le_bytes());
    header[24..32].copy_from_slice(&lengths.total.to_le_bytes());
    header[32..36].copy_from_slice(&0x01020304_u32.to_le_bytes());
    header[36] = 1;
    header[37] = 10;
    out.write_hashed(&header)?;

    write_section(out, TAGS[0], lengths.payloads[0], |out| {
        write_identity(out, s)
    })?;
    write_section(out, TAGS[1], lengths.payloads[1], |out| write_arch(out, s))?;
    write_section(out, TAGS[2], lengths.payloads[2], |out| write_model(out, s))?;
    write_section(out, TAGS[3], lengths.payloads[3], |out| write_adam(out, s))?;
    write_section(out, TAGS[4], lengths.payloads[4], |out| {
        write_schedule(out, s)
    })?;
    write_section(out, TAGS[5], lengths.payloads[5], |out| write_rng(out, s))?;
    write_section(out, TAGS[6], lengths.payloads[6], |out| write_data(out, s))?;
    write_section(out, TAGS[7], lengths.payloads[7], |out| {
        write_counters(out, s)
    })?;
    write_section(out, TAGS[8], lengths.payloads[8], |out| {
        write_accumulation(out)
    })?;
    Ok(())
}

fn write_identity<W: Write>(out: &mut StreamWriter<W>, s: &CheckpointSnapshot<'_>) -> Result<()> {
    let identity = s.identity;
    let run = run_config_bytes(s.config);
    let policy_digest: [u8; 32] = Sha256::digest(run).into();
    write_field(out, 1, MODEL_ID.as_bytes())?;
    write_field(out, 2, identity.source_revision().as_bytes())?;
    write_field(out, 3, &[u8::from(identity.source_clean())])?;
    write_field(out, 4, &s.config.digest(identity))?;
    write_field(out, 5, identity.dataset_id().as_bytes())?;
    write_field(out, 6, &identity.dataset_manifest_sha256())?;
    write_field(out, 7, &identity.train_split_sha256())?;
    write_field(out, 8, &identity.validation_split_sha256())?;
    write_field(out, 9, &identity.test_split_sha256())?;
    write_field(out, 10, identity.provenance_revision().as_bytes())?;
    write_field(out, 11, identity.tokenizer_algorithm().as_bytes())?;
    write_field(out, 12, &identity.tokenizer_sha256())?;
    write_u32(out, 13, identity.vocabulary_size())?;
    write_field(out, 14, &identity.special_registry_sha256())?;
    write_field(out, 15, &run)?;
    write_field(out, 16, &policy_digest)
}

fn write_arch<W: Write>(out: &mut StreamWriter<W>, s: &CheckpointSnapshot<'_>) -> Result<()> {
    let model = s.config.model();
    let mut dims = Vec::with_capacity(24);
    for value in [
        model.vocabulary,
        model.width,
        model.layers,
        model.heads,
        model.ffn,
        model.context,
    ] {
        dims.extend_from_slice(
            &u32::try_from(value)
                .map_err(|_| TrainingError::ArithmeticOverflow)?
                .to_le_bytes(),
        );
    }
    write_field(out, 1, &dims)?;
    write_u64(out, 2, 914_400)?;
    write_u32(out, 3, 74)?;
    write_field(out, 4, &s.inventory_digest)?;
    write_field(out, 5, &[1])?;
    write_field_stream(out, 6, inventory_payload_len(&s.entries)?, |out| {
        out.write_hashed(&74_u32.to_le_bytes())?;
        for (index, entry) in s.entries.iter().enumerate() {
            out.write_hashed(
                &u32::try_from(index)
                    .map_err(|_| TrainingError::ArithmeticOverflow)?
                    .to_le_bytes(),
            )?;
            out.write_hashed(
                &u16::try_from(entry.name.len())
                    .map_err(|_| TrainingError::ArithmeticOverflow)?
                    .to_le_bytes(),
            )?;
            out.write_hashed(entry.name.as_bytes())?;
            out.write_hashed(&[
                u8::try_from(entry.shape.len()).map_err(|_| TrainingError::ArithmeticOverflow)?
            ])?;
            for &dimension in &entry.shape {
                out.write_hashed(&dimension.to_le_bytes())?;
            }
        }
        Ok(())
    })
}

fn write_model<W: Write>(out: &mut StreamWriter<W>, s: &CheckpointSnapshot<'_>) -> Result<()> {
    write_u32(out, 1, 74)?;
    write_field_stream(out, 2, tensor_payload_len(&s.entries, 4)?, |out| {
        write_tensor_records(out, &s.entries, false)
    })
}

fn write_adam<W: Write>(out: &mut StreamWriter<W>, s: &CheckpointSnapshot<'_>) -> Result<()> {
    write_u32(out, 1, 1)?;
    write_f32(out, 2, s.adam.beta1())?;
    write_f32(out, 3, s.adam.beta2())?;
    write_f32(out, 4, s.adam.epsilon())?;
    write_f32(out, 5, s.adam.weight_decay())?;
    write_u64(out, 6, s.step)?;
    write_f64(out, 7, s.beta1_power)?;
    write_f64(out, 8, s.beta2_power)?;
    write_field(out, 9, &[1])?;
    write_field_stream(out, 10, tensor_payload_len(&s.entries, 8)?, |out| {
        write_tensor_records(out, &s.entries, true)
    })
}

fn write_schedule<W: Write>(out: &mut StreamWriter<W>, s: &CheckpointSnapshot<'_>) -> Result<()> {
    let schedule = s.config.schedule();
    write_field(out, 1, &[1])?;
    write_u32(out, 2, 1)?;
    write_u64(out, 3, schedule.total_updates())?;
    write_u64(out, 4, schedule.warmup_updates())?;
    write_f32(out, 5, schedule.peak_rate())?;
    write_f32(out, 6, schedule.final_rate())?;
    write_u64(out, 7, s.step)
}

fn write_rng<W: Write>(out: &mut StreamWriter<W>, s: &CheckpointSnapshot<'_>) -> Result<()> {
    write_u32(out, 1, 1)?;
    write_u64(out, 2, s.config.seed())?;
    write_u32(out, 3, 1)?;
    let mut state = [0_u8; 16];
    state[..8].copy_from_slice(&s.config.seed().to_le_bytes());
    state[8..].copy_from_slice(&0_u64.to_le_bytes());
    write_field(out, 4, &state)
}

fn write_data<W: Write>(out: &mut StreamWriter<W>, s: &CheckpointSnapshot<'_>) -> Result<()> {
    let cursor = s.counters.committed_cursor;
    let next_batch = next_batch_identity(&s.identity.data_order_identity(), cursor);
    write_field(out, 1, &s.identity.data_order_identity())?;
    write_u64(out, 2, 0)?;
    write_u64(out, 3, 0)?;
    write_u64(out, 4, 0)?;
    write_u64(out, 5, cursor)?;
    write_field(out, 6, &[0])?;
    write_field(out, 7, &next_batch)?;
    write_field(out, 8, &s.identity.validation_order_identity())
}

fn write_counters<W: Write>(out: &mut StreamWriter<W>, s: &CheckpointSnapshot<'_>) -> Result<()> {
    let c = s.counters;
    for (id, value) in [
        (1, c.committed_consumed_tokens),
        (2, c.committed_supervised_tokens),
        (3, c.validation_supervised_tokens),
        (4, c.attempted_consumed_tokens),
        (5, c.discarded_consumed_tokens),
        (6, c.cumulative_work),
        (7, c.cumulative_runtime_ns),
        (8, c.completed_updates),
        (9, c.last_metric_sequence),
        (10, c.checkpoint_sequence),
        (11, c.attempted_updates),
        (12, c.attempted_supervised_tokens),
    ] {
        write_u64(out, id, value)?;
    }
    Ok(())
}

fn write_accumulation<W: Write>(out: &mut StreamWriter<W>) -> Result<()> {
    write_field(out, 1, &[1])?;
    write_u64(out, 2, 0)
}

fn write_tensor_records<W: Write>(
    out: &mut StreamWriter<W>,
    entries: &[Entry<'_>],
    moments: bool,
) -> Result<()> {
    out.write_hashed(&74_u32.to_le_bytes())?;
    for (index, entry) in entries.iter().enumerate() {
        out.write_hashed(
            &u32::try_from(index)
                .map_err(|_| TrainingError::ArithmeticOverflow)?
                .to_le_bytes(),
        )?;
        out.write_hashed(
            &u32::try_from(entry.values.len())
                .map_err(|_| TrainingError::ArithmeticOverflow)?
                .to_le_bytes(),
        )?;
        if moments {
            out.write_f32_slice(entry.first)?;
            out.write_f32_slice(entry.second)?;
        } else {
            out.write_f32_slice(entry.values)?;
        }
    }
    Ok(())
}

fn write_section<W: Write>(
    out: &mut StreamWriter<W>,
    tag: &[u8; 4],
    payload_len: u64,
    body: impl FnOnce(&mut StreamWriter<W>) -> Result<()>,
) -> Result<()> {
    out.write_hashed(tag)?;
    out.write_hashed(&0_u32.to_le_bytes())?;
    out.write_hashed(&payload_len.to_le_bytes())?;
    body(out)
}

fn write_field<W: Write>(out: &mut StreamWriter<W>, id: u16, payload: &[u8]) -> Result<()> {
    write_field_stream(out, id, payload.len() as u64, |out| {
        out.write_hashed(payload)
    })
}

fn write_field_stream<W: Write>(
    out: &mut StreamWriter<W>,
    id: u16,
    length: u64,
    body: impl FnOnce(&mut StreamWriter<W>) -> Result<()>,
) -> Result<()> {
    out.write_hashed(&id.to_le_bytes())?;
    out.write_hashed(&length.to_le_bytes())?;
    body(out)
}

fn write_u32<W: Write>(out: &mut StreamWriter<W>, id: u16, value: u32) -> Result<()> {
    write_field(out, id, &value.to_le_bytes())
}
fn write_u64<W: Write>(out: &mut StreamWriter<W>, id: u16, value: u64) -> Result<()> {
    write_field(out, id, &value.to_le_bytes())
}
fn write_f32<W: Write>(out: &mut StreamWriter<W>, id: u16, value: f32) -> Result<()> {
    write_field(out, id, &value.to_bits().to_le_bytes())
}
fn write_f64<W: Write>(out: &mut StreamWriter<W>, id: u16, value: f64) -> Result<()> {
    write_field(out, id, &value.to_bits().to_le_bytes())
}

struct StreamWriter<W: Write> {
    writer: W,
    hasher: Sha256,
}

impl<W: Write> StreamWriter<W> {
    fn new(writer: W) -> Self {
        Self {
            writer,
            hasher: Sha256::new(),
        }
    }
    fn write_hashed(&mut self, bytes: &[u8]) -> Result<()> {
        self.writer.write_all(bytes)?;
        self.hasher.update(bytes);
        Ok(())
    }
    fn write_unhashed(&mut self, bytes: &[u8]) -> Result<()> {
        self.writer.write_all(bytes).map_err(Into::into)
    }
    fn digest(&self) -> [u8; 32] {
        self.hasher.clone().finalize().into()
    }
    fn write_f32_slice(&mut self, values: &[f32]) -> Result<()> {
        for chunk in values.chunks(IO_BUFFER / 4) {
            let mut bytes = [0_u8; IO_BUFFER];
            for (index, value) in chunk.iter().enumerate() {
                bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_bits().to_le_bytes());
            }
            self.write_hashed(&bytes[..chunk.len() * 4])?;
        }
        Ok(())
    }
}

impl StreamWriter<BufWriter<File>> {
    fn flush_sync(&mut self) -> Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(())
    }
}

fn run_config_bytes(config: TrainingConfig) -> [u8; 125] {
    let limits = config.limits();
    let memory = config.memory();
    let autograd = config.autograd_limits();
    let mut bytes = [0_u8; 125];
    let mut cursor = 0;
    macro_rules! put_u32 {
        ($value:expr) => {{
            bytes[cursor..cursor + 4].copy_from_slice(&($value as u32).to_le_bytes());
            cursor += 4;
        }};
    }
    macro_rules! put_u64 {
        ($value:expr) => {{
            bytes[cursor..cursor + 8].copy_from_slice(&($value as u64).to_le_bytes());
            cursor += 8;
        }};
    }
    put_u32!(1);
    put_u32!(limits.microbatches_per_update());
    for value in [
        limits.max_updates(),
        limits.consumed_tokens(),
        limits.supervised_tokens(),
        limits.cumulative_work(),
        limits.owned_memory_bytes() as u64,
        limits.max_runtime().as_nanos() as u64,
        limits.max_attempt_runtime().as_nanos() as u64,
        limits.checkpoint_every_updates(),
        limits.validation_every_updates(),
        memory.data_allowance() as u64,
        autograd.max_bytes() as u64,
        autograd.max_values() as u64,
        autograd.max_operations() as u64,
        autograd.max_work() as u64,
    ] {
        put_u64!(value);
    }
    match config.gradient_clip() {
        Some(value) => {
            bytes[cursor] = 1;
            cursor += 1;
            bytes[cursor..cursor + 4].copy_from_slice(&value.to_bits().to_le_bytes());
            cursor += 4;
        }
        None => {
            bytes[cursor] = 0;
            cursor += 1;
            bytes[cursor..cursor + 4].copy_from_slice(&0_u32.to_le_bytes());
            cursor += 4;
        }
    }
    debug_assert_eq!(cursor, bytes.len());
    bytes
}

fn inventory_digest(entries: &[Entry<'_>]) -> Result<[u8; 32]> {
    let mut hasher = Sha256::new();
    hasher.update(b"XYNOOS-INVENTORY-V1\0");
    for (index, entry) in entries.iter().enumerate() {
        hasher.update(
            u32::try_from(index)
                .map_err(|_| TrainingError::ArithmeticOverflow)?
                .to_le_bytes(),
        );
        hasher.update(
            u16::try_from(entry.name.len())
                .map_err(|_| TrainingError::ArithmeticOverflow)?
                .to_le_bytes(),
        );
        hasher.update(entry.name.as_bytes());
        hasher.update([
            u8::try_from(entry.shape.len()).map_err(|_| TrainingError::ArithmeticOverflow)?
        ]);
        for &dimension in &entry.shape {
            hasher.update(dimension.to_le_bytes());
        }
    }
    Ok(hasher.finalize().into())
}

fn inventory_payload_len(entries: &[Entry<'_>]) -> Result<u64> {
    let mut bytes = 4_u64;
    for entry in entries {
        bytes = add_u64(
            bytes,
            4 + 2
                + u64::try_from(entry.name.len()).map_err(|_| TrainingError::ArithmeticOverflow)?
                + 1,
        )?;
        bytes = add_u64(
            bytes,
            u64::try_from(entry.shape.len())
                .map_err(|_| TrainingError::ArithmeticOverflow)?
                .checked_mul(4)
                .ok_or(TrainingError::ArithmeticOverflow)?,
        )?;
    }
    Ok(bytes)
}

fn tensor_payload_len(entries: &[Entry<'_>], bytes_per_value: u64) -> Result<u64> {
    let mut bytes = 4_u64;
    for entry in entries {
        bytes = add_u64(bytes, 8)?;
        let count =
            u64::try_from(entry.values.len()).map_err(|_| TrainingError::ArithmeticOverflow)?;
        bytes = add_u64(
            bytes,
            count
                .checked_mul(bytes_per_value)
                .ok_or(TrainingError::ArithmeticOverflow)?,
        )?;
    }
    Ok(bytes)
}

fn section_payload(fields: &[u64]) -> Result<u64> {
    fields
        .iter()
        .try_fold(0_u64, |total, length| add_u64(total, add_u64(10, *length)?))
}

fn len(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| TrainingError::ArithmeticOverflow)
}
fn add_u64(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right)
        .ok_or(TrainingError::ArithmeticOverflow)
}

pub(crate) fn next_batch_identity(order: &[u8; 32], cursor: u64) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"XYNOOS-NEXT-BATCH-V1\0");
    hasher.update(order);
    hasher.update(cursor.to_le_bytes());
    hasher.finalize().into()
}

fn validate_root(root: &Path) -> Result<()> {
    if !root.is_absolute() {
        return Err(TrainingError::InvalidConfiguration);
    }
    let mut current = PathBuf::new();
    for component in root.components() {
        current.push(component.as_os_str());
        let metadata = fs::symlink_metadata(&current)?;
        if metadata.file_type().is_symlink() {
            return Err(TrainingError::InvalidConfiguration);
        }
        if !metadata.is_dir() {
            return Err(TrainingError::InvalidConfiguration);
        }
    }
    Ok(())
}

fn reject_unsafe_existing(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => Ok(()),
        Ok(_) => Err(TrainingError::InvalidConfiguration),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

struct ScanReader<R: Read> {
    input: R,
    position: u64,
    hasher: Sha256,
}

impl<R: Read> ScanReader<R> {
    fn new(input: R) -> Self {
        Self {
            input,
            position: 0,
            hasher: Sha256::new(),
        }
    }
    fn hashed(&mut self, bytes: &mut [u8]) -> Result<()> {
        read_exact_bounded(&mut self.input, bytes)?;
        self.hasher.update(&*bytes);
        self.position = self
            .position
            .checked_add(bytes.len() as u64)
            .ok_or(TrainingError::ArithmeticOverflow)?;
        Ok(())
    }
    fn unhashed(&mut self, bytes: &mut [u8]) -> Result<()> {
        read_exact_bounded(&mut self.input, bytes)?;
        self.position = self
            .position
            .checked_add(bytes.len() as u64)
            .ok_or(TrainingError::ArithmeticOverflow)?;
        Ok(())
    }
    fn skip_hashed(&mut self, mut length: u64) -> Result<()> {
        let mut buffer = [0_u8; IO_BUFFER];
        while length > 0 {
            let count = usize::try_from(length.min(IO_BUFFER as u64))
                .map_err(|_| TrainingError::ArithmeticOverflow)?;
            self.hashed(&mut buffer[..count])?;
            length -= count as u64;
        }
        Ok(())
    }
    fn digest(&self) -> [u8; 32] {
        self.hasher.clone().finalize().into()
    }
}

fn scan_file(path: &Path, actual_length: u64) -> Result<[u8; 32]> {
    let file = File::open(path)?;
    let mut reader = ScanReader::new(BufReader::with_capacity(IO_BUFFER, file));
    let mut header = [0_u8; HEADER_BYTES];
    reader.hashed(&mut header)?;
    validate_header(&header, actual_length)?;
    for (section_index, expected_tag) in TAGS[..9].iter().enumerate() {
        let mut tag = [0_u8; 4];
        reader.hashed(&mut tag)?;
        let flags = read_hashed_u32(&mut reader)?;
        let payload_length = read_hashed_u64(&mut reader)?;
        if &tag != *expected_tag || flags != 0 {
            return Err(TrainingError::Checkpoint("invalid_section_order_or_flags"));
        }
        let end = reader
            .position
            .checked_add(payload_length)
            .ok_or(TrainingError::ArithmeticOverflow)?;
        if end > actual_length {
            return Err(TrainingError::Checkpoint("section_out_of_bounds"));
        }
        let mut previous = 0_u16;
        for &expected_id in required_fields(section_index) {
            let id = read_hashed_u16(&mut reader)?;
            let length = read_hashed_u64(&mut reader)?;
            if id != expected_id || id <= previous || !field_length_valid(section_index, id, length)
            {
                return Err(TrainingError::Checkpoint("invalid_field_id_or_length"));
            }
            previous = id;
            let after = reader
                .position
                .checked_add(length)
                .ok_or(TrainingError::ArithmeticOverflow)?;
            if after > end {
                return Err(TrainingError::Checkpoint("field_out_of_bounds"));
            }
            reader.skip_hashed(length)?;
        }
        if reader.position != end {
            return Err(TrainingError::Checkpoint("section_length_mismatch"));
        }
    }
    let mut tag = [0_u8; 4];
    reader.unhashed(&mut tag)?;
    let flags = read_unhashed_u32(&mut reader)?;
    let section_length = read_unhashed_u64(&mut reader)?;
    if &tag != TAGS[9] || flags != 0 || section_length != 42 {
        return Err(TrainingError::Checkpoint("invalid_integrity_section"));
    }
    let id = read_unhashed_u16(&mut reader)?;
    let length = read_unhashed_u64(&mut reader)?;
    if id != 1 || length != 32 {
        return Err(TrainingError::Checkpoint("invalid_integrity_field"));
    }
    let mut expected = [0_u8; 32];
    reader.unhashed(&mut expected)?;
    let digest = reader.digest();
    if digest != expected {
        return Err(TrainingError::Checkpoint("checksum_mismatch"));
    }
    if reader.position != actual_length {
        return Err(TrainingError::Checkpoint("trailing_or_missing_bytes"));
    }
    Ok(digest)
}

fn required_fields(section: usize) -> &'static [u16] {
    const IDEN: &[u16] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    const ARCH: &[u16] = &[1, 2, 3, 4, 5, 6];
    const MODL: &[u16] = &[1, 2];
    const ADAM: &[u16] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const SCHD: &[u16] = &[1, 2, 3, 4, 5, 6, 7];
    const RNGS: &[u16] = &[1, 2, 3, 4];
    const DATA: &[u16] = &[1, 2, 3, 4, 5, 6, 7, 8];
    const CNTR: &[u16] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const ACUM: &[u16] = &[1, 2];
    match section {
        0 => IDEN,
        1 => ARCH,
        2 => MODL,
        3 => ADAM,
        4 => SCHD,
        5 => RNGS,
        6 => DATA,
        7 => CNTR,
        _ => ACUM,
    }
}

fn field_length_valid(section: usize, id: u16, length: u64) -> bool {
    let exact = match (section, id) {
        (0, 1) => Some(16),
        (0, 2) => Some(40),
        (0, 3) => Some(1),
        (0, 4) => Some(32),
        (0, 6..=9) => Some(32),
        (0, 12) => Some(32),
        (0, 13) => Some(4),
        (0, 14) => Some(32),
        (0, 15) => Some(125),
        (0, 16) => Some(32),
        (1, 1) => Some(24),
        (1, 2) => Some(8),
        (1, 3) => Some(4),
        (1, 4) => Some(32),
        (1, 5) => Some(1),
        (2, 1) => Some(4),
        (3, 1..=5) => Some(4),
        (3, 6) => Some(8),
        (3, 7..=8) => Some(8),
        (3, 9) => Some(1),
        (4, 1) => Some(1),
        (4, 2) => Some(4),
        (4, 3..=4) => Some(8),
        (4, 5..=6) => Some(4),
        (4, 7) => Some(8),
        (5, 1) => Some(4),
        (5, 2) => Some(8),
        (5, 3) => Some(4),
        (5, 4) => Some(16),
        (6, 1) => Some(32),
        (6, 2..=5) => Some(8),
        (6, 6) => Some(1),
        (6, 7..=8) => Some(32),
        (7, _) => Some(8),
        (8, 1) => Some(1),
        (8, 2) => Some(8),
        _ => None,
    };
    if let Some(exact) = exact {
        return length == exact;
    }
    match (section, id) {
        (0, 5) | (0, 10) => (1..=128).contains(&length),
        (0, 11) => (1..=32).contains(&length),
        (1, 6) => length <= 4096,
        (2, 2) | (3, 10) => length <= FILE_LIMIT as u64,
        _ => false,
    }
}

fn validate_header(header: &[u8; HEADER_BYTES], actual_length: u64) -> Result<()> {
    if &header[..18] != MAGIC
        || u16::from_le_bytes(
            header[18..20]
                .try_into()
                .map_err(|_| TrainingError::Checkpoint("header"))?,
        ) != 1
        || u16::from_le_bytes(
            header[20..22]
                .try_into()
                .map_err(|_| TrainingError::Checkpoint("header"))?,
        ) != 0
        || u16::from_le_bytes(
            header[22..24]
                .try_into()
                .map_err(|_| TrainingError::Checkpoint("header"))?,
        ) as usize
            != HEADER_BYTES
        || u64::from_le_bytes(
            header[24..32]
                .try_into()
                .map_err(|_| TrainingError::Checkpoint("header"))?,
        ) != actual_length
        || u32::from_le_bytes(
            header[32..36]
                .try_into()
                .map_err(|_| TrainingError::Checkpoint("header"))?,
        ) != 0x01020304
        || header[36] != 1
        || header[37] != 10
        || header[38] != 0
        || header[39] != 0
    {
        return Err(TrainingError::Checkpoint("unsupported_or_malformed_header"));
    }
    Ok(())
}

fn read_exact_bounded(reader: &mut impl Read, bytes: &mut [u8]) -> Result<()> {
    reader.read_exact(bytes).map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            TrainingError::Checkpoint("truncated")
        } else {
            TrainingError::Io(error.kind())
        }
    })
}
fn read_hashed_u16<R: Read>(reader: &mut ScanReader<R>) -> Result<u16> {
    let mut b = [0; 2];
    reader.hashed(&mut b)?;
    Ok(u16::from_le_bytes(b))
}
fn read_hashed_u32<R: Read>(reader: &mut ScanReader<R>) -> Result<u32> {
    let mut b = [0; 4];
    reader.hashed(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_hashed_u64<R: Read>(reader: &mut ScanReader<R>) -> Result<u64> {
    let mut b = [0; 8];
    reader.hashed(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn read_unhashed_u16<R: Read>(reader: &mut ScanReader<R>) -> Result<u16> {
    let mut b = [0; 2];
    reader.unhashed(&mut b)?;
    Ok(u16::from_le_bytes(b))
}
fn read_unhashed_u32<R: Read>(reader: &mut ScanReader<R>) -> Result<u32> {
    let mut b = [0; 4];
    reader.unhashed(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_unhashed_u64<R: Read>(reader: &mut ScanReader<R>) -> Result<u64> {
    let mut b = [0; 8];
    reader.unhashed(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

struct DecodeReader<R: Read> {
    input: R,
    position: u64,
    hasher: Sha256,
}

impl<R: Read> DecodeReader<R> {
    fn new(input: R) -> Self {
        Self {
            input,
            position: 0,
            hasher: Sha256::new(),
        }
    }
    fn read_exact(&mut self, bytes: &mut [u8]) -> Result<()> {
        read_exact_bounded(&mut self.input, bytes)?;
        self.hasher.update(&*bytes);
        self.position = self
            .position
            .checked_add(bytes.len() as u64)
            .ok_or(TrainingError::ArithmeticOverflow)?;
        Ok(())
    }
    fn digest(&self) -> [u8; 32] {
        self.hasher.clone().finalize().into()
    }
    fn u8(&mut self) -> Result<u8> {
        let mut b = [0];
        self.read_exact(&mut b)?;
        Ok(b[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let mut b = [0; 2];
        self.read_exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    fn u32(&mut self) -> Result<u32> {
        let mut b = [0; 4];
        self.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn u64(&mut self) -> Result<u64> {
        let mut b = [0; 8];
        self.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }
    fn small(&mut self, length: u64, max: usize) -> Result<Vec<u8>> {
        let length = usize::try_from(length).map_err(|_| TrainingError::ArithmeticOverflow)?;
        if length > max {
            return Err(TrainingError::Checkpoint("field_too_large"));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| TrainingError::AllocationFailed)?;
        bytes.resize(length, 0);
        self.read_exact(&mut bytes)?;
        Ok(bytes)
    }
    fn string(&mut self, length: u64, max: usize) -> Result<String> {
        String::from_utf8(self.small(length, max)?)
            .map_err(|_| TrainingError::Checkpoint("invalid_utf8"))
    }
    fn field(&mut self, expected_id: u16) -> Result<u64> {
        let id = self.u16()?;
        let length = self.u64()?;
        if id != expected_id {
            return Err(TrainingError::Checkpoint("field_order"));
        }
        Ok(length)
    }
    fn small_after_field(&mut self, id: u16, max: usize) -> Result<Vec<u8>> {
        let length = self.field(id)?;
        self.small(length, max)
    }
    fn string_after_field(&mut self, id: u16, max: usize) -> Result<String> {
        let length = self.field(id)?;
        self.string(length, max)
    }
    fn section(&mut self, expected: &[u8; 4]) -> Result<u64> {
        let mut tag = [0; 4];
        self.read_exact(&mut tag)?;
        if &tag != expected || self.u32()? != 0 {
            return Err(TrainingError::Checkpoint("section_order_or_flags"));
        }
        let length = self.u64()?;
        self.position
            .checked_add(length)
            .ok_or(TrainingError::ArithmeticOverflow)
    }
}

struct IdentityFields {
    source: String,
    clean: bool,
    config_digest: [u8; 32],
    dataset: String,
    manifest: [u8; 32],
    train: [u8; 32],
    validation: [u8; 32],
    test: [u8; 32],
    provenance: String,
    tokenizer_algorithm: String,
    tokenizer_digest: [u8; 32],
    vocabulary: u32,
    special_digest: [u8; 32],
    run: [u8; 125],
}

struct RunSettings {
    limits: RunLimits,
    memory: crate::TrainingMemoryPlan,
    autograd: ResourceLimits,
    clip: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InventoryRecord {
    name: String,
    shape: Vec<u32>,
}

fn decode_identity<R: Read>(r: &mut DecodeReader<R>) -> Result<IdentityFields> {
    let end = r.section(TAGS[0])?;
    let model = r.string_after_field(1, 32)?;
    if model != MODEL_ID {
        return Err(TrainingError::Checkpoint("model_id_mismatch"));
    }
    let source = r.string_after_field(2, 40)?;
    let clean = match r.small_after_field(3, 1)?.as_slice() {
        [0] => false,
        [1] => true,
        _ => return Err(TrainingError::Checkpoint("invalid_clean_flag")),
    };
    let config_digest = array32(r.small_after_field(4, 32)?)?;
    let dataset = r.string_after_field(5, 128)?;
    let manifest = array32(r.small_after_field(6, 32)?)?;
    let train = array32(r.small_after_field(7, 32)?)?;
    let validation = array32(r.small_after_field(8, 32)?)?;
    let test = array32(r.small_after_field(9, 32)?)?;
    let provenance = r.string_after_field(10, 128)?;
    let tokenizer_algorithm = r.string_after_field(11, 32)?;
    let tokenizer_digest = array32(r.small_after_field(12, 32)?)?;
    let vocabulary = u32::from_le_bytes(
        r.small_after_field(13, 4)?
            .try_into()
            .map_err(|_| TrainingError::Checkpoint("vocab"))?,
    );
    let special_digest = array32(r.small_after_field(14, 32)?)?;
    let run: [u8; 125] = r
        .small_after_field(15, 125)?
        .try_into()
        .map_err(|_| TrainingError::Checkpoint("run_config"))?;
    let policy = array32(r.small_after_field(16, 32)?)?;
    let expected: [u8; 32] = Sha256::digest(run).into();
    if policy != expected || r.position != end {
        return Err(TrainingError::Checkpoint("identity_section_mismatch"));
    }
    Ok(IdentityFields {
        source,
        clean,
        config_digest,
        dataset,
        manifest,
        train,
        validation,
        test,
        provenance,
        tokenizer_algorithm,
        tokenizer_digest,
        vocabulary,
        special_digest,
        run,
    })
}

fn decode_arch<R: Read>(
    r: &mut DecodeReader<R>,
) -> Result<(Config, [u8; 32], Vec<InventoryRecord>)> {
    let end = r.section(TAGS[1])?;
    let dims = r.small_after_field(1, 24)?;
    let mut values = [0_usize; 6];
    for (index, chunk) in dims.as_chunks::<4>().0.iter().enumerate() {
        values[index] = u32::from_le_bytes(*chunk) as usize;
    }
    let config = Config {
        vocabulary: values[0],
        width: values[1],
        layers: values[2],
        heads: values[3],
        ffn: values[4],
        context: values[5],
    };
    let parameter_count = r.u64_after_field(2)?;
    let tensor_count = r.u32_after_field(3)?;
    let digest = array32(r.small_after_field(4, 32)?)?;
    let tied = r.u8_after_field(5)?;
    if parameter_count != 914_400 || tensor_count != 74 || tied != 1 {
        return Err(TrainingError::Checkpoint("inventory_identity"));
    }
    let inventory_length = r.field(6)?;
    let inventory = read_inventory(r, inventory_length)?;
    if r.position != end {
        return Err(TrainingError::Checkpoint("architecture_section_length"));
    }
    if inventory != expected_inventory()? {
        return Err(TrainingError::Checkpoint("inventory_mismatch"));
    }
    if inventory_digest_records(&inventory)? != digest {
        return Err(TrainingError::Checkpoint("inventory_digest_mismatch"));
    }
    Ok((config, digest, inventory))
}

impl<R: Read> DecodeReader<R> {
    fn u8_after_field(&mut self, id: u16) -> Result<u8> {
        if self.field(id)? != 1 {
            return Err(TrainingError::Checkpoint("field_width"));
        }
        self.u8()
    }
    fn u32_after_field(&mut self, id: u16) -> Result<u32> {
        if self.field(id)? != 4 {
            return Err(TrainingError::Checkpoint("field_width"));
        }
        self.u32()
    }
    fn u64_after_field(&mut self, id: u16) -> Result<u64> {
        if self.field(id)? != 8 {
            return Err(TrainingError::Checkpoint("field_width"));
        }
        self.u64()
    }
    fn f32_after_field(&mut self, id: u16) -> Result<f32> {
        if self.field(id)? != 4 {
            return Err(TrainingError::Checkpoint("field_width"));
        }
        self.f32()
    }
    fn f64_after_field(&mut self, id: u16) -> Result<f64> {
        if self.field(id)? != 8 {
            return Err(TrainingError::Checkpoint("field_width"));
        }
        self.f64()
    }
}

fn read_inventory<R: Read>(r: &mut DecodeReader<R>, length: u64) -> Result<Vec<InventoryRecord>> {
    let start = r.position;
    let count = r.u32()?;
    if count != 74 {
        return Err(TrainingError::Checkpoint("inventory_count"));
    }
    let mut records = Vec::new();
    records
        .try_reserve_exact(74)
        .map_err(|_| TrainingError::AllocationFailed)?;
    for expected in 0..74_u32 {
        if r.u32()? != expected {
            return Err(TrainingError::Checkpoint("inventory_order"));
        }
        let name_len = r.u16()? as usize;
        let name = r.string(name_len as u64, 128)?;
        let rank = r.u8()? as usize;
        if !(1..=2).contains(&rank) {
            return Err(TrainingError::Checkpoint("inventory_rank"));
        }
        let mut shape = Vec::new();
        shape
            .try_reserve_exact(rank)
            .map_err(|_| TrainingError::AllocationFailed)?;
        for _ in 0..rank {
            shape.push(r.u32()?);
        }
        records.push(InventoryRecord { name, shape });
    }
    if r.position
        .checked_sub(start)
        .ok_or(TrainingError::ArithmeticOverflow)?
        != length
    {
        return Err(TrainingError::Checkpoint("inventory_length"));
    }
    Ok(records)
}

fn expected_inventory() -> Result<Vec<InventoryRecord>> {
    let mut records = Vec::new();
    records
        .try_reserve_exact(74)
        .map_err(|_| TrainingError::AllocationFailed)?;
    records.push(InventoryRecord {
        name: "embedding".into(),
        shape: vec![292, 96],
    });
    let families = [
        "attention_gain",
        "q",
        "k",
        "v",
        "o",
        "ffn_gain",
        "gate",
        "value",
        "down",
    ];
    for layer in 0..8 {
        for (index, name) in families.iter().enumerate() {
            let shape = match index {
                0 | 5 => vec![96],
                1..=4 => vec![96, 96],
                6 | 7 => vec![96, 256],
                8 => vec![256, 96],
                _ => unreachable!(),
            };
            records.push(InventoryRecord {
                name: format!("block.{layer}.{name}"),
                shape,
            });
        }
    }
    records.push(InventoryRecord {
        name: "final_gain".into(),
        shape: vec![96],
    });
    Ok(records)
}

fn inventory_digest_records(records: &[InventoryRecord]) -> Result<[u8; 32]> {
    let mut hasher = Sha256::new();
    hasher.update(b"XYNOOS-INVENTORY-V1\0");
    for (index, record) in records.iter().enumerate() {
        hasher.update(
            u32::try_from(index)
                .map_err(|_| TrainingError::ArithmeticOverflow)?
                .to_le_bytes(),
        );
        hasher.update(
            u16::try_from(record.name.len())
                .map_err(|_| TrainingError::ArithmeticOverflow)?
                .to_le_bytes(),
        );
        hasher.update(record.name.as_bytes());
        hasher.update([
            u8::try_from(record.shape.len()).map_err(|_| TrainingError::ArithmeticOverflow)?
        ]);
        for &dimension in &record.shape {
            hasher.update(dimension.to_le_bytes());
        }
    }
    Ok(hasher.finalize().into())
}

fn array32(bytes: Vec<u8>) -> Result<[u8; 32]> {
    bytes
        .try_into()
        .map_err(|_| TrainingError::Checkpoint("digest_width"))
}

fn decode_tensor_section<R: Read>(
    r: &mut DecodeReader<R>,
    tag: &[u8; 4],
    field_id: u16,
    inventory: &[InventoryRecord],
    _moments: bool,
) -> Result<Vec<Vec<f32>>> {
    let end = r.section(tag)?;
    if r.u32_after_field(1)? != 74 {
        return Err(TrainingError::Checkpoint("tensor_count"));
    }
    let length = r.field(field_id)?;
    let start = r.position;
    let count = r.u32()?;
    if count != 74 {
        return Err(TrainingError::Checkpoint("tensor_record_count"));
    }
    let mut tensors = Vec::new();
    tensors
        .try_reserve_exact(74)
        .map_err(|_| TrainingError::AllocationFailed)?;
    for index in 0..74_u32 {
        if r.u32()? != index {
            return Err(TrainingError::Checkpoint("tensor_record_order"));
        }
        let elements = r.u32()? as usize;
        let expected = element_count(&inventory[index as usize].shape)?;
        if elements != expected {
            return Err(TrainingError::Checkpoint("tensor_shape_count"));
        }
        tensors.push(read_f32_values(r, elements, false)?);
    }
    if r.position
        .checked_sub(start)
        .ok_or(TrainingError::ArithmeticOverflow)?
        != length
        || r.position != end
    {
        return Err(TrainingError::Checkpoint("tensor_section_length"));
    }
    Ok(tensors)
}

type DecodedAdam = (AdamWConfig, u64, f64, f64, Vec<Vec<f32>>, Vec<Vec<f32>>);

fn decode_adam<R: Read>(
    r: &mut DecodeReader<R>,
    inventory: &[InventoryRecord],
) -> Result<DecodedAdam> {
    let end = r.section(TAGS[3])?;
    if r.u32_after_field(1)? != 1 {
        return Err(TrainingError::Checkpoint("optimizer_version"));
    }
    let beta1 = r.f32_after_field(2)?;
    let beta2 = r.f32_after_field(3)?;
    let epsilon = r.f32_after_field(4)?;
    let decay = r.f32_after_field(5)?;
    let step = r.u64_after_field(6)?;
    let q1 = r.f64_after_field(7)?;
    let q2 = r.f64_after_field(8)?;
    if r.u8_after_field(9)? != 1 {
        return Err(TrainingError::Checkpoint("optimizer_group_policy"));
    }
    let config = AdamWConfig::new(beta1, beta2, epsilon, decay)?;
    let length = r.field(10)?;
    let start = r.position;
    if r.u32()? != 74 {
        return Err(TrainingError::Checkpoint("moment_count"));
    }
    let mut first = Vec::new();
    let mut second = Vec::new();
    first
        .try_reserve_exact(74)
        .map_err(|_| TrainingError::AllocationFailed)?;
    second
        .try_reserve_exact(74)
        .map_err(|_| TrainingError::AllocationFailed)?;
    for index in 0..74_u32 {
        if r.u32()? != index {
            return Err(TrainingError::Checkpoint("moment_order"));
        }
        let elements = r.u32()? as usize;
        if elements != element_count(&inventory[index as usize].shape)? {
            return Err(TrainingError::Checkpoint("moment_shape"));
        }
        first.push(read_f32_values(r, elements, false)?);
        second.push(read_f32_values(r, elements, true)?);
    }
    if r.position
        .checked_sub(start)
        .ok_or(TrainingError::ArithmeticOverflow)?
        != length
        || r.position != end
    {
        return Err(TrainingError::Checkpoint("adam_section_length"));
    }
    Ok((config, step, q1, q2, first, second))
}

fn decode_schedule<R: Read>(r: &mut DecodeReader<R>) -> Result<(LinearWarmupDecay, u64)> {
    let end = r.section(TAGS[4])?;
    if r.u8_after_field(1)? != 1 || r.u32_after_field(2)? != 1 {
        return Err(TrainingError::Checkpoint("schedule_version"));
    }
    let total = r.u64_after_field(3)?;
    let warmup = r.u64_after_field(4)?;
    let peak = r.f32_after_field(5)?;
    let final_rate = r.f32_after_field(6)?;
    let completed = r.u64_after_field(7)?;
    if r.position != end || completed > total {
        return Err(TrainingError::Checkpoint("schedule_state"));
    }
    Ok((
        LinearWarmupDecay::new(total, warmup, peak, final_rate)?,
        completed,
    ))
}

fn decode_rng<R: Read>(r: &mut DecodeReader<R>) -> Result<(u64, u64, u64)> {
    let end = r.section(TAGS[5])?;
    if r.u32_after_field(1)? != 1 {
        return Err(TrainingError::Checkpoint("init_rng_id"));
    }
    let seed = r.u64_after_field(2)?;
    if r.u32_after_field(3)? != 1 {
        return Err(TrainingError::Checkpoint("order_rng_id"));
    }
    let state = r.small_after_field(4, 16)?;
    if r.position != end {
        return Err(TrainingError::Checkpoint("rng_section_length"));
    }
    let order_state = u64::from_le_bytes(
        state[..8]
            .try_into()
            .map_err(|_| TrainingError::Checkpoint("rng_state"))?,
    );
    let draws = u64::from_le_bytes(
        state[8..]
            .try_into()
            .map_err(|_| TrainingError::Checkpoint("rng_state"))?,
    );
    Ok((seed, order_state, draws))
}

type DecodedData = ([u8; 32], [u8; 32], u64, [u8; 32], u8);

fn decode_data(r: &mut DecodeReader<impl Read>) -> Result<DecodedData> {
    let end = r.section(TAGS[6])?;
    let order = array32(r.small_after_field(1, 32)?)?;
    let epoch = r.u64_after_field(2)?;
    let document = r.u64_after_field(3)?;
    let window = r.u64_after_field(4)?;
    let cursor = r.u64_after_field(5)?;
    let tail = r.u8_after_field(6)?;
    let next = array32(r.small_after_field(7, 32)?)?;
    let validation = array32(r.small_after_field(8, 32)?)?;
    if r.position != end || epoch != 0 || document != 0 || window != 0 || tail > 1 {
        return Err(TrainingError::Checkpoint("data_cursor"));
    }
    Ok((order, validation, cursor, next, tail))
}

fn decode_counters<R: Read>(r: &mut DecodeReader<R>) -> Result<CheckpointCounters> {
    let end = r.section(TAGS[7])?;
    let values = [
        r.u64_after_field(1)?,
        r.u64_after_field(2)?,
        r.u64_after_field(3)?,
        r.u64_after_field(4)?,
        r.u64_after_field(5)?,
        r.u64_after_field(6)?,
        r.u64_after_field(7)?,
        r.u64_after_field(8)?,
        r.u64_after_field(9)?,
        r.u64_after_field(10)?,
        r.u64_after_field(11)?,
        r.u64_after_field(12)?,
    ];
    if r.position != end {
        return Err(TrainingError::Checkpoint("counter_section_length"));
    }
    Ok(CheckpointCounters {
        committed_consumed_tokens: values[0],
        committed_supervised_tokens: values[1],
        validation_supervised_tokens: values[2],
        attempted_consumed_tokens: values[3],
        discarded_consumed_tokens: values[4],
        cumulative_work: values[5],
        cumulative_runtime_ns: values[6],
        completed_updates: values[7],
        last_metric_sequence: values[8],
        checkpoint_sequence: values[9],
        attempted_updates: values[10],
        attempted_supervised_tokens: values[11],
        committed_cursor: 0,
    })
}

fn decode_accumulation<R: Read>(r: &mut DecodeReader<R>) -> Result<()> {
    let end = r.section(TAGS[8])?;
    if r.u8_after_field(1)? != 1 || r.u64_after_field(2)? != 0 || r.position != end {
        return Err(TrainingError::Checkpoint("partial_accumulation"));
    }
    Ok(())
}

fn read_f32_values<R: Read>(
    r: &mut DecodeReader<R>,
    count: usize,
    nonnegative: bool,
) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| TrainingError::AllocationFailed)?;
    let mut remaining = count;
    let mut bytes = [0_u8; IO_BUFFER];
    while remaining > 0 {
        let cells = remaining.min(IO_BUFFER / 4);
        let length = cells * 4;
        r.read_exact(&mut bytes[..length])?;
        for chunk in bytes[..length].as_chunks::<4>().0 {
            let value = f32::from_bits(u32::from_le_bytes(*chunk));
            if !value.is_finite() || (nonnegative && value < 0.0) {
                return Err(TrainingError::Checkpoint("non_finite_or_negative_moment"));
            }
            values.push(value);
        }
        remaining -= cells;
    }
    Ok(values)
}

fn element_count(shape: &[u32]) -> Result<usize> {
    shape.iter().try_fold(1_usize, |count, &dimension| {
        count
            .checked_mul(dimension as usize)
            .ok_or(TrainingError::ArithmeticOverflow)
    })
}

fn parse_run_config(bytes: &[u8; 125]) -> Result<RunSettings> {
    fn get_u32(b: &[u8], i: usize) -> u32 {
        u32::from_le_bytes(b[i..i + 4].try_into().expect("fixed RUN1 field"))
    }
    fn get_u64(b: &[u8], i: usize) -> u64 {
        u64::from_le_bytes(b[i..i + 8].try_into().expect("fixed RUN1 field"))
    }
    if get_u32(bytes, 0) != 1 {
        return Err(TrainingError::Checkpoint("run_config_version"));
    }
    let micro =
        usize::try_from(get_u32(bytes, 4)).map_err(|_| TrainingError::ArithmeticOverflow)?;
    let mut values = [0_u64; 14];
    for (index, value) in values.iter_mut().enumerate() {
        *value = get_u64(bytes, 8 + index * 8);
    }
    let flag = bytes[120];
    let clip_value = f32::from_bits(get_u32(bytes, 121));
    let clip = match flag {
        0 if clip_value.to_bits() == 0 => None,
        1 if clip_value.is_finite() && clip_value > 0.0 => Some(clip_value),
        _ => return Err(TrainingError::Checkpoint("clip_config")),
    };
    let limits = RunLimits::new(
        micro,
        values[0],
        values[1],
        values[2],
        values[3],
        usize::try_from(values[4]).map_err(|_| TrainingError::ArithmeticOverflow)?,
        std::time::Duration::from_nanos(values[5]),
        std::time::Duration::from_nanos(values[6]),
        values[7],
        values[8],
    )?;
    let data = usize::try_from(values[9]).map_err(|_| TrainingError::ArithmeticOverflow)?;
    let memory = crate::TrainingMemoryPlan::candidate(data, limits.owned_memory_bytes())?;
    let bytes = usize::try_from(values[10]).map_err(|_| TrainingError::ArithmeticOverflow)?;
    let vals = usize::try_from(values[11]).map_err(|_| TrainingError::ArithmeticOverflow)?;
    let operations = usize::try_from(values[12]).map_err(|_| TrainingError::ArithmeticOverflow)?;
    let work = usize::try_from(values[13]).map_err(|_| TrainingError::ArithmeticOverflow)?;
    let autograd = ResourceLimits::new(bytes, vals, operations, work);
    if autograd.max_bytes() != bytes
        || autograd.max_values() != vals
        || autograd.max_operations() != operations
        || autograd.max_work() != work
    {
        return Err(TrainingError::Checkpoint("resource_limit_configuration"));
    }
    Ok(RunSettings {
        limits,
        memory,
        autograd,
        clip,
    })
}

fn decode_file(path: &Path, expected_digest: [u8; 32]) -> Result<CheckpointData> {
    let file = File::open(path)?;
    let length = file.metadata()?.len();
    let mut r = DecodeReader::new(BufReader::with_capacity(IO_BUFFER, file));
    let mut header = [0_u8; HEADER_BYTES];
    r.read_exact(&mut header)?;
    validate_header(&header, length)?;

    let identity_fields = decode_identity(&mut r)?;
    let (model, inventory_digest, inventory) = decode_arch(&mut r)?;
    if model != Config::RESEARCH_CANDIDATE {
        return Err(TrainingError::Checkpoint("architecture_mismatch"));
    }
    let parameters = decode_tensor_section(&mut r, TAGS[2], 2, &inventory, false)?;
    let (optimizer, optimizer_step, beta1_power, beta2_power, first_moments, second_moments) =
        decode_adam(&mut r, &inventory)?;
    let (schedule, schedule_completed) = decode_schedule(&mut r)?;
    let (seed, order_state, order_draw_count) = decode_rng(&mut r)?;
    let (data_order, validation_order, cursor, next_batch_identity, _tail_policy) =
        decode_data(&mut r)?;
    let mut counters = decode_counters(&mut r)?;
    counters.committed_cursor = cursor;
    decode_accumulation(&mut r)?;
    if r.digest() != expected_digest {
        return Err(TrainingError::Checkpoint("changed_during_decode"));
    }

    if schedule_completed != optimizer_step
        || schedule.total_updates() < optimizer_step
        || order_state != seed
        || order_draw_count != 0
        || next_batch_identity != next_batch_identity_for(&data_order, cursor)
        || counters.completed_updates != optimizer_step
        || counters.committed_supervised_tokens > counters.committed_consumed_tokens
        || counters.attempted_consumed_tokens < counters.committed_consumed_tokens
        || counters.attempted_supervised_tokens < counters.committed_supervised_tokens
        || counters.discarded_consumed_tokens
            > counters
                .attempted_consumed_tokens
                .saturating_sub(counters.committed_consumed_tokens)
        || counters.attempted_updates < counters.completed_updates
    {
        return Err(TrainingError::Checkpoint("counter_or_resume_state"));
    }
    let mut q1 = 1.0_f64;
    let mut q2 = 1.0_f64;
    for _ in 0..optimizer_step {
        q1 *= f64::from(optimizer.beta1());
        q2 *= f64::from(optimizer.beta2());
    }
    if !beta1_power.is_finite()
        || !beta2_power.is_finite()
        || beta1_power.to_bits() != q1.to_bits()
        || beta2_power.to_bits() != q2.to_bits()
    {
        return Err(TrainingError::Checkpoint("bias_power_mismatch"));
    }

    let settings = parse_run_config(&identity_fields.run)?;
    if settings.limits.max_updates() < optimizer_step {
        return Err(TrainingError::Checkpoint("run_update_limit"));
    }
    let config = TrainingConfig::new(
        optimizer,
        schedule,
        settings.limits,
        settings.memory,
        settings.autograd,
        settings.clip,
        seed,
    )?;
    let identity = RunIdentity::new(
        identity_fields.source,
        identity_fields.clean,
        identity_fields.dataset,
        identity_fields.manifest,
        identity_fields.train,
        identity_fields.validation,
        identity_fields.test,
        identity_fields.provenance,
        identity_fields.tokenizer_algorithm,
        identity_fields.tokenizer_digest,
        identity_fields.vocabulary,
        data_order,
        validation_order,
    );
    if identity.special_registry_sha256() != identity_fields.special_digest
        || config.digest(&identity) != identity_fields.config_digest
        || config.optimizer() != optimizer
        || inventory_digest != inventory_digest_records(&inventory)?
    {
        return Err(TrainingError::Checkpoint(
            "configuration_or_inventory_digest",
        ));
    }
    Ok(CheckpointData {
        identity,
        config,
        parameters,
        first_moments,
        second_moments,
        optimizer_step,
        beta1_power,
        beta2_power,
        cursor,
        next_batch_identity,
        order_state,
        order_draw_count,
        counters,
        digest: expected_digest,
        configuration_digest: identity_fields.config_digest,
    })
}

fn next_batch_identity_for(order: &[u8; 32], cursor: u64) -> [u8; 32] {
    next_batch_identity(order, cursor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TrainingMemoryPlan;
    use std::time::Duration;
    use xynoos_autograd::ResourceLimits;
    use xynoos_tokenizer::{TokenizerArtifact, TokenizerLimits};

    const DESIGN_HEAD: &str = "26777fa95383600c003ff094216d3d1c310b051c";

    fn root() -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "xyckpt-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn tokenizer() -> Tokenizer {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../xynoos-data/tests/fixtures/task004.json"
        ))
        .unwrap();
        TokenizerArtifact::read(
            fixture["artifact"].as_str().unwrap().as_bytes(),
            TokenizerLimits::MAX,
        )
        .unwrap()
    }

    fn identity(tokenizer: &Tokenizer) -> RunIdentity {
        let artifact = TokenizerArtifact::serialize(tokenizer, TokenizerLimits::MAX).unwrap();
        RunIdentity::new(
            DESIGN_HEAD.to_owned(),
            true,
            "task006-objective-a-v1".to_owned(),
            [0x10; 32],
            [0x11; 32],
            [0x12; 32],
            [0x13; 32],
            "objective-a-fixture-v1".to_owned(),
            tokenizer.algorithm().to_owned(),
            Sha256::digest(artifact).into(),
            292,
            [0x14; 32],
            [0x15; 32],
        )
    }

    fn config() -> TrainingConfig {
        let limits = RunLimits::new(
            1,
            2,
            10_000,
            10_000,
            2_000_000_000,
            ResourceLimits::MAX_BYTES,
            Duration::from_secs(600),
            Duration::from_secs(300),
            1,
            1,
        )
        .unwrap();
        TrainingConfig::new(
            AdamWConfig::new(0.9, 0.999, 1e-8, 0.01).unwrap(),
            LinearWarmupDecay::new(2, 0, 1e-4, 1e-5).unwrap(),
            limits,
            TrainingMemoryPlan::candidate(4 * 1024 * 1024, ResourceLimits::MAX_BYTES).unwrap(),
            ResourceLimits::default(),
            Some(1.0),
            0x006b,
        )
        .unwrap()
    }

    fn snapshot<'a>(
        identity: &'a RunIdentity,
        config: TrainingConfig,
        model: &'a Transformer,
        optimizer: &'a AdamW,
        checkpoint_sequence: u64,
    ) -> CheckpointSnapshot<'a> {
        CheckpointSnapshot::new(
            identity,
            config,
            model,
            optimizer,
            CheckpointCounters {
                checkpoint_sequence,
                ..CheckpointCounters::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn streams_round_trip_and_reject_corruption_without_mutating_a_model() {
        let tokenizer = tokenizer();
        let identity = identity(&tokenizer);
        let config = config();
        config.validate(&identity, &tokenizer).unwrap();
        let model = Transformer::new(
            Config::RESEARCH_CANDIDATE,
            &tokenizer,
            config.seed(),
            config.autograd_limits(),
        )
        .unwrap();
        let optimizer = AdamW::new(model.parameters(), config.optimizer()).unwrap();
        let path = root();
        let snap = snapshot(&identity, config, &model, &optimizer, 0);
        let publication = CheckpointCodec::write_atomic(&path, &snap).unwrap();
        assert!(publication.bytes <= FILE_LIMIT as u64);
        assert!(publication.durable);
        let loaded = CheckpointCodec::load(&path, &identity, config, &tokenizer).unwrap();
        assert_eq!(loaded.config, config);
        assert_eq!(loaded.optimizer_step, 0);
        assert_eq!(loaded.cursor, 0);
        assert_eq!(loaded.digest, publication.sha256);
        assert_eq!(loaded.parameters.len(), 74);
        for ((_, id), values) in model.inventory().zip(&loaded.parameters) {
            assert_eq!(model.parameters().value(id).unwrap().data(), values);
        }
        assert!(
            loaded
                .first_moments
                .iter()
                .flatten()
                .all(|value| *value == 0.0)
        );
        assert!(
            loaded
                .second_moments
                .iter()
                .flatten()
                .all(|value| *value == 0.0)
        );

        let file = path.join("checkpoint.xyckpt");
        let mut bytes = fs::read(&file).unwrap();
        let corruption = bytes.len() - 59;
        bytes[corruption] ^= 0x20;
        fs::write(&file, bytes).unwrap();
        assert_eq!(
            CheckpointCodec::load(&path, &identity, config, &tokenizer),
            Err(TrainingError::Checkpoint("checksum_mismatch"))
        );
        assert_eq!(optimizer.step_count(), 0);
        assert_eq!(model.parameters().revision(), 74);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn publication_failures_distinguish_pre_rename_from_post_rename() {
        let tokenizer = tokenizer();
        let identity = identity(&tokenizer);
        let config = config();
        let model = Transformer::new(
            Config::RESEARCH_CANDIDATE,
            &tokenizer,
            config.seed(),
            config.autograd_limits(),
        )
        .unwrap();
        let optimizer = AdamW::new(model.parameters(), config.optimizer()).unwrap();
        let path = root();
        let old = snapshot(&identity, config, &model, &optimizer, 0);
        CheckpointCodec::write_atomic(&path, &old).unwrap();
        let newer = snapshot(&identity, config, &model, &optimizer, 1);
        assert_eq!(
            CheckpointCodec::write_atomic_impl(&path, &newer, true, false),
            Err(TrainingError::Io(std::io::ErrorKind::Other))
        );
        let still_old = CheckpointCodec::load(&path, &identity, config, &tokenizer).unwrap();
        assert_eq!(still_old.counters.checkpoint_sequence, 0);
        assert_eq!(
            CheckpointCodec::write_atomic_impl(&path, &newer, false, true),
            Err(TrainingError::PublicationDurabilityUncertain)
        );
        let already_published =
            CheckpointCodec::load(&path, &identity, config, &tokenizer).unwrap();
        assert_eq!(already_published.counters.checkpoint_sequence, 1);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn rejects_noncanonical_fields_truncation_oversize_and_symlink_artifacts() {
        let tokenizer = tokenizer();
        let identity = identity(&tokenizer);
        let config = config();
        let model = Transformer::new(
            Config::RESEARCH_CANDIDATE,
            &tokenizer,
            config.seed(),
            config.autograd_limits(),
        )
        .unwrap();
        let optimizer = AdamW::new(model.parameters(), config.optimizer()).unwrap();
        let path = root();
        CheckpointCodec::write_atomic(&path, &snapshot(&identity, config, &model, &optimizer, 0))
            .unwrap();
        let checkpoint = path.join("checkpoint.xyckpt");
        let canonical = fs::read(&checkpoint).unwrap();

        let mut unknown_section = canonical.clone();
        unknown_section[40..44].copy_from_slice(b"UNKN");
        fs::write(&checkpoint, unknown_section).unwrap();
        assert_eq!(
            CheckpointCodec::load(&path, &identity, config, &tokenizer),
            Err(TrainingError::Checkpoint("invalid_section_order_or_flags"))
        );

        let mut duplicate_field = canonical.clone();
        // IDEN field 2 follows the fixed-width 16-byte model ID.
        duplicate_field[82..84].copy_from_slice(&1_u16.to_le_bytes());
        fs::write(&checkpoint, duplicate_field).unwrap();
        assert_eq!(
            CheckpointCodec::load(&path, &identity, config, &tokenizer),
            Err(TrainingError::Checkpoint("invalid_field_id_or_length"))
        );

        fs::write(&checkpoint, &canonical[..canonical.len() - 1]).unwrap();
        assert!(CheckpointCodec::load(&path, &identity, config, &tokenizer).is_err());

        let file = OpenOptions::new().write(true).open(&checkpoint).unwrap();
        file.set_len((FILE_LIMIT + 1) as u64).unwrap();
        assert_eq!(
            CheckpointCodec::load(&path, &identity, config, &tokenizer),
            Err(TrainingError::Checkpoint("invalid_file_size_or_type"))
        );
        fs::remove_file(&checkpoint).unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let target = path.join("target");
            fs::write(&target, b"not a checkpoint").unwrap();
            symlink(&target, &checkpoint).unwrap();
            assert_eq!(
                CheckpointCodec::load(&path, &identity, config, &tokenizer),
                Err(TrainingError::InvalidConfiguration)
            );
        }
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn auditor_semantic_mutation_with_recomputed_integrity_is_rejected() {
        let tokenizer = tokenizer();
        let identity = identity(&tokenizer);
        let config = config();
        let model = Transformer::new(
            Config::RESEARCH_CANDIDATE,
            &tokenizer,
            config.seed(),
            config.autograd_limits(),
        )
        .unwrap();
        let optimizer = AdamW::new(model.parameters(), config.optimizer()).unwrap();
        let path = root();
        CheckpointCodec::write_atomic(&path, &snapshot(&identity, config, &model, &optimizer, 0))
            .unwrap();
        let checkpoint = path.join("checkpoint.xyckpt");
        let mut bytes = fs::read(&checkpoint).unwrap();

        // Independently walk the documented section/field/container framing.
        let mut section = HEADER_BYTES;
        let mut adam_start = None;
        let mut hash_start = None;
        while section < bytes.len() {
            let tag = &bytes[section..section + 4];
            let payload_len =
                u64::from_le_bytes(bytes[section + 8..section + 16].try_into().unwrap()) as usize;
            if tag == b"ADAM" {
                adam_start = Some(section + 16);
            }
            if tag == b"HASH" {
                hash_start = Some(section);
                break;
            }
            section += 16 + payload_len;
        }
        let mut field = adam_start.unwrap();
        let second_moment_cell = loop {
            let id = u16::from_le_bytes(bytes[field..field + 2].try_into().unwrap());
            let length =
                u64::from_le_bytes(bytes[field + 2..field + 10].try_into().unwrap()) as usize;
            let value = field + 10;
            if id == 10 {
                let mut record = value + 4; // tensor count
                let elements =
                    u32::from_le_bytes(bytes[record + 4..record + 8].try_into().unwrap()) as usize;
                record += 8; // first tensor's index and element count
                break record + elements * 4; // second-moment values follow first-moment values
            }
            field = value + length;
        };
        bytes[second_moment_cell..second_moment_cell + 4]
            .copy_from_slice(&(-1.0_f32).to_bits().to_le_bytes());

        let hash_start = hash_start.unwrap();
        let digest: [u8; 32] = Sha256::digest(&bytes[..hash_start]).into();
        let hash_value = hash_start + 16 + 10; // section header, field id, field length
        bytes[hash_value..hash_value + 32].copy_from_slice(&digest);
        fs::write(&checkpoint, bytes).unwrap();
        assert_eq!(
            CheckpointCodec::load(&path, &identity, config, &tokenizer),
            Err(TrainingError::Checkpoint("non_finite_or_negative_moment"))
        );
        assert_eq!(model.parameters().revision(), 74);
        assert_eq!(optimizer.step_count(), 0);
        fs::remove_dir_all(path).unwrap();
    }
}
